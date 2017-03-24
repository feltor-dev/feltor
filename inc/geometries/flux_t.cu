#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"

#include "dg/backend/timer.cuh"
#include "orthogonal.h"
#include "curvilinear.h"
//#include "guenther.h"
#include "solovev.h"
#include "flux.h"
#include "dg/ds.h"
#include "init.h"

#include "file/nc_utilities.h"

using namespace dg::geo::solovev;

thrust::host_vector<double> periodify( const thrust::host_vector<double>& in, const dg::Grid2d& g)
{
    thrust::host_vector<double> out(g.size());
    for( unsigned i=0; i<g.Ny()-1; i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n()+l] = 
            in[((i*g.n() + k)*g.Nx() + j)*g.n()+l];
    for( unsigned i=g.Ny()-1; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n()+l] = 
            in[((0*g.n() + k)*g.Nx() + j)*g.n()+l];
    return out;
}

double sineX( double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY( double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}
typedef dg::FieldAligned< dg::CurvilinearGrid3d<dg::HVec> , dg::IHMatrix, dg::HVec> DFA;
//typedef dg::FieldAligned< OrthogonalGrid3d<dg::HVec> , dg::IHMatrix, dg::HVec> DFA;

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    Json::Reader reader;
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint.js");
        reader.parse(is,js,false);
    }
    else
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    //write parameters from file into variables
    GeomParameters gp(js);
    Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    std::cout << "Constructing flux grid ... \n";
    t.tic();
    MagneticField c( gp);
    dg::geo::FluxGenerator<Psip, PsipR, PsipZ, PsipRR, PsipRZ, PsipZZ, Ipol, IpolR, IpolZ>
        flux( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, c.ipol, c.ipolR, c.ipolZ, psi_0, psi_1, gp.R_0, 0., 1);
    dg::CurvilinearGrid3d<dg::HVec> g3d(flux, n, Nx, Ny,Nz, dg::DIR);
    dg::CurvilinearGrid2d<dg::HVec> g2d = g3d.perp_grid();
    //OrthogonalGrid3d<dg::HVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    //OrthogonalGrid2d<dg::HVec> g2d = g3d.perp_grid();
    dg::Grid2d g2d_periodic(g2d.x0(), g2d.x1(), g2d.y0(), g2d.y1(), g2d.n(), g2d.Nx(), g2d.Ny()+1); 
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "flux.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[2];
    err = file::define_dimensions(  ncid, dim3d, g2d_periodic);
    int coordsID[2], onesID, defID, confID, volID, divBID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "d",    NC_DOUBLE, 2, dim3d, &defID);
    err = nc_def_var( ncid, "R", NC_DOUBLE, 2, dim3d, &confID);
    err = nc_def_var( ncid, "vol", NC_DOUBLE, 2, dim3d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim3d, &divBID);

    thrust::host_vector<double> psi_p = dg::pullback( psip, g2d);
    //g.display();
    err = nc_put_var_double( ncid, onesID, periodify(psi_p, g2d_periodic).data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.r()[i];
        Y[i] = g2d.z()[i];
    }

    dg::HVec temp0( g2d.size()), temp1(temp0);
    dg::HVec w3d = dg::create::weights( g2d);

    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g2d_periodic).data());
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());

    //compute and write deformation into netcdf
    dg::blas1::pointwiseDivide( g2d.g_xy(), g2d.g_xx(), temp0);
    const dg::HVec ones = dg::evaluate( dg::one, g2d);
    X=g2d.g_yy();
    err = nc_put_var_double( ncid, defID, periodify(X, g2d_periodic).data());
    //compute and write conformalratio into netcdf
    dg::blas1::pointwiseDivide( g2d.g_yy(), g2d.g_xx(), temp0);
    X=g2d.g_xx();
    err = nc_put_var_double( ncid, confID, periodify(X, g2d_periodic).data());

    std::cout << "Construction successful!\n";

    ////compute error in volume element
    //const dg::HVec f_ = g2d.f();
    //dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    //dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    //dg::blas1::axpby( 1., temp0, -1., temp1, temp0); //temp0=1/g = g^xx g^yy - g^xy^2
    //solovev::flux::FieldY fieldY(gp);
    //dg::HVec fby = dg::pullback( fieldY, g2d);//?
    //dg::blas1::pointwiseDot( f_, fby,fby);
//  //       dg::blas1::scal( fby, 2.*M_PI);

    //dg::blas1::pointwiseDot( fby, fby, temp1);
    //dg::blas1::axpby( 1., temp1, -1., temp0, temp0); ////temp0= g_xx g_yy - g_xy^2 - g
    //double error = sqrt( dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( temp1, w3d, temp1));
    //std::cout<< "Rel Error in Determinant is "<<error<<"\n";

    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>()); //temp0=1/sqrt(g) = sqrt(g^xx g^yy - g^xy^2)
    dg::blas1::pointwiseDivide( ones, temp0, temp0); //temp0=sqrt(g)
    X=temp0;
    err = nc_put_var_double( ncid, volID, periodify(X, g2d_periodic).data());
    dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0); //temp0 = sqrt(g)-vol
    double error = sqrt(dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( g2d.vol(), w3d, g2d.vol()));
    std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //dg::blas1::pointwiseDivide(ones,fby,temp1); //=sqrt(g)
    //dg::blas1::axpby( 1., temp1, -1., g2d.vol(), temp0);
    //error=sqrt(dg::blas2::dot( temp0, w3d, temp0))/sqrt( dg::blas2::dot(g2d.vol(), w3d, g2d.vol()));
    //std::cout << "Rel Error of volume form is "<<error<<"\n";

    const dg::HVec vol = dg::create::volume( g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    double volume = dg::blas1::dot( vol, ones3d);

    std::cout << "TEST VOLUME IS:\n";
    if( psi_0 < psi_1) gp.psipmax = psi_1, gp.psipmin = psi_0;
    else               gp.psipmax = psi_0, gp.psipmin = psi_1;
    dg::geo::Iris<Psip> iris( c.psip, gp.psipmin, gp.psipmax);
    //dg::CylindricalGrid3d<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
    dg::CartesianGrid2d g2dC( gp.R_0 -2.0*gp.a, gp.R_0 + 2.0*gp.a, -2.0*gp.a,2.0*gp.a,1, 2e3, 2e3, dg::PER, dg::PER);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec R  = dg::evaluate( dg::cooX2d, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    ///////////////////////////TEST 3d grid//////////////////////////////////////
    //std::cout << "Start DS test!"<<std::endl;
    //const dg::HVec vol3d = dg::create::volume( g3d);
    //t.tic();
    //DFA fieldaligned( flux::Field( gp, g3d.x(), g3d.f_x()), g3d, gp.rk4eps, dg::NoLimiter()); 
    //dg::DS<DFA, dg::DMatrix, dg::HVec> ds( fieldaligned, flux::Field(gp, g3d.x(), g3d.f_x()), dg::normed, dg::centered);

    //
    //t.toc();
    //std::cout << "Construction took "<<t.diff()<<"s\n";
    //dg::HVec B = dg::pullback( solovev::InvB(gp), g3d), divB(B);
    //dg::HVec lnB = dg::pullback( solovev::LnB(gp), g3d), gradB(B);
    //dg::HVec gradLnB = dg::pullback( solovev::GradLnB(gp), g3d);
    //dg::blas1::pointwiseDivide( ones3d, B, B);
    //dg::HVec function = dg::pullback( solovev::FuncNeu(gp), g3d), derivative(function);
    //ds( function, derivative);

    //ds.centeredT( B, divB);
    //double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    //std::cout << "Divergence of B is "<<norm<<"\n";

    //ds.centered( lnB, gradB);
    //std::cout << "num. norm of gradLnB is "<<sqrt( dg::blas2::dot( gradB,vol3d, gradB))<<"\n";
    //norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    //std::cout << "ana. norm of gradLnB is "<<norm<<"\n";
    //dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    //X = divB;
    //err = nc_put_var_double( ncid, divBID, periodify(X, g2d_periodic).data());
    //double norm2 = sqrt(dg::blas2::dot(gradLnB, vol3d,gradLnB));
    //std::cout << "rel. error of lnB is    "<<norm2/norm<<"\n";
    err = nc_close( ncid);



    return 0;
}
