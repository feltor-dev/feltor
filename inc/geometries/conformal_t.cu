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
//#include "guenther.h"
#include "solovev.h"
#include "conformal.h"
#include "orthogonal.h"
#include "dg/ds.h"
#include "init.h"

#include "file/nc_utilities.h"

thrust::host_vector<double> periodify( const thrust::host_vector<double>& in, const dg::Grid2d<double>& g)
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
typedef dg::FieldAligned< conformal::RingGrid3d<dg::HVec> , dg::IHMatrix, dg::HVec> DFA;
//typedef dg::FieldAligned< orthogonal::RingGrid3d<dg::HVec> , dg::IHMatrix, dg::HVec> DFA;

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params_Xpoint.txt"); 
        }
        else
        {
            v = file::read_input( argv[1]); 
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    solovev::GeomParameters gp(v);
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    std::cout << "Constructing conformal grid ... \n";
    t.tic();
    conformal::RingGrid3d<dg::HVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    conformal::RingGrid2d<dg::HVec> g2d = g3d.perp_grid();
    //orthogonal::RingGrid3d<dg::HVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    //orthogonal::RingGrid2d<dg::HVec> g2d = g3d.perp_grid();
    dg::Grid2d<double> g2d_periodic(g2d.x0(), g2d.x1(), g2d.y0(), g2d.y1(), g2d.n(), g2d.Nx(), g2d.Ny()+1); 
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[2];
    err = file::define_dimensions(  ncid, dim3d, g2d_periodic);
    int coordsID[2], onesID, defID, divBID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);
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

    //dg::blas1::pointwiseDivide( g2d.g_xy(), g2d.g_xx(), temp0);
    dg::blas1::pointwiseDivide( g2d.g_yy(), g2d.g_xx(), temp0);
    const dg::HVec ones = dg::evaluate( dg::one, g2d);
    dg::blas1::axpby( 1., ones, -1., temp0, temp0);
    X=temp0;
    err = nc_put_var_double( ncid, defID, periodify(X, g2d_periodic).data());

    std::cout << "Construction successful!\n";

    //compute error in volume element
    //const dg::HVec f_ = g2d.f1();
    const dg::HVec f_ = g2d.f();
    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    //dg::blas1::pointwiseDot( f_, f_, temp1);
    temp1 = ones;
    dg::blas1::axpby( 0.0, temp1, 1.0, g2d.g_xx(),  temp1);
    dg::blas1::pointwiseDot( temp1, temp1, temp1);
    dg::blas1::axpby( 1., temp1, -1., temp0, temp0);
    double error = sqrt( dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( temp1, w3d, temp1));
    std::cout<< "Rel Error in Determinant is "<<error<<"\n";

    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::pointwiseDot( temp0, g.g_pp(), temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0);
    error = sqrt(dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( g2d.vol(), w3d, g2d.vol()));
    std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //temp0=g.r();
    //dg::blas1::pointwiseDivide( temp0, g.g_xx(), temp0);
    dg::blas1::pointwiseDot( f_, f_, temp0);
    dg::blas1::axpby( 0.0,temp0 , 1.0, g2d.g_xx(), temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    //dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0);
    dg::blas1::axpby( 1., ones, -1., g2d.vol(), temp0);
    error=sqrt(dg::blas2::dot( temp0, w3d, temp0))/sqrt( dg::blas2::dot(g2d.vol(), w3d, g2d.vol()));
    std::cout << "Rel Error of volume form is "<<error<<"\n";

    solovev::conformal::FieldY fieldY(gp);
    //solovev::ConformalField fieldY(gp);
    dg::HVec fby = dg::pullback( fieldY, g2d);
    dg::blas1::pointwiseDot( fby, f_, fby);
    dg::blas1::pointwiseDot( fby, f_, fby);
    //for( unsigned k=0; k<Nz; k++)
        //for( unsigned i=0; i<n*Ny; i++)
        //    for( unsigned j=0; j<n*Nx; j++)
        //        //by[k*n*n*Nx*Ny + i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
        //        fby[i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
    //dg::HVec fby_device = fby;
    dg::blas1::scal( fby, 1./gp.R_0);
    temp0=g2d.r();
    dg::blas1::pointwiseDot( temp0, fby, fby);
    dg::blas1::pointwiseDivide( ones, g2d.vol(), temp0);
    dg::blas1::axpby( 1., temp0, -1., fby, temp1);
    error= dg::blas2::dot( temp1, w3d, temp1)/dg::blas2::dot(fby,w3d,fby);
    std::cout << "Rel Error of g.g_xx() is "<<sqrt(error)<<"\n";
    const dg::HVec vol = dg::create::volume( g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    double volume = dg::blas1::dot( vol, ones3d);

    std::cout << "TEST VOLUME IS:\n";
    if( psi_0 < psi_1) gp.psipmax = psi_1, gp.psipmin = psi_0;
    else               gp.psipmax = psi_0, gp.psipmin = psi_1;
    solovev::Iris iris( gp);
    //dg::CylindricalGrid<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
    dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, -1.2*gp.a, 1.2*gp.a, 1, 1e3, 1e3, dg::PER, dg::PER);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec R  = dg::evaluate( dg::coo1, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    /////////////////////////TEST 3d grid//////////////////////////////////////
    std::cout << "Start DS test!"<<std::endl;
    const dg::HVec vol3d = dg::create::volume( g3d);
    t.tic();
    DFA fieldaligned( conformal::Field( gp, g3d.x(), g3d.f_x()), g3d, gp.rk4eps, dg::NoLimiter()); 
    //DFA fieldaligned( orthogonal::Field( gp, g2d, g2d.f2_xy()), g3d, gp.rk4eps, dg::NoLimiter()); 

    dg::DS<DFA, dg::DMatrix, dg::HVec> ds( fieldaligned, conformal::Field(gp, g3d.x(), g3d.f_x()), dg::normed, dg::centered);
    //dg::DS<DFA, dg::DMatrix, dg::HVec> ds( fieldaligned, orthogonal::Field(gp, g2d, g2d.f2_xy()), dg::normed, dg::centered);
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    dg::HVec B = dg::pullback( solovev::InvB(gp), g3d), divB(B);
    dg::HVec lnB = dg::pullback( solovev::LnB(gp), g3d), gradB(B);
    dg::HVec gradLnB = dg::pullback( solovev::GradLnB(gp), g3d);
    dg::blas1::pointwiseDivide( ones3d, B, B);
    dg::HVec function = dg::pullback( solovev::FuncNeu(gp), g3d), derivative(function);
    ds( function, derivative);

    ds.centeredT( B, divB);
    double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    std::cout << "Divergence of B is "<<norm<<"\n";

    ds.centered( lnB, gradB);
    std::cout << "num. norm of gradLnB is "<<sqrt( dg::blas2::dot( gradB,vol3d, gradB))<<"\n";
    norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    std::cout << "ana. norm of gradLnB is "<<norm<<"\n";
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    X = divB;
    err = nc_put_var_double( ncid, divBID, periodify(X, g2d_periodic).data());
    double norm2 = sqrt(dg::blas2::dot(gradLnB, vol3d,gradLnB));
    std::cout << "rel. error of lnB is    "<<norm2/norm<<"\n";
    err = nc_close( ncid);



    return 0;
}
