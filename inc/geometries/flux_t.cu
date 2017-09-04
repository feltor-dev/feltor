#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"

#include "dg/backend/timer.cuh"
#include "curvilinear.h"
//#include "guenther.h"
#include "testfunctors.h"
#include "solovev.h"
#include "flux.h"
#include "fieldaligned.h"
#include "ds.h"
#include "init.h"

#include "file/nc_utilities.h"

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

int main( int argc, char* argv[])
{
    std::cout << "Type n(3), Nx(8), Ny(80), Nz(20)\n";
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
    dg::geo::solovev::GeomParameters gp(js);
    Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    std::cout << "Constructing flux grid ... \n";
    t.tic();
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField( gp);
    dg::geo::FluxGenerator flux( c.get_psip(), c.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::CurvilinearProductGrid3d g3d(flux, n, Nx, Ny,Nz, dg::DIR);
    dg::CurvilinearGrid2d g2d(flux, n, Nx,Ny, dg::NEU);
    dg::Grid2d g2d_periodic(g2d.x0(), g2d.x1(), g2d.y0(), g2d.y1(), g2d.n(), g2d.Nx(), g2d.Ny()+1); 
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    //////////////////////////////setup netcdf//////////////////
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "flux.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    err = file::define_dimensions(  ncid, dim2d, g2d_periodic);
    int coordsID[2];
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    dg::HVec X=dg::pullback(dg::cooX2d, g2d), Y=dg::pullback(dg::cooY2d, g2d); //P = dg::pullback( dg::coo3, g);
    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g2d_periodic).data());
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());

    std::string names[] = {"psi", "d", "R", "vol", "divB"};
    unsigned size=5;
    int varID[size];
    for( unsigned i=0; i<size; i++)
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 2, dim2d, &varID[i]);
    ///////////////////////now fill variables///////////////////

    thrust::host_vector<double> psi_p = dg::pullback( psip, g2d);
    //g.display();
    err = nc_put_var_double( ncid, varID[0], periodify(psi_p, g2d_periodic).data());
    dg::HVec temp0( g2d.size()), temp1(temp0);
    dg::HVec w3d = dg::create::weights( g2d);

    //compute and write deformation into netcdf
    dg::SparseTensor<dg::HVec> metric = g2d.metric();
    dg::HVec g_xx = metric.value(0,0), g_xy = metric.value(0,1), g_yy=metric.value(1,1);
    dg::SparseElement<dg::HVec> vol_ = dg::tensor::volume(metric);
    dg::blas1::pointwiseDivide( g_xy, g_xx, temp0);
    const dg::HVec ones = dg::evaluate( dg::one, g2d);
    X=g_yy;
    err = nc_put_var_double( ncid, varID[1], periodify(X, g2d_periodic).data());
    //compute and write conformalratio into netcdf
    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    X=g_xx;
    err = nc_put_var_double( ncid, varID[2], periodify(X, g2d_periodic).data());

    std::cout << "Construction successful!\n";

    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>()); //temp0=1/sqrt(g) = sqrt(g^xx g^yy - g^xy^2)
    dg::blas1::pointwiseDivide( ones, temp0, temp0); //temp0=sqrt(g)
    X=temp0;
    err = nc_put_var_double( ncid, varID[3], periodify(X, g2d_periodic).data());
    dg::blas1::axpby( 1., temp0, -1., vol_.value(), temp0); //temp0 = sqrt(g)-vol
    double error = sqrt(dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot(vol_.value(), w3d, vol_.value()));
    std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    const dg::HVec vol3d = dg::create::volume( g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    double volume3d = dg::blas1::dot( vol3d, ones3d);
    const dg::HVec vol2d = dg::create::volume( g2d);
    dg::HVec ones2d = dg::evaluate( dg::one, g2d);
    double volume2d = dg::blas1::dot( vol2d, ones2d);

    std::cout << "TEST VOLUME IS:\n";
    if( psi_0 < psi_1) gp.psipmax = psi_1, gp.psipmin = psi_0;
    else               gp.psipmax = psi_0, gp.psipmin = psi_1;
    dg::geo::Iris iris( c.psip(), gp.psipmin, gp.psipmax);
    dg::CartesianGrid2d g2dC( gp.R_0 -2.0*gp.a, gp.R_0 + 2.0*gp.a, -2.0*gp.a,2.0*gp.a,1, 2e3, 2e3, dg::PER, dg::PER);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec R  = dg::evaluate( dg::cooX2d, g2dC);
    dg::HVec onesC = dg::evaluate( dg::one, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    double volumeRZ = dg::blas2::dot( vec, g2d_weights, onesC);
    std::cout << "volumeXYP is "<< volume3d  <<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP <<std::endl;
    std::cout << "volumeXY  is "<< volume2d  <<std::endl;
    std::cout << "volumeRZ  is "<< volumeRZ  <<std::endl;
    std::cout << "relative difference in volume3d is "<<fabs(volumeRZP - volume3d)/volume3d<<std::endl;
    std::cout << "relative difference in volume2d is "<<fabs(volumeRZ - volume2d)/volume2d<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    err = nc_close( ncid);
    return 0;
}
