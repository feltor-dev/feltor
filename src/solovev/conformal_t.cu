#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "draw/host_window.h"

#include "dg/backend/timer.cuh"
#include "conformal.h"
#include "feltor/parameters.h"
#include "init.h"

#include "file/nc_utilities.h"

double sineX( double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY( double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    std::vector<double> v, v2;
try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params.txt"); 
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
    std::cout << "Test naive derivatives:\n";
    dg::Grid2d<double> g2d( 0,1,0,1, n, Nx, Ny);
    const dg::HVec initX = dg::evaluate( sineX, g2d);
    const dg::HVec errorX = dg::evaluate( cosineX, g2d);
    const dg::HVec initY = dg::evaluate( sineY, g2d);
    const dg::HVec errorY = dg::evaluate( cosineY, g2d);
    const dg::HVec w2d = dg::create::weights(g2d);
    dg::HVec deriX(initX), deriY(initY);
    solovev::detail::Naive naive( g2d);

    naive.dx( initX, deriX);
    naive.dy( initY, deriY);
    dg::blas1::axpby( 1.,deriX, -1., errorX, deriX);
    dg::blas1::axpby( 1.,deriY, -1., errorY, deriY);
    double errX = dg::blas2::dot( deriX, w2d, deriX);
    double errY = dg::blas2::dot( deriY, w2d, deriY);
    std::cout << "Errors from naive derivatives are: "<<errX<<" and "<<errY<<"\n";



    //write parameters from file into variables
    const solovev::GeomParameters gp(v);
    gp.display( std::cout);
    dg::Timer t;
    solovev::detail::Fpsi fpsi( gp, -10);
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    t.tic();
    double f_psi = fpsi( -10);
    t.toc();
    std::cout << f_psi<<" took "<<t.diff()<<"s"<<std::endl;
    t.tic();
    solovev::ConformalRingGrid g(gp, psi_0, psi_1, n, Nx, Ny, dg::DIR);
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    err = file::define_dimensions(  ncid, dim2d, g.grid());
    int coordsID[2], onesID;
    err = nc_def_var( ncid, "r", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    err = nc_def_var( ncid, "z", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim2d, &onesID);

    t.tic();
    thrust::host_vector<double> pi( n*Nx, 0);
    g.construct_psi( );
    t.toc();
    std::cout << "Psi vector took "<<t.diff()<<"s"<<std::endl;
    t.tic();
    thrust::host_vector<double> r( n*n*Nx*Ny, 0);
    thrust::host_vector<double> z( n*n*Nx*Ny, 0);
    g.construct_rz(r,z );
    t.toc();
    std::cout << "RZ vector took "<<t.diff()<<"s"<<std::endl;

    thrust::host_vector<double> ones = dg::evaluate( dg::one, g.grid());
    for( unsigned i=0; i<ones.size(); i++)
        ones[i] = psip(r[i], z[i]);
    g.grid().display();
    err = nc_put_var_double( ncid, onesID, ones.data());
    err = nc_put_var_double( ncid, coordsID[0], r.data());
    err = nc_put_var_double( ncid, coordsID[1], z.data());
    err = nc_close( ncid);


    return 0;
}
