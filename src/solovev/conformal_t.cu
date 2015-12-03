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

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
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
    solovev::ConformalRingGrid g(gp, -10, -3, n, Nx, Ny, dg::DIR);
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;

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
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    err = file::define_dimensions(  ncid, dim2d, g.grid());
    int coordsID[2], onesID;
    err = nc_def_var( ncid, "r", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    err = nc_def_var( ncid, "z", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim2d, &onesID);

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
