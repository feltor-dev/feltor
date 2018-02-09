#include <iostream>
#include <string>
#include <mpi.h>
#include <netcdf_par.h>
#include <cmath>

#include "dg/algorithm.h"
#include "nc_utilities.h"

double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}

int main(int argc, char* argv[])
{
    //init MPI
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //create a grid and some data
    if( size != 4){ std::cerr << "Please run with 4 threads!\n"; return -1;}
    double Tmax=2.*M_PI;
    double NT = 10;
    double dt = Tmax/NT;
    dg::Grid1d gx( 0, 2.*M_PI, 3, 10);
    dg::Grid1d gy( 0, 2.*M_PI, 3, 10);
    dg::Grid1d gz( 0, 2.*M_PI, 1, 20);
    dg::Grid3d g( gx, gy, gz);
    std::string hello = "Hello world\n";
    thrust::host_vector<double> data = dg::evaluate( function, g);

    //create NetCDF File
    int ncid;
    file::NC_Error_Handle err;
    MPI_Info info = MPI_INFO_NULL;
    err = nc_create_par( "testmpi.nc", NC_NETCDF4|NC_MPIIO|NC_CLOBBER, MPI_COMM_WORLD, info, &ncid); 
    err = nc_put_att_text( ncid, NC_GLOBAL, "input", hello.size(), hello.data());

    int dimids[4], tvarID;
    err = file::define_dimensions( ncid, dimids, &tvarID, g);
    int dataID;
    err = nc_def_var( ncid, "data", NC_DOUBLE, 4, dimids, &dataID);

    /* Write metadata to file. */
    err = nc_enddef(ncid);

    //err = nc_enddef( ncid);
    size_t start[4] = {0, rank*g.Nz()/size, 0, 0};
    size_t count[4] = {1, g.Nz()/size, g.Ny()*g.n(), g.Nx()*g.n()};
    if( rank==0) std::cout<< "Write from "<< start[0]<< " "<<start[1]<<" "<<start[2]<<" "<<start[3]<<std::endl;
    if( rank==0) std::cout<< "Number of elements "<<count[0]<< " "<<count[1]<<" "<<count[2]<<" "<<count[3]<<std::endl;
    err = nc_var_par_access(ncid, dataID , NC_COLLECTIVE);
    err = nc_var_par_access(ncid, tvarID , NC_COLLECTIVE);
    size_t Tcount=1, Tstart=0;
    double time = 0;
    //err = nc_close(ncid);
    for(unsigned i=0; i<=NT; i++)
    {
        if(rank==0)std::cout<<"Write timestep "<<i<<"\n";
        //err = nc_open_par( "testmpi.nc", NC_WRITE|NC_MPIIO, MPI_COMM_WORLD, info, &ncid); //doesn't work I don't know why
        time = i*dt;
        Tstart = i;
        data = dg::evaluate( function, g);
        dg::blas1::scal( data, cos( time));
        start[0] = i;
        //write dataset (one timeslice)
        err = nc_put_vara_double( ncid, dataID, start, count, data.data() + start[1]*count[2]*count[3]);
        err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
        //err = nc_close(ncid);
    }
    err = nc_close(ncid);
    MPI_Finalize();
    return 0;
}
