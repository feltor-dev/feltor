#include <iostream>
#include <string>
#include <mpi.h>
#include <cmath>

#include "dg/algorithm.h"
#define _FILE_INCLUDED_BY_DG_
#include "nc_utilities.h"

double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
double gradientX(double x, double y, double z){return cos(x)*sin(y)*cos(z);}
double gradientY(double x, double y, double z){return sin(x)*cos(y)*cos(z);}
double gradientZ(double x, double y, double z){return -sin(x)*sin(y)*sin(z);}

struct Record
{
    std::string name;
    std::string long_name;
    std::function<void (dg::x::DVec&, const dg::MPIGrid3d&, double)> function;
};

std::vector<Record> records_list = {
    {"vectorX", "X-component of vector",
        [] ( dg::x::DVec& resultD, const dg::MPIGrid3d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", "Y-component of vector",
        [] ( dg::x::DVec& resultD, const dg::MPIGrid3d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorZ", "Z-component of vector",
        [] ( dg::x::DVec& resultD, const dg::MPIGrid3d& g, double time){
            resultD = dg::evaluate( gradientZ, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};

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
    double x0 = 0., x1 = 2.*M_PI;
    MPI_Comm comm;
    std::stringstream ss;
    ss<< "2 1 2";
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);
    dg::MPIGrid3d grid( x0,x1,x0,x1,x0,x1,3,10,10,20, comm);
    std::string hello = "Hello world\n";
    dg::MPI_Vector<thrust::host_vector<double>> data = dg::evaluate( function, grid);

    //create NetCDF File
    int ncid=0;
    dg::file::NC_Error_Handle err;
    if(rank==0)err = nc_create( "testmpi.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    if(rank==0)err = nc_put_att_text( ncid, NC_GLOBAL, "input", hello.size(), hello.data());

    int dimids[4], tvarID;
    if(rank==0)err = dg::file::define_dimensions( ncid, dimids, &tvarID, grid);
    int dataID;
    if(rank==0)err = nc_def_var( ncid, "data", NC_DOUBLE, 4, dimids, &dataID);
    dg::file::WriteRecordsList<4> records( ncid, dimids, records_list);

    /* Write metadata to file. */
    size_t Tcount=1, Tstart=0;
    double time = 0;
    //err = nc_close(ncid);
    for(unsigned i=0; i<=NT; i++)
    {
        if(rank==0)std::cout<<"Write timestep "<<i<<"\n";
        time = i*dt;
        Tstart = i;
        data = dg::evaluate( function, grid);
        dg::blas1::scal( data, cos( time));
        records.write( ncid, grid, records_list, grid, time);
        //write dataset (one timeslice)
        dg::file::put_vara_double( ncid, dataID, i, grid, data, false);
        if(rank==0)err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
    }
    if(rank==0)err = nc_close(ncid);
    MPI_Finalize();
    return 0;
}
