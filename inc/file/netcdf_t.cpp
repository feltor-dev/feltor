#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#endif

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
    std::function<void (dg::x::DVec&, const dg::x::Grid3d&, double)> function;
};

std::vector<Record> records_list = {
    {"vectorX", "X-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", "Y-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorZ", "Z-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientZ, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};

int main(int argc, char* argv[])
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm;
    //create a grid and some data
    if( size != 4){ std::cerr << "Please run with 4 threads!\n"; return -1;}
    std::stringstream ss;
    ss<< "2 1 2";
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);
#endif
#ifdef WITH_MPI
    std::string filename = "testmpi.nc";
#else
    std::string filename = "test.nc";
#endif
    DG_RANK0 std::cout << "WRITE A TIMEDEPENDENT SCALAR, SCALAR FIELD, AND VECTOR FIELD TO NETCDF4 FILE "
                       << filename<<"\n";
    double Tmax=2.*M_PI;
    double NT = 10;
    double h = Tmax/NT;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid3d grid( x0,x1,x0,x1,x0,x1,3,10,10,20
#ifdef WITH_MPI
    , comm
#endif
    );
    std::string hello = "Hello world\n";
    dg::x::HVec data = dg::evaluate( function, grid);

    //create NetCDF File
    int ncid;
    dg::file::NC_Error_Handle err;
    DG_RANK0 err = nc_create( filename.data(), NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4
    DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL, "input", hello.size(), hello.data());

    int dim_ids[4], tvarID;
    DG_RANK0 err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid);

    int dataID, scalarID;
    DG_RANK0 err = nc_def_var( ncid, "data", NC_DOUBLE, 1, dim_ids, &dataID);
    DG_RANK0 err = nc_def_var( ncid, "scalar", NC_DOUBLE, 4, dim_ids, &scalarID);

    dg::file::WriteRecordsList<4> records( ncid, dim_ids, records_list);
    for(unsigned i=0; i<=NT; i++)
    {
        DG_RANK0 std::cout<<"Write timestep "<<i<<"\n";
        double time = i*h;
        const size_t Tcount = 1;
        const size_t Tstart = i;
        data = dg::evaluate( function, grid);
        dg::blas1::scal( data, cos( time));
        double energy = dg::blas1::dot( data, data);
        //write scalar data point
        DG_RANK0 err = nc_put_vara_double( ncid, dataID, &Tstart, &Tcount, &energy);
        //write scalar field
        dg::file::put_vara_double( ncid, scalarID, Tstart, grid, data);
        //write vector field
        records.write( ncid, grid, records_list, grid, time);
        //write time
        DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
    }

    DG_RANK0 err = nc_close(ncid);
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
