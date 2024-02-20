#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#include "easy_dims.h"

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
    DG_RANK0 std::cout << "WRITE AND CHECK A COUPLE OF DIMENSIONS IN "
                       << filename<<"\n";
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::CartesianGrid3d grid( x0,x1,x0,x1,x0,x1,3,10,10,20
#ifdef WITH_MPI
    , comm
#endif
    );
    dg::ClonePtr<dg::x::aGeometry2d> grid2d_ptr = grid.perp_grid();

    //create NetCDF File
    int ncid;
    dg::file::NC_Error_Handle err;
    DG_RANK0 err = nc_create( filename.data(), NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4

    int dim_ids[4], tvarID;
    err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid);
    int dim2_ids[3];
    try
    {
        err = dg::file::define_dimensions( ncid, dim2_ids, *grid2d_ptr);
    }
    catch ( dg::file::NC_Error& err)
    {
        std::cerr << "EXPECTED ERROR: ";
        std::cerr << err.what()<<"\n";
    }
    try
    {
        bool exists = dg::file::check_dimensions( ncid, dim2_ids, *grid2d_ptr);
        std::cout << "All dimensions exist "<<std::boolalpha<<exists<<"\n";
    }
    catch ( dg::file::NC_Error& err)
    {
        std::cerr << err.what()<<"\n";
    }
    try
    {
        int tvarID;
        err = dg::file::define_dimensions( ncid, dim2_ids, &tvarID, *grid2d_ptr, {"time", "x", "y"}, true);
    }
    catch ( dg::file::NC_Error& err)
    {
        std::cerr << err.what()<<"\n";
    }

    DG_RANK0 err = nc_close(ncid);
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
