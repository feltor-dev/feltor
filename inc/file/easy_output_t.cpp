#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "catch2/catch_all.hpp"

#include "dg/algorithm.h"
#define _FILE_INCLUDED_BY_DG_
#include "easy_dims.h"
#include "easy_output.h"

static double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
static double function( double x, double y){return sin(x)*sin(y);}

#ifdef WITH_MPI
TEST_CASE( "Test mpi_comm_global2local")
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm new_comm;
    MPI_Comm_split( MPI_COMM_WORLD, (rank+1)/3, -rank, &new_comm);
    int local_rank;
    MPI_Comm_rank( new_comm, &local_rank);

    int local_root_rank = dg::file::detail::mpi_comm_global2local_rank( new_comm);
    if( size == 1)
    {
        CHECK( local_root_rank == 0);
    }
    else
    {
        if( rank > 1)
            CHECK( local_root_rank == MPI_UNDEFINED);
        else
            CHECK( local_root_rank == 1);
    }

    if( size > 5)
    {
        local_root_rank = dg::file::detail::mpi_comm_global2local_rank( new_comm, 2, MPI_COMM_WORLD);
        if( rank < 2 or rank > 4)
            CHECK( local_root_rank == MPI_UNDEFINED);
        else
            CHECK( local_root_rank == 2);
    }

    INFO("Global Rank "<<rank<<" has new rank "<<local_rank
        <<" with local root rank "<<local_root_rank<<"\n");

}
#endif // WITH_MPI


TEST_CASE( "Easy output")
{
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm;
    //create a grid and some data
    int dims[3] = {0,0,0};
    MPI_Dims_create( size, 3, dims);
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1]<<" "<<dims[2];
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);
#endif
#ifdef WITH_MPI
    std::string filename = "testmpi.nc";
#else
    std::string filename = "test.nc";
#endif
    DG_RANK0 std::cout << "WRITE A TIME-DEPENDENT SCALAR, SCALAR FIELD AND SUB-FIELDS TO NETCDF4 FILE "
                       << filename<<"\n";
    DG_RANK0 std::cout << "THEN READ IT BACK IN AND TEST EQUALITY\n";
    DG_RANK0 std::cout << "test.nc and testmpi.nc should be binary equal\n";
    double Tmax=2.*M_PI;
    double NT = 10;
    double h = Tmax/NT;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::CartesianGrid3d grid( x0,x1,x0,x1,x0,x1,3,4,4,3, dg::PER,
            dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    std::unique_ptr<dg::x::aGeometry2d> perp_grid_ptr( grid.perp_grid());
    //dg::ClonePtr<dg::x::aGeometry2d> perp_grid_ptr = grid.perp_grid();
    dg::x::HVec data = dg::evaluate( function, grid);
    auto sliced_data = dg::split( data, grid); // a vector of views

    //create NetCDF File
    int ncid;
    dg::file::NC_Error_Handle err;
    DG_RANK0 err = nc_create( filename.data(), NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4
    std::string hello = "Hello world\n";
    DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL, "hello", hello.size(), hello.data());

    int dim_ids[4], tvarID;
    DG_RANK0 err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid);

    int scalarID, arrayID, subArrayID;
    DG_RANK0 err = nc_def_var( ncid, "scalar", NC_DOUBLE, 1, dim_ids, &scalarID);
    DG_RANK0 err = nc_def_var( ncid, "array", NC_DOUBLE, 4, dim_ids, &arrayID);
    int sub_dim_ids[3] = {dim_ids[0], dim_ids[2], dim_ids[3]};
    DG_RANK0 err = nc_def_var( ncid, "sub-array", NC_DOUBLE, 3, sub_dim_ids, &subArrayID);

    int staticArrayID;
    DG_RANK0 err = nc_def_var( ncid, "static-array", NC_DOUBLE, 3, &dim_ids[1], &staticArrayID);
    dg::file::put_var( ncid, staticArrayID, grid, data);

    for(unsigned i=0; i<=NT; i++)
    {
        DG_RANK0 std::cout<<"Write timestep "<<i<<"\n";
        double time = i*h;
        const size_t Tstart = i;
        data = dg::evaluate( function, grid);
        dg::blas1::scal( data, cos( time));
        double energy = dg::blas1::dot( data, data);
        //write scalar data point
        dg::file::put_vara( ncid, scalarID, Tstart, dg::x::Grid0d(), energy);
        //write array
        dg::file::put_vara( ncid, arrayID, Tstart, grid, data);
        //write sub array
        dg::split( data, sliced_data, grid);
        dg::file::put_vara( ncid, subArrayID, Tstart, *perp_grid_ptr, sliced_data[0]);
        //write time
        dg::file::put_vara( ncid, tvarID, Tstart, dg::x::Grid0d(), time);
    }

    DG_RANK0 err = nc_close(ncid);
}
