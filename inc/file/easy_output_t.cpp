#include <iostream>
#include <filesystem>
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
#include "easy_input.h"

static double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
//static double function( double x, double y){return sin(x)*sin(y);}

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

template<class Container>
bool compare( const Container& lhs, const Container& rhs)
{
    if( lhs.size() != rhs.size())
    {
        UNSCOPED_INFO( "lhs size "<<lhs.size()<<" rhs size "<<rhs.size());
        return false;
    }
    for( unsigned i=0; i<lhs.size(); i++)
    {
        UNSCOPED_INFO( "i "<<i<<" lhs "<<*(lhs.begin() +i)
                              <<" rhs "<<*(rhs.begin()+i)
                              <<" diff "<<*(lhs.begin()+i) - *(rhs.begin()+i));
        if( fabs( *(lhs.begin() + i) - *(rhs.begin()+i)) > 1e-15)
            return false;
    }
    return true;
}
#ifdef WITH_MPI
template<class Container>
bool compare( const dg::MPI_Vector<Container>& lhs, const dg::MPI_Vector<Container>& rhs)
{
    return compare( lhs.data(), rhs.data());
}
#endif


// This is a test of legacy code
// The idea is to write data to a file, read it back in and check that what we
// read is what we have written
TEST_CASE( "Easy output")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //create a grid and some data
    int dims[3] = {0,0,0};
    MPI_Dims_create( size, 3, dims);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {1,1,1});
#endif
    INFO( "Write/Read a time-dependent scalar, scalar field "
            <<"and sub-fields to netcdf4 file \"output.nc\"");
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
    dg::x::Grid2d grid2d = (dg::x::Grid2d)*perp_grid_ptr;
    dg::x::HVec data = dg::evaluate( function, grid);
    auto sliced_data = dg::split( data, grid); // a vector of views

    //create NetCDF File
    int ncid = 0;
    dg::file::NC_Error_Handle err;
    DG_RANK0 err = nc_create( "output.nc", NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4
    int dim_ids[4], tvarID;
    DG_RANK0 err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid);

    int scalarID = 0, arrayID = 0, subArrayID = 0;
    DG_RANK0 err = nc_def_var( ncid, "scalar", NC_DOUBLE, 1, dim_ids, &scalarID);
    DG_RANK0 err = nc_def_var( ncid, "array", NC_DOUBLE, 4, dim_ids, &arrayID);
    int sub_dim_ids[3] = {dim_ids[0], dim_ids[2], dim_ids[3]};
    DG_RANK0 err = nc_def_var( ncid, "sub-array", NC_DOUBLE, 3, sub_dim_ids, &subArrayID);

    int staticArrayID = 0;
    DG_RANK0 err = nc_def_var( ncid, "static-array", NC_DOUBLE, 3, &dim_ids[1], &staticArrayID);
    dg::file::put_var( ncid, staticArrayID, grid, data);

    for(unsigned i=0; i<=NT; i++)
    {
        INFO("Write timestep "<<i);
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
        dg::file::put_vara( ncid, subArrayID, Tstart, grid2d, sliced_data[0]);
        //write time
        dg::file::put_vara( ncid, tvarID, Tstart, dg::x::Grid0d(), time);
    }

    DG_RANK0 err = nc_close(ncid);

    // Every rank can open a NetCDF file
    err = nc_open( "output.nc", 0, &ncid);

    bool exists = dg::file::check_dimensions( ncid, dim_ids, &tvarID, grid);
    INFO("Do dimensions exist?\n");
    CHECK( exists);

    err = nc_inq_varid( ncid, "scalar", &scalarID);
    err = nc_inq_varid( ncid, "array", &arrayID);
    err = nc_inq_varid( ncid, "sub-array", &subArrayID);
    err = nc_inq_varid( ncid, "static-array", &staticArrayID);

    data = dg::evaluate( function, grid);
    dg::x::HVec read = dg::evaluate( dg::zero, grid);
    dg::file::get_var( ncid, staticArrayID, grid, read);
    CHECK( compare( read, data));

    for(unsigned i=0; i<=NT; i++)
    {
        INFO("Read timestep "<<i);
        const size_t Tstart = i;
        // The expected solution
        double time = i*h;
        data = dg::evaluate( function, grid);
        dg::blas1::scal( data, cos( time));
        auto sliced_data = dg::split( data, grid); // a vector of views
        double energy;
        //read scalar data point
        dg::file::get_vara( ncid, scalarID, Tstart, dg::x::Grid0d(), energy);
        CHECK( energy == dg::blas1::dot( data, data));

        //read array
        dg::file::get_vara( ncid, arrayID, Tstart, grid, read);
        CHECK( compare( read, data));

        //read sub array
        auto sliced_read = dg::split( read, grid);
        dg::file::get_vara( ncid, subArrayID, Tstart, grid2d, sliced_read[0]);
#ifdef MPI_VERSION
        INFO( "data "<<read.data()[0]<<" dataP "<<sliced_read[0].data().data()[0]);
#else
        INFO( "data "<<read[0]<<" dataP "<<sliced_read[0].data()[0]);
#endif
        CHECK( compare( sliced_read[0], sliced_data[0]));
        //read time
        double read_time;
        dg::file::get_vara( ncid, tvarID, Tstart, dg::x::Grid0d(), read_time);
        INFO( "At time "<<read_time<<" vs expected "<<time);
        CHECK( time == read_time);
    }

    DG_RANK0 err = nc_close(ncid);
    DG_RANK0 std::filesystem::remove( "output.nc");
}
