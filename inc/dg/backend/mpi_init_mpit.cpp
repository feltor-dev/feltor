#include <iostream>

#include <mpi.h>

#include "mpi_init.h"
#include "catch2/catch_all.hpp"

// run with mpirun -n 10 --oversubscribe ./mpi_init_mpit

TEST_CASE( "mpi_read_as")
{
    int rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank);
    SECTION( "Read no values")
    {
        auto vals = dg::mpi_read_as<int>( 0, comm);
        CHECK( vals.empty());
    }
    SECTION( "Read 4 values")
    {
        std::string values;
        if(rank==0)
            values = "12 17 42 2";
        std::istringstream iss( values);
        auto vals = dg::mpi_read_as<unsigned>( 4, comm, iss);
        CHECK( vals == std::vector<unsigned>{ 12, 17, 42, 2});
    }
}

TEST_CASE( "mpi_read_grid")
{
    int rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank);
    std::stringstream os;
    std::string values;
    if(rank==0)
        values = "12 17 42 2";
    std::istringstream iss( values);
    unsigned n, Nx, Ny, Nz;
    SECTION( "Nothing written for false verbose")
    {
        dg::mpi_read_grid( n, {&Nx,&Ny, &Nz}, comm, iss, false, os);
        CHECK( os.str().empty());
        CHECK( n == 12);
        CHECK( Nx == 17);
        CHECK( Ny == 42);
        CHECK( Nz == 2);
    }
    SECTION( "Stuff written for true verbose")
    {
        dg::mpi_read_grid( n, {&Nx,&Ny, &Nz}, comm, iss, true, os);
        if( rank == 0)
            CHECK( (not os.str().empty()));
        else
            CHECK( os.str().empty());
        CHECK( n == 12);
        CHECK( Nx == 17);
        CHECK( Ny == 42);
        CHECK( Nz == 2);
    }
}

TEST_CASE( "mpi_cart_create")
{
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    SECTION( "passing 0 as dims works")
    {
        MPI_Comm cart = dg::mpi_cart_create( comm, {0,0}, {false, true});
        int dims[2], periods[2], coords[2];
        int err = dg::mpi_cart_get( cart, 2, dims, periods, coords);
        CHECK(  err == MPI_SUCCESS);
        CHECK( dims[0]*dims[1] == size);
        CHECK( periods[0] == false);
        CHECK( periods[1] == true);
    }
    SECTION( "passing explicit dims as dims works")
    {
        std::vector<int> dd(2);
        MPI_Dims_create( size, 2, &dd[0]);
        MPI_Comm cart = dg::mpi_cart_create( comm, dd, {true, true});
        int dims[2], periods[2], coords[2];
        int err = dg::mpi_cart_get( cart, 2, dims, periods, coords);
        CHECK(  err == MPI_SUCCESS);
        CHECK( dims[0]*dims[1] == size);
        CHECK( periods[0] == true);
        CHECK( periods[1] == true);
        CHECK( dims[0] == dd[0]);
        CHECK( dims[1] == dd[1]);
    }
}

TEST_CASE( "mpi_cart_create verbose")
{
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    std::stringstream os;
    std::string values;
    if(rank==0)
        values = "0 0";
    std::istringstream iss( values);
    SECTION( "Periodic bcs")
    {
        MPI_Comm cart = dg::mpi_cart_create( {dg::PER, dg::PER}, iss, comm, true, false, os);
        CHECK( os.str().empty());
        int dims[2], periods[2], coords[2];
        int err = dg::mpi_cart_get( cart, 2, dims, periods, coords);
        CHECK(  err == MPI_SUCCESS);
        CHECK( dims[0]*dims[1] == size);
        CHECK( periods[0] == true);
        CHECK( periods[1] == true);
    }
    SECTION( "Non-periodic bcs")
    {
        MPI_Comm cart = dg::mpi_cart_create( {dg::DIR, dg::NEU}, iss, comm, true, true, os);
        if( rank == 0)
            CHECK( not os.str().empty());
        else
            CHECK( os.str().empty());
        int dims[2], periods[2], coords[2];
        int err = dg::mpi_cart_get( cart, 2, dims, periods, coords);
        CHECK(  err == MPI_SUCCESS);
        CHECK( dims[0]*dims[1] == size);
        CHECK( periods[0] == false);
        CHECK( periods[1] == false);
    }
}
