#include <iostream>
#include "exblas/mpi_accumulate.h"



#include "catch2/catch_all.hpp"

// TODO Find a test to test accumulation for >128 threads
TEST_CASE( "Accumulate reuses comm")
{
    int rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank);

    MPI_Comm comm_mod, comm_red;
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_red);
    // No new communicators should be generated
    MPI_Comm comm_mod2, comm_red2;
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod2, &comm_red2);
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod2, &comm_red2);
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod2, &comm_red2);
    CHECK( comm_mod == comm_mod2);
    CHECK( comm_red == comm_red2);

}

