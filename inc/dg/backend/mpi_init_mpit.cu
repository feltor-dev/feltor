#include <iostream>

#include <mpi.h>

#include "mpi_init.h"
#include "catch2/catch.hpp"

// run with mpirun -n 10 --oversubscribe ./mpi_init_mpit
TEST_CASE( "Test mpi_comm_global2local")
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm new_comm;
    MPI_Comm_split( MPI_COMM_WORLD, (rank+1)/3, -rank, &new_comm);
    int local_rank;
    MPI_Comm_rank( new_comm, &local_rank);

    int local_root_rank = dg::mpi_comm_global2local_rank( new_comm);
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
        local_root_rank = dg::mpi_comm_global2local_rank( new_comm, 2, MPI_COMM_WORLD);
        if( rank < 2 or rank > 4)
            CHECK( local_root_rank == MPI_UNDEFINED);
        else
            CHECK( local_root_rank == 2);
    }

    INFO("Global Rank "<<rank<<" has new rank "<<local_rank
        <<" with local root rank "<<local_root_rank<<"\n");

}
