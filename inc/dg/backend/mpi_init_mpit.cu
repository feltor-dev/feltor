#include <iostream>

#include <mpi.h>

#include "mpi_init.h"

// run with mpirun -n 10 --oversubscribe ./mpi_init_mpit
int main(int argc, char * argv[])
{
    dg::mpi_init( argc, argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_rank( MPI_COMM_WORLD, &size);
    MPI_Comm new_comm;
    MPI_Comm_split( MPI_COMM_WORLD, (rank+1)/3, -rank, &new_comm);
    int local_rank;
    MPI_Comm_rank( new_comm, &local_rank);

    int local_root_rank = dg::mpi_comm_global2local_rank( new_comm);
    std::cout << "Global Rank "<<rank<<" has new rank "<<local_rank<<" with local root rank "<<local_root_rank<<"\n";

    MPI_Finalize();
    return 0;
}
