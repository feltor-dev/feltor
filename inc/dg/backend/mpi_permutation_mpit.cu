#include <iostream>
#include <mpi.h>
#include "mpi_permutation.h"

int main( int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    if(rank==0)std::cout <<"# processes =  " <<size <<std::endl;

    // 1. Send each following rank an integer

    {
    std::map<int, int> messages = { {(rank + 1)%size, rank}};
    std::cout<< "Rank "<<rank<<" send message "<<messages[(rank+1)%size]<<"\n";
    assert( dg::is_communicating( messages, MPI_COMM_WORLD));
    auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
    std::cout<< "Rank "<<rank<<" received message "<<recv[(rank+size-1)%size]<<"\n";
    assert( recv[(rank+size-1)%size] == (rank+size-1)%  size);
    std::cout << "Rank "<<rank<<" PASSED\n";
    }

    // 2. Send each following rank a vector
    {
    std::map<int, thrust::host_vector<int>> messages = { {(rank + 1)%size, thrust::host_vector<int>(10,rank)}};
    assert( dg::is_communicating( messages, MPI_COMM_WORLD));
    auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
    assert( recv[(rank+size-1)%size][0] == (rank+size-1)%  size);
    std::cout << "Rank "<<rank<<" PASSED\n";
    }
    // 3. Send each following rank a MsgChunk
    {
    std::map<int, dg::detail::MsgChunk> messages = { {(rank + 1)%size, dg::detail::MsgChunk{rank,rank}}};
    assert( dg::is_communicating( messages, MPI_COMM_WORLD));
    auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
    assert( recv[(rank+size-1)%size].idx == (rank+size-1)%  size);
    std::cout << "Rank "<<rank<<" PASSED\n";
    }
    // 4. Send each following rank a vector of MsgChunk
    {
    std::map<int, thrust::host_vector<dg::detail::MsgChunk>> messages =
    { {(rank + 1)%size, thrust::host_vector<dg::detail::MsgChunk>( 10, {rank,rank})}};
    assert( dg::is_communicating( messages, MPI_COMM_WORLD));
    auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
    assert( recv[(rank+size-1)%size][0].idx == (rank+size-1)%  size);
    std::cout << "Rank "<<rank<<" PASSED\n";
    }

    MPI_Finalize();

    return 0;
}
