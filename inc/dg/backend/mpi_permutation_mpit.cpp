#include <iostream>
#include <mpi.h>
#include "mpi_permutation.h"
#include "catch2/catch_all.hpp"

TEST_CASE( "MPI permutation")
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    INFO("# processes =  " <<size);

    SECTION( "Send each following rank an integer")
    {
        //! [permute]
        int rank, size;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        MPI_Comm_size( MPI_COMM_WORLD, &size);
        // Send an integer to the next process in a "circle"
        std::map<int, int> messages = { {(rank + 1)%size, rank}};
        INFO( "Rank "<<rank<<" send message "<<messages[(rank+1)%size]<<"\n");
        auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
        // Each rank received a message from the previous rank
        INFO("Rank "<<rank<<" received message "<<recv[(rank+size-1)%size]<<"\n");
        CHECK( recv[(rank+size-1)%size] == (rank+size-1)%  size);
        // If we permute again we send everything back
        auto messages_num = dg::mpi_permute( recv, MPI_COMM_WORLD);
        CHECK( messages == messages_num);
        //! [permute]
        CHECK( (size==1 or dg::is_communicating( messages, MPI_COMM_WORLD)));
    }
    SECTION( "Send each following rank a vector")
    {
        std::map<int, thrust::host_vector<int>> messages = { {(rank + 1)%size, thrust::host_vector<int>(10,rank)}};
        CHECK( ( size == 1 or dg::is_communicating( messages, MPI_COMM_WORLD)));
        auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
        CHECK( recv[(rank+size-1)%size][0] == (rank+size-1)%  size);
        auto messages_num = dg::mpi_permute( recv, MPI_COMM_WORLD);
        CHECK( messages == messages_num);
    }
    SECTION( "Send each following rank a MsgChunk")
    {
        std::map<int, dg::detail::MsgChunk> messages = { {(rank + 1)%size, dg::detail::MsgChunk{rank,rank}}};
        CHECK(( size == 1 or dg::is_communicating( messages, MPI_COMM_WORLD)));
        auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
        CHECK( recv[(rank+size-1)%size].idx == (rank+size-1)%  size);
        auto messages_num = dg::mpi_permute( recv, MPI_COMM_WORLD);
        CHECK( messages == messages_num);
    }
    SECTION( "Send each following rank a vector of MsgChunk")
    {
        std::map<int, thrust::host_vector<dg::detail::MsgChunk>> messages =
        { {(rank + 1)%size, thrust::host_vector<dg::detail::MsgChunk>( 10, {rank,rank})}};
        CHECK(( size == 1 or dg::is_communicating( messages, MPI_COMM_WORLD)));
        auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
        CHECK( recv[(rank+size-1)%size][0].idx == (rank+size-1)%  size);
        auto messages_num = dg::mpi_permute( recv, MPI_COMM_WORLD);
        CHECK( messages == messages_num);
    }
    SECTION( "Send each following rank a vector of std::array")
    {
        std::map<int, thrust::host_vector<std::array<int,3>>> messages =
            { {(rank + 1)%size, thrust::host_vector<std::array<int,3>>( 10, {rank,rank,rank})}};
        CHECK(( size == 1 or dg::is_communicating( messages, MPI_COMM_WORLD)));
        auto recv = dg::mpi_permute( messages, MPI_COMM_WORLD);
        CHECK( recv[(rank+size-1)%size][0][0] == (rank+size-1)%  size);
        auto messages_num = dg::mpi_permute( recv, MPI_COMM_WORLD);
        CHECK( messages == messages_num);
    }

}
