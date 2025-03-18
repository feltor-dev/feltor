#include <iostream>

// Define this Macro to test cuda-unaware mpi behaviour

//#define DG_CUDA_UNAWARE_MPI

#include <mpi.h>
#include "../blas1.h"
#include "mpi_gather_kron.h"
#include "mpi_init.h"
#include "catch2/catch_all.hpp"


TEST_CASE( "Kronecker Gather")
{
    MPI_Comm comm1d = dg::mpi_cart_create( MPI_COMM_WORLD, {0}, {1});
    int rank, size;
    MPI_Comm_rank( comm1d, &rank);
    MPI_Comm_size( comm1d, &size);

    thrust::host_vector<double> v(9); // a 3 x 3 block
    for( unsigned u=0; u<v.size(); u++)
        v[u] = u + 9*rank; // global index
    int rP = (rank+1)%size, r0 = rank, rM = (rank+size-1)%size;

    auto test = GENERATE( 0,1);
    INFO( (test == 0 ? "TEST Contiguous comm\n" : "TEST column comm\n"));
    // mimic a comm pattern for a centered derivative
    std::vector<std::array<int,2>> gIdx = { {rM, 2}, {r0, 0}, {r0,1}, {r0,1}, {r0, 2}, {rP,0}};
    std::vector<std::vector<double>> ana(gIdx.size());
    for( unsigned u=0; u<gIdx.size(); u++)
        for( unsigned k=0; k<3; k++)
        {
            if( test == 0)
                ana[u].push_back( gIdx[u][0]*9+gIdx[u][1]*3+k); // one row
            else
                ana[u].push_back( gIdx[u][0]*9+k*3+gIdx[u][1]); // one column
        }
    thrust::host_vector<int> bufferIdx;
    auto gather_map = dg::gIdx2unique_idx( gIdx, bufferIdx);

    dg::MPIKroneckerGather<thrust::host_vector>
        mpi_gather( test == 0 ? 1 : 3, gather_map, 1, 3, test == 0 ? 3 : 1, comm1d);
    thrust::host_vector<const double*> ptrs_g( mpi_gather.buffer_size());
    thrust::host_vector<const double*> ptrs(gIdx.size());
    mpi_gather.global_gather_init( v);
    mpi_gather.global_gather_wait( v, ptrs_g);
    thrust::gather( bufferIdx.begin(), bufferIdx.end(), ptrs_g.begin(), ptrs.begin());
    for( unsigned u=0; u<ptrs.size(); u++)
    {
        for( int k=0; k<3; k++)
        {
            INFO("Rank "<<rank<<" "<<ptrs[u][k]<<" ana "<<ana[u][k]<<"\n");
            CHECK( ptrs[u][k] == ana[u][k]);
        }
    }

}
