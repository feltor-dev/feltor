#include <iostream>

// Define this Macro to test cuda-unaware mpi behaviour

//#define DG_CUDA_UNAWARE_MPI

#include <random> // for random shuffle
#include <mpi.h>
#include "../blas1.h"
#include "mpi_gather.h"
#include "catch2/catch_all.hpp"

// If you get cuIpcCloseMemHandle failed errors when executing with cuda
// The cause the IN_PLACE option in mpi functions
// then https://github.com/horovod/horovod/issues/82
// --mca btl_smcuda_use_cuda_ipc 0
TEMPLATE_TEST_CASE( "Gather MPI", "", int, double)
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    INFO("# processes =  " <<size );
    thrust::host_vector<TestType> v, ana;
    thrust::host_vector<std::array<int,2>> gIdx;
    bool bijective = true;

    SECTION( "Simple shift (non-symmetric, bijective)")
    {
        unsigned N=8, shift = 3, global_N = N*size;
        v.resize( N), ana.resize(N); gIdx.resize( N);
        for( unsigned i=0; i<gIdx.size(); i++)
        {
            gIdx[i][0] = i+shift >= N ? (rank+1)%size : rank; // PID
            gIdx[i][1] = (i + shift)%N; // local idx on PID
            v[i] = TestType( rank*N + i);
            ana[i] = TestType(( rank*N + i + shift) %global_N);
        }
    }
    SECTION( "Simple transpose ( symmetric, bijective)")
    {

        unsigned global_rows = 2*size, global_cols = size;
        unsigned local_rows = global_rows, local_cols = 1;// matrix size  every rank holds a column
        unsigned local_rowsT = global_cols, local_colsT = 2;// matrix size  every rank holds a column
        // in the transposed matrix every rank holds 2 columns
        v.resize( local_rows*local_cols);
        ana.resize(local_rowsT*local_colsT);
        gIdx.resize( local_rowsT*local_colsT);
        for( unsigned i=0; i<local_rows; i++)
        for( unsigned k=0; k<local_cols; k++)
            v[i*local_cols + k] = (TestType)(i*global_cols + rank + k);
        for( unsigned i=0; i<local_rowsT; i++)
        for( unsigned k=0; k<local_colsT; k++)
        {
            gIdx[i*local_colsT+k][0] = i; // PID
            gIdx[i*local_colsT+k][1] = (rank*2 + k); // local idx
            ana[i*local_colsT+k] = TestType(rank*local_rowsT*local_colsT + k*local_rowsT+i);
        }
    }
    SECTION( " Repartition test with quadratic sizes ( bijective, inhomogeneous)\n")
    {
        unsigned N=rank;
        unsigned NT = size-1-rank;
        v.resize( N), ana.resize(NT); gIdx.resize( NT);
        TestType value = TestType(rank*rank-rank)/2;
        for( unsigned i=0; i<N; i++)
            v[i] = value+i;
        int start = 0;
        for ( unsigned r=0; r<(unsigned)rank; r++)
        {
            int back_rank = size-1-r;
            start += back_rank;
        }
        for( unsigned i=0; i<NT; i++)
        {
            int globalIdx = start + i;
            int pid = floor( (1.+sqrt( 1.+globalIdx*8.))/2.);
            gIdx[i][0] = pid;
            gIdx[i][1] = globalIdx - (pid*pid-pid)/2;
            ana[i] = TestType( globalIdx);
        }
    }
    SECTION(" Random shuffle ( non-bijective, homogeneous)\n")
    {
        unsigned N = 1000, global_N = size*N;
        v.resize( N), ana.resize(N); gIdx.resize( N);
        thrust::sequence( v.begin(), v.end());
        // The idea for a test is that if we gather the index we get the gather map

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distr( 0, global_N-1);
        for( int i=0; i<(int)N; i++)
        {
            v[i] += rank*N; // we want the global index
            int idx = distr( gen);
            int pid = idx / N;
            int lIdx = idx % N;
            gIdx[i] = {pid, lIdx};
            ana[i] = idx;
        }

        INFO("RANK Random"<< rank<<"\n");
        for( unsigned u=0; u<10; u++)
            INFO(gIdx[u][0]<<" "<<gIdx[u][1]<<" ");
        bijective = false;
    }
    thrust::device_vector<TestType> v_d = v, ana_d = ana;
    thrust::host_vector<int> bufferIdx;
    dg::MPIGather<thrust::device_vector> mpi_gather(gIdx, bufferIdx, MPI_COMM_WORLD);
    thrust::device_vector<int> local_gather(bufferIdx);
    thrust::device_vector<TestType> buffer( mpi_gather.buffer_size());
    mpi_gather.global_gather_init( v_d, buffer);
    mpi_gather.global_gather_wait( buffer);
    thrust::device_vector<TestType> num_d(ana_d);
    thrust::gather( local_gather.begin(), local_gather.end(), buffer.begin(),
        num_d.begin());
    for( unsigned i=0; i<ana.size(); i++)
    {
        INFO("Gather Rank "<<rank<<" "<<ana_d[i] << " "<<num_d[i]<<"\n");
        CHECK( ana_d[i] == num_d[i]);
    }
    if( bijective) // Scatter the index
    {
        auto sIdx = dg::mpi_invert_permutation( gIdx, MPI_COMM_WORLD);
        dg::MPIGather<thrust::device_vector > mpi_gather(sIdx, bufferIdx, MPI_COMM_WORLD);
        thrust::device_vector<int> local_gather(bufferIdx);
        num_d = v_d;
        dg::blas1::copy( 0, num_d);
        thrust::device_vector<TestType> buffer( mpi_gather.buffer_size());
        mpi_gather.global_gather_init( ana_d, buffer);
        mpi_gather.global_gather_wait( buffer);
        thrust::gather( local_gather.begin(), local_gather.end(), buffer.begin(),
            num_d.begin());
        for( unsigned i=0; i<v.size(); i++)
        {
            INFO("Scatter Rank "<<rank<<" "<<v_d[i] << " "<<num_d[i]<<"\n");
            CHECK( v_d[i] == num_d[i]);
        }
    }
    // test mpi_gather
    thrust::host_vector<TestType> num= ana;
    dg::mpi_gather( gIdx, v, num, MPI_COMM_WORLD);
    for( unsigned i=0; i<ana.size(); i++)
    {
        INFO("Gather Rank "<<rank<<" "<<ana[i] << " "<<num[i]<<"\n");
        CHECK( ana[i] == num[i]);
    }
    if( bijective)
    {
        num = v;
        dg::blas1::copy( 0, num);
        dg::mpi_scatter( gIdx, ana, num, MPI_COMM_WORLD);
        for( unsigned i=0; i<v.size(); i++)
        {
            INFO("Scatter Rank "<<rank<<" "<<v[i] << " "<<num[i]<<"\n");
            CHECK( v[i] == num[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("MPIAllreduce")
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    unsigned N = 100;
    thrust::host_vector<int> v(N, 1);
    dg::detail::MPIAllreduce red( MPI_COMM_WORLD);
    red.reduce(v);
    CHECK( v[99] == size);
    MPI_Barrier(MPI_COMM_WORLD);
}
