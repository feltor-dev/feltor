#include <iostream>

#include <random> // for random shuffle
#include <mpi.h>
#include "../blas1.h"
#include "mpi_gather.h"

template<class T>
bool is_equal( const T& v, const T& w)
{
    bool equal = true;
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    for( unsigned i=0; i<v.size(); i++)
    {
        if( v[i] != w[i])
        {
            std::cout << rank<<" "<<v[i] << " "<<w[i]<<"\n";
            equal = false;
        }
    }
    return equal;
}

template<template<class> class Vector, class value_type>
void gather_test( const thrust::host_vector<std::array<int,2>>& gIdx,
    const Vector<value_type>& v,
    const Vector<value_type>& ana, bool bijective = false
    )
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    thrust::host_vector<int> bufferIdx;
    auto recv_map = dg::gIdx2unique_idx( gIdx, bufferIdx);
    dg::MPIGather<Vector> mpi_gather(recv_map, MPI_COMM_WORLD);
    dg::LocalGatherMatrix<Vector> local_gather(bufferIdx);
    Vector<value_type> buffer( mpi_gather.buffer_size());
    mpi_gather.global_gather_init( v, buffer);
    mpi_gather.global_gather_wait( buffer);
    Vector<value_type> num(ana);
    local_gather.gather( buffer, num);
    bool equal  = is_equal( ana, num);
    std::cout <<"GATHER Rank "<<rank<< (equal ? " PASSED" : " FAILED")<<std::endl;
    if( bijective) // Scatter the index
    {
        auto sIdx = dg::mpi_invert_permutation( gIdx, MPI_COMM_WORLD);
        auto recv_map = dg::gIdx2unique_idx( sIdx, bufferIdx);
        dg::LocalGatherMatrix<Vector> local_gather(bufferIdx);

        dg::MPIGather<Vector > mpi_gather(recv_map, MPI_COMM_WORLD);
        num = v;
        dg::blas1::copy( 0, num);
        Vector<value_type> buffer( mpi_gather.buffer_size());
        mpi_gather.global_gather_init( ana, buffer);
        mpi_gather.global_gather_wait( buffer);
        local_gather.gather( buffer, num);
        equal  = is_equal( v, num);
        if(!equal)std::cout <<"SCATTER Rank "<<rank<<" FAILED"<<std::endl;
    }
}
template<class value_type>
void mpi_gather_test( const thrust::host_vector<std::array<int,2>>& gIdx,
    const thrust::host_vector<value_type>& v,
    const thrust::host_vector<value_type>& ana, bool bijective = false
    )
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    // test mpi_gather
    auto num= ana;
    dg::mpi_gather( gIdx, v, num, MPI_COMM_WORLD);
    bool equal  = is_equal( ana, num);
    if(!equal) std::cout <<"MPI GATHER Rank "<<rank<< " FAILED"<<std::endl;
    if( bijective)
    {
        num = v;
        dg::blas1::copy( 0, num);
        dg::mpi_scatter( gIdx, ana, num, MPI_COMM_WORLD);
        equal  = is_equal( v, num);
        if(!equal) std::cout <<"MPI SCATTER Rank "<<rank<<" FAILED"<<std::endl;
    }
}

// If you get cuIpcCloseMemHandle failed errors when executing with cuda
// The cause the IN_PLACE option in mpi functions
// then https://github.com/horovod/horovod/issues/82
// --mca btl_smcuda_use_cuda_ipc 0
int main( int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    if(rank==0)std::cout <<"# processes =  " <<size <<std::endl;
    // TODO Add random shuffle test

    {
    if(rank==0)std::cout << "Test: Simple shift (non-symmetric, bijective)\n";
    unsigned N=8, shift = 3, global_N = N*size;
    thrust::host_vector<double> v(N), ana(N);
    thrust::host_vector<std::array<int,2>> gIdx( N);
    for( unsigned i=0; i<gIdx.size(); i++)
    {
        gIdx[i][0] = i+shift >= N ? (rank+1)%size : rank; // PID
        gIdx[i][1] = (i + shift)%N; // local idx on PID
        v[i] = double( rank*N + i);
        ana[i] = double(( rank*N + i + shift) %global_N);
    }
    gather_test<thrust::device_vector, double>( gIdx, v, ana, true);
    mpi_gather_test( gIdx, v, ana, true);
    MPI_Barrier(MPI_COMM_WORLD);
    }
    {
    if(rank==0)std::cout << "Test simple transpose ( symmetric, bijective)"<<std::endl;

    unsigned global_rows = 2*size, global_cols = size;
    unsigned local_rows = global_rows, local_cols = 1;// matrix size  every rank holds a column
    unsigned local_rowsT = global_cols, local_colsT = 2;// matrix size  every rank holds a column
    // in the transposed matrix every rank holds 2 columns
    thrust::host_vector<double> mat( local_rows*local_cols ),
                                matT( local_rowsT*local_colsT);
    thrust::host_vector<std::array<int,2>> gIdx( local_rowsT*local_colsT);
    for( unsigned i=0; i<local_rows; i++)
    for( unsigned k=0; k<local_cols; k++)
        mat[i*local_cols + k] = (double)(i*global_cols + rank + k);
    for( unsigned i=0; i<local_rowsT; i++)
    for( unsigned k=0; k<local_colsT; k++)
    {
        gIdx[i*local_colsT+k][0] = i; // PID
        gIdx[i*local_colsT+k][1] = (rank*2 + k); // local idx
        matT[i*local_colsT+k] = double(rank*local_rowsT*local_colsT + k*local_rowsT+i);
    }
    gather_test<thrust::device_vector,double>( gIdx, mat, matT, true);
    mpi_gather_test( gIdx, mat, matT, true);
    MPI_Barrier(MPI_COMM_WORLD);
    }
    {
    if(rank==0)std::cout << " Repartition test with quadratic sizes ( bijective, inhomogeneous)\n";
    unsigned N=rank;
    unsigned NT = size-1-rank;
    thrust::host_vector<double> v(N), ana(NT);
    thrust::host_vector<std::array<int,2>> gIdx( NT);
    double value = double(rank*rank-rank)/2;
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
        ana[i] = double( globalIdx);
    }
    gather_test<thrust::device_vector, double>( gIdx, v, ana, true);
    mpi_gather_test( gIdx, v, ana, true);
    MPI_Barrier(MPI_COMM_WORLD);

    }
    {
    if(rank==0)std::cout << " Random shuffle ( non-bijective, homogeneous)\n";
    unsigned N = 1000, global_N = size*N;
    thrust::host_vector<int> v(N), ana( N);
    thrust::host_vector<std::array<int,2>> gIdx( N);
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

    std::cout << "RANK Random"<< rank<<"\n";
    for( unsigned u=0; u<10; u++)
        std::cout << gIdx[u][0]<<" "<<gIdx[u][1]<<" ";
    std::cout << std::endl;
    gather_test<thrust::device_vector, int>( gIdx, v, ana, false);
    mpi_gather_test( gIdx, v, ana, false);
    MPI_Barrier(MPI_COMM_WORLD);


    }


    MPI_Finalize();

    return 0;
}
