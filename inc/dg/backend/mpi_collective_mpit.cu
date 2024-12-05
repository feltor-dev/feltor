#include <iostream>

#include <mpi.h>
#include "../blas1.h"
#include "mpi_collective.h"
#include "mpi_vector.h"

template<class T>
bool is_equal( const T& v, const T& w)
{
    bool equal = true;
    for( unsigned i=0; i<v.size(); i++)
    {
        if( v[i] != w[i])
        {
            equal = false;
        }
    }
    return equal;
}

int main( int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    if(rank==0)std::cout <<"# processes =  " <<size <<std::endl;
    // TODO Repartition test with quadratic sizes 0, 1, 2, 4, ... To  ..., 4, 2, 1, 0
    // TODO Add random shuffle test

    {
    if(rank==0)std::cout << "Test: Simple shift\n";
    unsigned N=8, shift = 3, global_N = N*size;
    thrust::host_vector<double> v(N), ana(N), num(N) ;
    thrust::host_vector<std::array<int,2>> gIdx( N);
    for( unsigned i=0; i<gIdx.size(); i++)
    {
        gIdx[i][0] = i+shift >= N ? (rank+1)%size : rank; // PID
        gIdx[i][1] = (i + shift)%N; // local idx on PID
        v[i] = double( rank*N + i);
        ana[i] = double(( rank*N + i + shift) %global_N);
    }
    dg::MPIGather<thrust::host_vector > mpi_gather(gIdx, MPI_COMM_WORLD);
    thrust::host_vector<double> buffer( mpi_gather.buffer_size());
    mpi_gather.global_gather_init( v, buffer);
    mpi_gather.global_gather_wait( buffer);
    mpi_gather.get_buffer_g1().gather( buffer, num);
    bool equal  = is_equal( ana, num);
    std::cout <<"Rank "<<rank<< (equal ? " PASSED" : " FAILED")<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    }
    {
    if(rank==0)std::cout << "Test simple transpose"<<std::endl;

    unsigned global_rows = 2*size, global_cols = size;
    unsigned local_rows = global_rows, local_cols = 1;// matrix size  every rank holds a column
    unsigned local_rowsT = global_cols, local_colsT = 2;// matrix size  every rank holds a column
    // in the transposed matrix every rank holds 2 columns
    thrust::host_vector<double> mat( local_rows*local_cols ),
                                matT( local_rowsT*local_colsT), numT(matT) ;
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
    dg::MPIGather<thrust::host_vector > mpi_gather(gIdx, MPI_COMM_WORLD);
    thrust::host_vector<double> buffer( mpi_gather.buffer_size());
    mpi_gather.global_gather_init( mat, buffer);
    mpi_gather.global_gather_wait( buffer);
    mpi_gather.get_buffer_g1().gather( buffer, numT);
    bool equal  = is_equal( matT, numT);
    std::cout <<"Rank "<<rank<<(equal ? " PASSED" : " FAILED")<<std::endl;
    MPI_Barrier( MPI_COMM_WORLD);
    auto num = mat;
    dg::blas1::copy( 0, num);
    mpi_gather.global_scatter_plus_init(buffer, num);
    mpi_gather.global_scatter_plus_wait( num);
    equal  = is_equal( mat, num);
    std::cout <<"Rank "<<rank<<(equal ? " PASSED" : " FAILED")<<std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    }
    {
    if(rank==0)std::cout << "Test GeneralComm and SurjectiveComm:"<<std::endl;
    if(rank==0)std::cout << "Test Norm Gv*Gv vs G^T(Gv)*v: "<<std::endl;
    unsigned N = 20;
    thrust::host_vector<double> vec( N, rank);
    for( int i=0; i<(int)(N); i++)
        vec[i]*=i;
    thrust::host_vector<std::array<int,2>> gIdx( N+rank);
    for( unsigned i=0; i<N+rank; i++)
    {
        gIdx[i][1] = i%N;
        if(i>=5 && i<10) gIdx[i][1]+=3;
        gIdx[i][0] = rank;
        if( i>=N) gIdx[i][0] = (rank+1)%size;
    }
    dg::MPIGather<thrust::host_vector> sur( gIdx, MPI_COMM_WORLD);
    thrust::host_vector<double> receive( sur.buffer_size());
    sur.global_gather_init( vec, receive);
    sur.global_gather_wait( receive);
    /// Test if global_scatter_reduce is in fact the transpose of global_gather
    dg::MPI_Vector<thrust::host_vector<double>> mpi_receive( receive, MPI_COMM_WORLD);
    double norm1 = dg::blas1::dot( mpi_receive, mpi_receive);
    thrust::host_vector<double> vec2(vec.size(), 0);
    sur.global_scatter_plus_init( receive, vec2);
    sur.global_scatter_plus_wait( vec2);
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec( vec, MPI_COMM_WORLD);
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec2( vec2, MPI_COMM_WORLD);
    double norm2 = dg::blas1::dot( mpi_vec, mpi_vec2);
    {
        if( fabs(norm1-norm2)<1e-14)
            std::cout <<"Rank "<<rank<<" PASSED "<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED "<<std::endl;
    }
    if(fabs(norm1-norm2)>1e-14 && rank==0)std::cout << norm1 << " "<<norm2<< " "<<norm1-norm2<<std::endl;
    }
    {
    if(rank==0) std::cout << "Test non-communicating versions\n";
    //  Same as previous with only local rank
    unsigned N = 20;
    thrust::host_vector<double> vec( N, rank);
    for( int i=0; i<(int)(N); i++)
        vec[i]*=i;
    thrust::host_vector<std::array<int,2>> gIdx( N+rank);
    for( unsigned i=0; i<N+rank; i++)
    {
        gIdx[i][1] = i%N;
        if(i>=5 && i<10) gIdx[i][1]+=3;
        gIdx[i][0] = rank;
        // if( i>=N) gIdx[i][0] = (rank+1)%size;
    }
    dg::MPIGather<thrust::host_vector> sur( gIdx, MPI_COMM_WORLD);
    thrust::host_vector<double> receive( sur.buffer_size());
    sur.global_gather_init( vec, receive);
    sur.global_gather_wait( receive);
    /// Test if global_scatter_reduce is in fact the transpose of global_gather
    dg::MPI_Vector<thrust::host_vector<double>> mpi_receive( receive, MPI_COMM_WORLD);
    double norm1 = dg::blas1::dot( mpi_receive, mpi_receive);
    thrust::host_vector<double> vec2(vec.size(), 0);
    sur.global_scatter_plus_init( receive, vec2);
    sur.global_scatter_plus_wait( vec2);
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec( vec, MPI_COMM_WORLD);
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec2( vec2, MPI_COMM_WORLD);
    double norm2 = dg::blas1::dot( mpi_vec, mpi_vec2);
    {
        if( fabs(norm1-norm2)<1e-14)
            std::cout <<"Rank "<<rank<<" PASSED "<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED "<<std::endl;
    }
    if(fabs(norm1-norm2)>1e-14 && rank==0)std::cout << norm1 << " "<<norm2<< " "<<norm1-norm2<<std::endl;

    assert( !sur.isCommunicating());
    // Just to show that even if not communciating the size is not zero
    //if(rank==0)std::cout << "Rank "<<rank<<" buffer size "<<sur.buffer_size()<<" \n";
    //if(rank==0)std::cout << "Rank "<<rank<<" local  size "<<vec.size()<<" \n";
    }

    MPI_Finalize();

    return 0;
}
