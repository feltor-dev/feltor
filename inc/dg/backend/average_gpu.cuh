#pragma once

#include "exblas/exdot_cuda.cuh"
#ifdef MPI_VERSION
#include "exblas/mpi_accumulate.h"
#endif //MPI_VERSION

namespace dg
{

template<class value_type>
__global__ void transpose_gpu_kernel( unsigned nx, unsigned ny, const value_type* __restrict__ in, value_type* __restrict__ out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[j*ny+i] = in[i*nx+j];
    }
}
template<class value_type>
void transpose_dispatch( CudaTag, unsigned nx, unsigned ny, const value_type* in, value_type* out){
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    transpose_gpu_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}

template<class value_type>
__global__ void extend_line_kernel( unsigned nx, unsigned ny, const value_type* __restrict__ in, value_type* __restrict__ out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[i*nx+j] = in[j];
    }
}
template<class value_type>
void extend_line( CudaTag, unsigned nx, unsigned ny, const value_type* in, value_type* out){
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    extend_line_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}

template<class value_type>
__global__ void extend_column_kernel( unsigned nx, unsigned ny, const value_type* __restrict__ in, value_type* __restrict__ out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[i*nx+j] = in[i];
    }
}
template<class value_type>
void extend_column( CudaTag, unsigned nx, unsigned ny, const value_type* in, value_type* out){
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    extend_column_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}

template<class value_type>
void average( CudaTag, unsigned nx, unsigned ny, const value_type* in0, const value_type* in1, value_type* out)
{
    static_assert( std::is_same<value_type, double>::value, "Value type must be double!");
    static thrust::device_vector<int64_t> d_accumulator;
    static thrust::host_vector<int64_t> h_accumulator;
    static thrust::host_vector<value_type> h_round;
    d_accumulator.resize( ny*exblas::BIN_COUNT);
    int64_t* d_ptr = thrust::raw_pointer_cast( d_accumulator.data());
    int status = 0;
    for( unsigned i=0; i<ny; i++)
        exblas::exdot_gpu(nx, &in0[i*nx], &in1[i*nx], &d_ptr[i*exblas::BIN_COUNT], &status);
    if(status != 0)
        throw dg::Error(dg::Message(_ping_)<<"GPU Average failed since one of the inputs contains NaN or Inf");
    h_accumulator = d_accumulator;
    h_round.resize( ny);
    for( unsigned i=0; i<ny; i++)
        h_round[i] = exblas::cpu::Round( &h_accumulator[i*exblas::BIN_COUNT]);
    // This test checks for errors in the current stream, the error may come
    // from any kernel prior to this point not necessarily the above one
    cudaError_t code = cudaGetLastError( );
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    code = cudaMemcpy( out, &h_round[0], ny*sizeof(value_type), cudaMemcpyHostToDevice);
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
}

#ifdef MPI_VERSION
//local data plus communication
template<class value_type>
void average_mpi( CudaTag, unsigned nx, unsigned ny, const value_type* in0, const value_type* in1, value_type* out, MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce )
{
    static_assert( std::is_same<value_type, double>::value, "Value type must be double!");
    static thrust::device_vector<int64_t> d_accumulator;
    static thrust::host_vector<int64_t> h_accumulator;
    static thrust::host_vector<int64_t> h_accumulator2;
    static thrust::host_vector<value_type> h_round;
    d_accumulator.resize( ny*exblas::BIN_COUNT);
    int64_t* d_ptr = thrust::raw_pointer_cast( d_accumulator.data());
    int status = 0;
    for( unsigned i=0; i<ny; i++)
        exblas::exdot_gpu(nx, &in0[i*nx], &in1[i*nx], &d_ptr[i*exblas::BIN_COUNT], &status);
    if(status != 0)
        throw dg::Error(dg::Message(_ping_)<<"MPI GPU Average failed since one of the inputs contains NaN or Inf");
    h_accumulator2 = d_accumulator;
    h_accumulator.resize( h_accumulator2.size());
    exblas::reduce_mpi_cpu( ny, &h_accumulator2[0], &h_accumulator[0], comm, comm_mod, comm_mod_reduce);

    h_round.resize( ny);
    for( unsigned i=0; i<ny; i++)
        h_round[i] = exblas::cpu::Round( &h_accumulator[i*exblas::BIN_COUNT]);

    // This test checks for errors in the current stream, the error may come
    // from any kernel prior to this point not necessarily the above one
    cudaError_t code = cudaGetLastError( );
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    code = cudaMemcpy( out, &h_round[0], ny*sizeof(value_type), cudaMemcpyHostToDevice);
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
}
#endif //MPI_VERSION

}//namespace dg
