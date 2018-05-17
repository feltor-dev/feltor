#ifndef _DG_BLAS_CUDA_
#define _DG_BLAS_CUDA_
#include "exblas/exdot_cuda.cuh"
namespace dg
{
namespace blas1
{
namespace detail
{
std::vector<int64_t> doDot_dispatch( CudaTag, unsigned size, const double* x_ptr, const double * y_ptr) {
    static thrust::device_vector<int64_t> d_superacc(exblas::BIN_COUNT);
    int64_t * d_ptr = thrust::raw_pointer_cast( d_superacc.data());
    exblas::exdot_gpu( size, x_ptr,y_ptr, d_ptr);
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    cudaMemcpy( &h_superacc[0], d_ptr, exblas::BIN_COUNT*sizeof(int64_t), cudaMemcpyDeviceToHost);
    return h_superacc;
}
template<class Op, class B, class T>
 __global__ void evaluate_kernel( int size, T* y, B f, Op op,
         const T* x)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
        f(y[i], op(x[i]) );
}

template< class UnaryOp, class Binary, class T>
inline void doEvaluate_dispatch( CudaTag, unsigned size, T* z, Binary f, UnaryOp op, const T* x) {
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    evaluate_kernel<UnaryOp, Binary, T><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, z, f, op,x);
}
template<class Op, class Binary, class T>
 __global__ void evaluate_kernel( int size, T* z, Binary f, Op op,
         const T* x, const T* y)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
        f(z[i], op(x[i], y[i]) );
}

template< class UnaryOp, class Binary, class T>
inline void doEvaluate_dispatch( CudaTag, unsigned size, T* z, Binary f, UnaryOp op, const T* x, const T* y) {
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    evaluate_kernel<UnaryOp, Binary, T><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, z, f, op, x,y);
}

template<class value_type>
 __global__ void scal_kernel( value_type alpha,
         value_type* __restrict__ x, const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
        x[i]*=alpha;
}
template< class T>
inline void doScal_dispatch( CudaTag, unsigned size, T* x, T alpha) {
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    scal_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, x, size);
}

template<class value_type>
 __global__ void plus_kernel( value_type alpha,
         value_type* __restrict__ x, const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
        x[i]+=alpha;
}
template<class T>
inline void doPlus_dispatch( CudaTag, unsigned size, T* x, T alpha)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    plus_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, x, size);
}
template<class value_type>
 __global__ void axpby_kernel( value_type alpha, value_type beta,
         const value_type* __restrict__ x_ptr, value_type* __restrict__ y_ptr, const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
    {
        double temp = y_ptr[i]*beta;
        y_ptr[i] = fma( alpha,x_ptr[i], temp);
    }
}
template<class T>
void doAxpby_dispatch( CudaTag, unsigned size, T alpha, const T * __restrict__ x, T beta, T* __restrict__ y)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    axpby_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, beta, x, y, size);
}
template<class value_type>
 __global__ void axpbypgz_kernel( value_type alpha, value_type beta, value_type gamma,
         const value_type* __restrict__ x_ptr, const value_type* __restrict__ y_ptr, value_type* __restrict__ z_ptr, const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
    {
        double temp = z_ptr[i]*gamma;
        temp = fma( alpha,x_ptr[i], temp);
        temp = fma( beta, y_ptr[i], temp);
        z_ptr[i] = temp;
    }
}
template<class T>
void doAxpbypgz_dispatch( CudaTag, unsigned size, T alpha, const T * __restrict__ x, T beta, const T* __restrict__ y, T gamma, T* __restrict__ z)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    axpbypgz_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, beta, gamma, x, y, z, size);
}

template<class value_type>
 __global__ void pointwiseDot_kernel( value_type alpha, value_type gamma,
         const value_type*  x_ptr, const value_type* y_ptr, value_type* z_ptr,  const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    for( int i = thread_id; i<size; i += grid_size)
    {
        double temp = z_ptr[i]*gamma;
        z_ptr[i] = fma( alpha*x_ptr[i], y_ptr[i], temp);
    }
}
template<class value_type>
inline void doPointwiseDot_dispatch( CudaTag, unsigned size,
              value_type alpha,
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    pointwiseDot_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, gamma, x_ptr, y_ptr, z_ptr, size);
}
template<class value_type>
 __global__ void pointwiseDivide_kernel( value_type alpha, value_type gamma,
         const value_type*  x_ptr, const value_type* y_ptr, value_type* z_ptr,  const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    for( int i = thread_id; i<size; i += grid_size)
    {
        double temp = z_ptr[i]*gamma;
        z_ptr[i] = fma( alpha, x_ptr[i]/y_ptr[i], temp);
    }
}
template<class value_type>
inline void doPointwiseDivide_dispatch( CudaTag, unsigned size,
              value_type alpha,
              const value_type* x1_ptr,
              const value_type* y1_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    pointwiseDivide_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, gamma, x1_ptr, y1_ptr, z_ptr, size);
}

template<class value_type>
 __global__ void pointwiseDot_kernel( value_type alpha, value_type beta, value_type gamma,
         const value_type*  x1_ptr, const value_type* y1_ptr, const value_type* x2_ptr,
         const value_type*  y2_ptr, value_type* z_ptr,  const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    for( int i = thread_id; i<size; i += grid_size)
    {
        double temp = z_ptr[i]*gamma;
        temp = fma( alpha*x1_ptr[i], y1_ptr[i], temp);
        temp = fma(  beta*x2_ptr[i], y2_ptr[i], temp);
        z_ptr[i] = temp;
    }
}
template<class value_type>
inline void doPointwiseDot_dispatch( CudaTag, unsigned size,
              value_type alpha,
              const value_type* x1_ptr,
              const value_type* y1_ptr,
              value_type beta,
              const value_type* x2_ptr,
              const value_type* y2_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    pointwiseDot_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, beta, gamma, x1_ptr, y1_ptr, x2_ptr, y2_ptr, z_ptr, size);
}

template<class value_type>
 __global__ void pointwiseDot_kernel( value_type alpha, value_type beta,
         const value_type*  x1, const value_type* x2, const value_type* x3,
         value_type* y,
         const int size
         )
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    for( int i = thread_id; i<size; i += grid_size)
    {
        double temp = y[i]*beta;
        y[i] = fma( alpha*x1[i], x2[i]*x3[i], temp);
    }
}
template<class value_type>
inline void doPointwiseDot_dispatch( CudaTag, unsigned size,
              value_type alpha,
              const value_type* x1_ptr,
              const value_type* x2_ptr,
              const value_type* x3_ptr,
              value_type beta,
              value_type* y_ptr)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    pointwiseDot_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, beta, x1_ptr, x2_ptr, x3_ptr, y_ptr, size);
}
}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_CUDA_
