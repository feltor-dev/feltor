#ifndef _DG_BLAS_CUDA_
#define _DG_BLAS_CUDA_
#include <thrust/transform.h>
#include "exblas/exdot.fpe.cu"
namespace dg
{
namespace blas1
{
namespace detail
{
exblas::Superaccumulator doDot_dispatch( CudaTag, unsigned size, const double* x_ptr, const double * y_ptr) {
    return exblas::Superaccumulator(  exblas::exdot_gpu( size, x_ptr,y_ptr)) ;
}
template< class Vector, class UnaryOp>
inline void doTransform_dispatch( CudaTag, const Vector& x, Vector& y, UnaryOp op) {
    thrust::transform( thrust::cuda::tag(), x.begin(), x.end(), y.begin(), op);
}
template< typename value_type>
struct Axpby_Functor
{
    Axpby_Functor( value_type alpha, value_type beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        value_type operator()( const value_type& x, const value_type& y)
        {
            return alpha*x+beta*y;
        }
    __host__ __device__
        value_type operator()( const value_type& y)
        {
            return beta*y;
        }
  private:
    value_type alpha, beta;
};
template< class T>
inline void doScal_dispatch( CudaTag, unsigned size, T* x, T alpha) {
    thrust::transform( thrust::cuda::tag(), x, x+size, x, detail::Axpby_Functor<T>( 0, alpha));
}
template <class value_type>
struct Plus_Functor
{
    Plus_Functor( value_type alpha): alpha(alpha){}
    
    __host__ __device__
        value_type operator()( const value_type& x)
        {
            return alpha+x;
        }
  private:
    value_type alpha;
};
template<class T>
inline void doPlus_dispatch( CudaTag, unsigned size, T* x, T alpha)
{
    thrust::transform( thrust::cuda::tag(), x, x+size, x, 
            detail::Plus_Functor<T>( alpha));
}
template<class T>
void doAxpby_dispatch( CudaTag, unsigned size, T alpha, const T * RESTRICT x, T beta, T* RESTRICT y)
{
    if( beta != 0)
        thrust::transform( thrust::cuda::tag(), x, x+size, y, y, 
            detail::Axpby_Functor< T>( alpha, beta) );
    else 
        thrust::transform( thrust::cuda::tag(), x, x+size, y,
            detail::Axpby_Functor< T>( 0., alpha) );
}
template<class value_type>
 __global__ void axpbypgz_kernel( value_type alpha, value_type beta, value_type gamma,
         const value_type* RESTRICT x, const value_type* RESTRICT y, value_type* RESTRICT z, const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<size; row += grid_size)
        z[row]=alpha*x[row]+beta*y[row]+gamma*z[row];
}
template<class T>
void doAxpbypgz_dispatch( CudaTag, unsigned size, T alpha, const T * RESTRICT x, T beta, const T* RESTRICT y, T gamma, T* RESTRICT z)
{
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    axpbypgz_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, beta, gamma, x, y, z, size);
}

template<class value_type>
 __global__ void pointwiseDot_kernel( value_type alpha, value_type gamma,
         const value_type*  x1, const value_type* y1, value_type* z,  const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    for( int row = thread_id; row<size; row += grid_size) {
        z[row]=alpha*x1[row]*y1[row]+gamma*z[row];
    }
}
template<class value_type>
inline void doPointwiseDot_dispatch( CudaTag, unsigned size, 
              value_type alpha, 
              const value_type* x1_ptr,
              const value_type* y1_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    pointwiseDot_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, gamma, x1_ptr, y1_ptr, z_ptr, size);
}
template<class value_type>
 __global__ void pointwiseDivide_kernel( value_type alpha, value_type gamma,
         const value_type*  x1, const value_type* y1, value_type* z,  const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    for( int row = thread_id; row<size; row += grid_size) {
        z[row]=alpha*x1[row]/y1[row]+gamma*z[row];
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
         const value_type*  x1, const value_type* y1, const value_type* x2, 
         const value_type*  y2, value_type* z,  const int size)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    for( int row = thread_id; row<size; row += grid_size) {
        z[row]=alpha*x1[row]*y1[row]+beta*x2[row]*y2[row]+gamma*z[row];
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
    for( int row = thread_id; row<size; row += grid_size) {
        y[row]=alpha*x1[row]*x2[row]*x3[row]+beta*y[row];
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
