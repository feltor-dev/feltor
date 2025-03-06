#ifndef _DG_BLAS_CUDA_
#define _DG_BLAS_CUDA_
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include "exceptions.h"
#include "exblas/exdot_cuda.cuh"
#include "exblas/fpedot_cuda.cuh"
namespace dg
{
namespace blas1
{
namespace detail
{
template<class T, size_t N, class Functor, class ...PointerOrValues>
inline void doDot_fpe_dispatch( CudaTag, int * status, unsigned size, std::array<T,N>& fpe,
    Functor f, PointerOrValues ...xs_ptr)
{
    static thrust::device_vector<T> d_fpe(N, T(0));
    T * d_ptr = thrust::raw_pointer_cast( d_fpe.data());
    exblas::fpedot_gpu<T,N,Functor,PointerOrValues...>( status, size, d_ptr, f, xs_ptr...);
    cudaError_t code = cudaGetLastError( );
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    for(unsigned u=0; u<N; u++)
        fpe[u] = d_fpe[u];
}


template<class PointerOrValue1, class PointerOrValue2>
inline std::vector<int64_t> doDot_dispatch( CudaTag, int* status, unsigned
    size, PointerOrValue1 x_ptr, PointerOrValue2 y_ptr)
{
    static thrust::device_vector<int64_t> d_superacc(exblas::BIN_COUNT);
    int64_t * d_ptr = thrust::raw_pointer_cast( d_superacc.data());
    exblas::exdot_gpu( size, x_ptr,y_ptr, d_ptr, status );
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    // This test checks for errors in the current stream, the error may come
    // from any kernel prior to this point not necessarily the above one
    cudaError_t code = cudaGetLastError( );
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    code = cudaMemcpy( &h_superacc[0], d_ptr,
            exblas::BIN_COUNT*sizeof(int64_t), cudaMemcpyDeviceToHost);
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    return h_superacc;
}
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3>
inline std::vector<int64_t> doDot_dispatch( CudaTag, int* status, unsigned size, PointerOrValue1 x_ptr, PointerOrValue2 y_ptr, PointerOrValue3 z_ptr) {
    static thrust::device_vector<int64_t> d_superacc(exblas::BIN_COUNT);
    int64_t * d_ptr = thrust::raw_pointer_cast( d_superacc.data());
    exblas::exdot_gpu( size, x_ptr,y_ptr,z_ptr, d_ptr, status);
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    // This test checks for errors in the current stream, the error may come
    // from any kernel prior to this point not necessarily the above one
    cudaError_t code = cudaGetLastError( );
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    code = cudaMemcpy( &h_superacc[0], d_ptr, exblas::BIN_COUNT*sizeof(int64_t), cudaMemcpyDeviceToHost);
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    return h_superacc;
}

template<class T>
__device__
inline T get_device_element( T x, int i){
	return x;
}
template<class T>
__device__
inline T& get_device_element( T* x, int i){
	return *(x+i);
}

template<class Subroutine, class PointerOrValue, class ...PointerOrValues>
 __global__ void subroutine_kernel( int size, Subroutine f, PointerOrValue x, PointerOrValues... xs)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
        f(get_device_element(x,i), get_device_element(xs,i)...);
        //f(x[i], xs[i]...);
        //f(thrust::raw_reference_cast(*(x+i)), thrust::raw_reference_cast(*(xs+i))...);
}

template< class Subroutine, class PointerOrValue, class ...PointerOrValues>
inline void doSubroutine_dispatch( CudaTag, int size, Subroutine f, PointerOrValue x, PointerOrValues... xs)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    subroutine_kernel<Subroutine, PointerOrValue, PointerOrValues...><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, f, x, xs...);
}

template<class T, class Pointer, class BinaryOp, class UnaryOp>
inline T doReduce_dispatch( CudaTag, int size, Pointer x, T init, BinaryOp op,
        UnaryOp unary_op)
{
    return thrust::transform_reduce(thrust::cuda::par, x, x+size, unary_op,
            init, op);
}

// Note: Here the universal reference is really important vs "F f" else we copy f on every call
template<class Binary, class F, class Pointer, std::size_t ...I, class ...PointerOrValues>
__device__
inline void call_device_F( Binary && binary, F && f, Pointer y, int i, size_t* a,
        std::index_sequence<I...>, PointerOrValues ... xs)
{
    binary( f( get_device_element( xs, a[I])...), y[i]);
}

template<class Binary, class F, size_t N, class Pointer, class ...PointerOrValues>
__global__ void kronecker_kernel( int size, const size_t* sizes, Pointer y,
        Binary binary, F f, PointerOrValues ...xs)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    size_t current[N];
    for( int i = thread_id; i<size; i += grid_size)
    {
        current[0] = i%sizes[0];
        int remain = i/sizes[0];
        for( int k=1; k<N; k++)
        {
            current[k] = remain%sizes[k];
            remain = remain/sizes[k];
        }
        call_device_F( binary, f, y, i, current, std::make_index_sequence<N>(), xs...);
    }
}

template<class Binary, class F, size_t N, class Pointer, class ...PointerOrValues>
inline void doKronecker_dispatch( dg::CudaTag, Pointer y, size_t size, Binary &&
        binary, F && f, const std::array<size_t, N>& sizes, PointerOrValues ...xs)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    thrust::device_vector<size_t> tmp( sizes.begin(), sizes.end());
    kronecker_kernel<Binary,F,N,Pointer,PointerOrValues...> <<<NUM_BLOCKS, BLOCK_SIZE>>>(size,
            thrust::raw_pointer_cast(tmp.data()),
            y,
            std::forward<Binary>(binary), std::forward<F>(f),
            xs...);
}


}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_CUDA_
