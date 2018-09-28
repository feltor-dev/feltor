#ifndef _DG_BLAS_CUDA_
#define _DG_BLAS_CUDA_
#include "exblas/exdot_cuda.cuh"
namespace dg
{
namespace blas1
{
namespace detail
{

template<class PointerOrValue1, class PointerOrValue2>
inline std::vector<int64_t> doDot_dispatch( CudaTag, unsigned size, PointerOrValue1 x_ptr, PointerOrValue2 y_ptr) {
    static thrust::device_vector<int64_t> d_superacc(exblas::BIN_COUNT);
    int64_t * d_ptr = thrust::raw_pointer_cast( d_superacc.data());
    exblas::exdot_gpu( size, x_ptr,y_ptr, d_ptr);
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    cudaMemcpy( &h_superacc[0], d_ptr, exblas::BIN_COUNT*sizeof(int64_t), cudaMemcpyDeviceToHost);
    return h_superacc;
}
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3>
inline std::vector<int64_t> doDot_dispatch( CudaTag, unsigned size, PointerOrValue1 x_ptr, PointerOrValue2 y_ptr, PointerOrValue3 z_ptr) {
    static thrust::device_vector<int64_t> d_superacc(exblas::BIN_COUNT);
    int64_t * d_ptr = thrust::raw_pointer_cast( d_superacc.data());
    exblas::exdot_gpu( size, x_ptr,y_ptr,z_ptr, d_ptr);
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    cudaMemcpy( &h_superacc[0], d_ptr, exblas::BIN_COUNT*sizeof(int64_t), cudaMemcpyDeviceToHost);
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
 __global__ void evaluate_kernel( int size, Subroutine f, PointerOrValue x, PointerOrValues... xs)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
        //f(x[i], xs[i]...);
        //f(thrust::raw_reference_cast(*(x+i)), thrust::raw_reference_cast(*(xs+i))...);
        f(get_device_element(x,i), get_device_element(xs,i)...);
}

template< class Subroutine, class PointerOrValue, class ...PointerOrValues>
inline void doEvaluate_dispatch( CudaTag, int size, Subroutine f, PointerOrValue x, PointerOrValues... xs)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    evaluate_kernel<Subroutine, PointerOrValue, PointerOrValues...><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, f, x, xs...);
}


}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_CUDA_
