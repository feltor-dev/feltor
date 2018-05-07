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
template<class Subroutine, class T, class ...Ts>
 __global__ void subroutine_kernel( int size, Subroutine f, T* x, Ts*... xs)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<size; i += grid_size)
        f(x[i], xs[i]... );
}
template< class Subroutine, class T, class ...Ts>
inline void doSubroutine_dispatch( CudaTag, int size, Subroutine f, T* x, Ts*... xs)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    subroutine_kernel<Subroutine, T, ...Ts><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, f, x, xs...);
}


}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_CUDA_
