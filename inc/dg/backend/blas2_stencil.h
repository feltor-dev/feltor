#pragma once
#include "config.h"
#include "exceptions.h"
#include "execution_policy.h"

namespace dg
{
namespace blas2
{
namespace detail
{

template< class Stencil, class PointerOrValue, class ...PointerOrValues>
inline void doParallelFor_dispatch( SerialTag, unsigned size, Stencil f, PointerOrValue x, PointerOrValues... xs)
{
    for( unsigned i=0; i<size; i++)
        f(i, x, xs...);
}
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class Stencil, class PointerOrValue, class ...PointerOrValues>
 __global__ void stencil_kernel( unsigned size, Stencil f, PointerOrValue x, PointerOrValues... xs)
{
    const unsigned thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( unsigned i = thread_id; i<size; i += grid_size)
        f(i, x, xs...);
}

template< class Stencil, class PointerOrValue, class ...PointerOrValues>
inline void doParallelFor_dispatch( CudaTag, unsigned size, Stencil f, PointerOrValue x, PointerOrValues... xs)
{
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    stencil_kernel<Stencil, PointerOrValue, PointerOrValues...><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, f, x, xs...);
}

#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP

constexpr int MIN_SIZE=100;//don't parallelize if work is too small

template< class Stencil, class PointerOrValue, class ...PointerOrValues>
inline void doParallelFor_omp( unsigned size, Stencil f, PointerOrValue x, PointerOrValues... xs)
{
#pragma omp for nowait
    for( int i=0; i<(int)size; i++) // MSVC OpenMP-2 only allows int (not unsigned)
        f(i, x, xs...);
}

template< class Stencil, class PointerOrValue, class ...PointerOrValues>
inline void doParallelFor_dispatch( OmpTag, unsigned size, Stencil f, PointerOrValue x, PointerOrValues... xs)
{
    if(omp_in_parallel())
    {
        doParallelFor_omp( size, f, x, xs... );
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doParallelFor_omp( size, f, x, xs...);
        }
    }
    else
        doParallelFor_dispatch( SerialTag(), size, f, x, xs...);
}
#endif


}//namespace detail
}//namespace blas1
}//namespace dg
