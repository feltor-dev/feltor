#pragma once
#include <thrust/device_vector.h>
#include "exblas/exdot_cuda.cuh"

namespace dg
{
namespace blas2
{
namespace detail
{

template<class T>
__device__
inline T gpuKnuthTwoFMA(T a, T b, T c, T & s)
{
    T r = __fma_rn( a,b,c);
    T z = r - c;
    s = (c - (r - z)) + __fma_rn( a,b, -z);
    return r;
}

template<class T, unsigned NBFPE>
__device__
void gpuAccumulateFPE( T a, T b, T* fpe)
{
    T s;
    fpe[0] = gpuKnuthTwoFMA( a, b, fpe[0], s);
    for(unsigned i = 1; i != NBFPE-1; ++i) {
        T x = s;
        fpe[i] = dg::exblas::gpu::KnuthTwoSum(x, fpe[i], &s);
    }
    fpe[NBFPE-1] += s; // we throw away the rest
}


template<class T, unsigned NBFPE>
__global__
void doDenseSymv_gpu(unsigned num_rows, unsigned num_cols, T alpha,
        const T * const * m_ptr, const T* __restrict__ x,
        T beta, T* __restrict__ y)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_is/grid_size is
    for( int i = thread_id; i<num_rows; i += grid_size)
    {
        T fpe [NBFPE] = {0};
        for( unsigned k=0; k<num_cols; k++)
        {
            T a = m_ptr[k][i];
            T b = x[k];
            gpuAccumulateFPE<T,NBFPE>( a,b, fpe);
        }
        // multiply fpe with alpha
        T fpe2 [NBFPE] = {0};
        for( unsigned k=0; k<NBFPE; k++)
            gpuAccumulateFPE<T,NBFPE>( alpha, fpe[k], fpe2);
        // Finally add beta*y
        gpuAccumulateFPE<T,NBFPE>( beta, y[i], fpe2);
        // Finally sum up everything starting with smallest value
        y[i] = 0;
        for( int k=(int)NBFPE-1; k>=0; k--)
            // round to nearest
            y[i] = y[i] + fpe2[k];
    }
}


template<class T, class Vector1>
void doDenseSymv(CudaTag, unsigned num_rows, unsigned num_cols, T alpha,
        const std::vector<const T*>& m_ptr, const Vector1& x,
        T beta, T* y)
{
    constexpr unsigned NBFPE = 2;
    const size_t BLOCK_SIZE = 256;
    const size_t NUM_BLOCKS = std::min<size_t>((num_rows-1)/BLOCK_SIZE+1, 65000);
    thrust::device_vector<const T*> m_dev( m_ptr.begin(), m_ptr.end());
    thrust::device_vector<T> x_dev( x.begin(), x.end());
    T* x_ptr = thrust::raw_pointer_cast( x_dev.data());
    thrust::device_vector<T> m_dev0( m_dev[0]+0, m_dev[0]+3);
    T const * const * m_dev_ptr = thrust::raw_pointer_cast( m_dev.data());
    doDenseSymv_gpu<T,NBFPE><<<NUM_BLOCKS, BLOCK_SIZE>>>(num_rows, num_cols,
            alpha, m_dev_ptr, x_ptr, beta, y);
}

}//namespace detail
}//namespace blas2
}//namespace dg
