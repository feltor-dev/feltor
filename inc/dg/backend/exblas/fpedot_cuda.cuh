/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Matthias Wiesenberger, 2020, within FELTOR license
 */
/**
 *  @file fpedot_cuda.cuh
 *  @brief CUDA version of fpedot
 *
 *  @authors
 *    Developers : \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include "thrust/device_vector.h"
#include "accumulate.cuh"

namespace dg
{
namespace exblas{
///@cond
namespace gpu{

////////////////////////////////////////////////////////////////////////////////
// Main Kernels
////////////////////////////////////////////////////////////////////////////////
//In the first kernel each block produces exactly one FPE (because threads within a block can be synchronized and shared memory lives for a block )
//the second kernel reduces all FPEs from the first kernel (because we need a global synchronization, which is induced by separate kernel launches)

template<class T, uint N, uint  THREADS_PER_BLOCK>
__device__ void warpReduce( T * a, unsigned int tid, volatile int * status)
{
    // assert( THREADS_PER_BLOCK == 2*(max_tid+1))
    // a has size THREADS_PER_BLOCK
    // This is a manually unrolled tree sum
    // 0. Currently THREADS_PER_BLOCK = 512 (first call) = 64 (second call) and tid = 0,1,...,31
    // 1. In each first iteration the 2nd half of a is summed onto the first half
    // 2. until there is only two left

    if( THREADS_PER_BLOCK >= 64)
    {
        #pragma unroll
        for(uint k = 0; k != N; ++k) {
            T x = a[k*THREADS_PER_BLOCK+tid+32];
            #pragma unroll
            for( uint i=0; i<N; i++)
            {
                T s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != T(0)) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 32)
    {
        #pragma unroll
        for(uint k = 0; k != N; ++k) {
            T x = a[k*THREADS_PER_BLOCK+tid+16];
            #pragma unroll
            for( uint i=0; i<N; i++)
            {
                T s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != T(0)) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 16)
    {
        #pragma unroll
        for(uint k = 0; k != N; ++k) {
            T x = a[k*THREADS_PER_BLOCK+tid+8];
            #pragma unroll
            for( uint i=0; i<N; i++)
            {
                T s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != T(0)) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 8)
    {
        #pragma unroll
        for(uint k = 0; k != N; ++k) {
            T x = a[k*THREADS_PER_BLOCK+tid+4];
            #pragma unroll
            for( uint i=0; i<N; i++)
            {
                T s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != T(0)) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 4)
    {
        #pragma unroll
        for(uint k = 0; k != N; ++k) {
            T x = a[k*THREADS_PER_BLOCK+tid+2];
            #pragma unroll
            for( uint i=0; i<N; i++)
            {
                T s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != T(0)) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 2)
    {
        #pragma unroll
        for(uint k = 0; k != N; ++k) {
            T x = a[k*THREADS_PER_BLOCK+tid+1];
            #pragma unroll
            for( uint i=0; i<N; i++)
            {
                T s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != T(0)) {
                *status = 2; //indicate overflow
            }
        }
    }
}

template<class T, uint N, uint THREADS_PER_BLOCK, class Functor, class ...PointerOrValues>
__global__ void fpeDOT(
    volatile int* status,
    const uint NbElements,
    T *d_PartialFPEs,
    Functor f,
    PointerOrValues ...d_xs
) {
    // MW: this generates a warning for thrust::complex<double> (maybe for cuda::std::complex it does not?)
    __shared__ T l_fpe[N*THREADS_PER_BLOCK]; //shared variables live for a thread block
    T *a = l_fpe + threadIdx.x;
    //Initialize FPEs
    for (uint i = 0; i < N; i++)
        a[i * THREADS_PER_BLOCK] = T(0);
    __syncthreads();

    //Read data from global memory and accumulate to sub-FPEs
    for(uint pos = blockIdx.x*THREADS_PER_BLOCK+threadIdx.x; pos < NbElements; pos += gridDim.x*THREADS_PER_BLOCK) {
        T x = f(get_element(d_xs,pos)...);

        //Check if the input is sane
        //does not work for complex
        //if( !isfinite(x) ) *status = 1;

        #pragma unroll
        for(uint i = 0; i != N; ++i) {
            T s;
            a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
            x = s;
        }
        if (x != T(0)) {
            *status = 2; //indicate overflow
        }
    }
    __syncthreads();

    //Tree reduction
    int modulo_factor = THREADS_PER_BLOCK/2;
    while( modulo_factor >=  64 )
    {
        if(threadIdx.x < modulo_factor) {
            for( uint k=0; k<N; k++)
            {
                T x = a[k*THREADS_PER_BLOCK+modulo_factor];
                #pragma unroll
                for(uint i = 0; i != N; ++i) {
                    T s;
                    a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
                    x = s;
                }
                if (x != T(0)) {
                    *status = 2; //indicate overflow
                }
            }
        }
        modulo_factor/=2;
        __syncthreads();
    }
    //Now merge sub-FPEs within each warp
    if( threadIdx.x < 32) warpReduce<T, N, THREADS_PER_BLOCK>( l_fpe, threadIdx.x, status);
    if( threadIdx.x == 0)
    {
        for( uint i=0; i<N; i++)
            d_PartialFPEs[blockIdx.x + i*gridDim.x] = l_fpe[i*THREADS_PER_BLOCK];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////

template<class T, uint N, uint NUM_FPES> //# of FPEs to merge (max 64)
__global__
void fpeDOTMerge(
     T *d_PartialFPEs,
     T *d_FPE,
     volatile int * status
) {
    //d_a holds max 64 FPEs
    //There may only be one block with 32 threads
    //Merge sub-FPEs within each warp
    if( threadIdx.x < 32) warpReduce<T, N, NUM_FPES>( d_PartialFPEs, threadIdx.x, status);
    if( threadIdx.x == 0)
    {
        for( uint i=0; i<N; i++)
            d_FPE[i] = d_PartialFPEs[i*NUM_FPES];
    }
}

static constexpr uint THREADS_PER_BLOCK           = 512; //# threads per block
static constexpr uint NUM_FPES                    = 64; //# of blocks

}//namespace gpu
///@endcond


/*!@brief GPU version of fpe generalized dot product
 *
 * @copydetails fpedot_cpu
 */
template<class T, size_t N, class Functor, class ...PointerOrValues>
__host__
void fpedot_gpu(int * status, unsigned size, T* fpe, Functor f, PointerOrValues ...xs_ptr)
{
    static thrust::device_vector<T> d_PartialFPEsV( gpu::NUM_FPES*N, 0.0);
    T *d_PartialFPEs = thrust::raw_pointer_cast( d_PartialFPEsV.data());
    thrust::device_vector<int> d_statusV(1, 0);
    int *d_status = thrust::raw_pointer_cast( d_statusV.data());
    gpu::fpeDOT<T, N, gpu::THREADS_PER_BLOCK, Functor, PointerOrValues...><<<gpu::NUM_FPES, gpu::THREADS_PER_BLOCK>>>(
            d_status, size, d_PartialFPEs, f, xs_ptr...);

    gpu::fpeDOTMerge<T, N, gpu::NUM_FPES><<<1, 32>>>( d_PartialFPEs, fpe, d_status);
    *status = d_statusV[0];
}

}//namespace exblas
} //namespace dg
