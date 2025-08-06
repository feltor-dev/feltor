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

template<class T, unsigned N, unsigned  THREADS_PER_BLOCK>
__device__ void warpReduce64( T * a, unsigned int tid)
{
    // We reduce 64 FPEs, THREADS_PER_BLOCK is the stride to access FPEs
    // assert( 64 == 2*(max_tid+1)) i.e. tid = 0,1,....,31 i.e. a warp
    // This is a tree sum that "folds a onto itself"
    // 1. In each first iteration the 2nd half of a is summed onto the first half
    // 2. until there is only one left

    // EVERY THREAD IN A WARP SHOULD DO THE SAME INSTRUCTION (so no if(threadIdx) allowed)
    int modulo_factor = 64/2;
    while( modulo_factor >=  1 )
    {
        for( unsigned k=0; k<N; k++)
        {
            // The reads and writes overlap here (but not the interesting part, so no sync
            // we don't care what happens in the overlap part))
            T x = a[k*THREADS_PER_BLOCK+tid+modulo_factor];
            #pragma unroll
            for(unsigned i = 0; i != N; ++i) {
                T s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            //if (x != T(0)) { // can't indicate nan at this point ( branch not allowed)
            //    *status = 2; //indicate overflow
            //}
        }
        modulo_factor/=2;
        // We need a memory barrier between iterations
        // (because threads need to read what other threads computed previously)
        // Otherwise results are wrong!!
        // https://stackoverflow.com/questions/46467011/thread-synchronization-with-syncwarp
        __syncwarp();
    }
}

template<class T, unsigned N, unsigned THREADS_PER_BLOCK, class Functor, class ...PointerOrValues>
__global__ void fpeDOT(
    volatile int* status,
    const unsigned NbElements,
    T *d_PartialFPEs,
    Functor f,
    PointerOrValues ...d_xs
) {
    // Dynamic shared memory
    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
    // and
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    //extern __shared__ T l_fpe[]; //shared variables live for a thread block
    extern __shared__ unsigned char memory[];
    T *l_fpe = reinterpret_cast<T *>(memory);
    T *a = l_fpe + threadIdx.x;
    //Initialize FPEs
    for (unsigned i = 0; i < N; i++)
        a[i * THREADS_PER_BLOCK] = T(0);
    __syncthreads();

    //Read data from global memory and accumulate to sub-FPEs
    for(unsigned pos = blockIdx.x*THREADS_PER_BLOCK+threadIdx.x; pos < NbElements; pos += gridDim.x*THREADS_PER_BLOCK) {
        T x = f(get_element(d_xs,pos)...);

        //if( (x -x) != T(0) ) *status = 1;

        #pragma unroll
        for(unsigned i = 0; i != N; ++i) {
            T s;
            a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
            x = s;
        }
        if (x != T(0)) {
            *status = 2; //indicate overflow (here is where Nan is caught)
        }
    }
    __syncthreads();

    //Tree reduction ("fold FPEs on themselves")
    int modulo_factor = THREADS_PER_BLOCK/2;
    while( modulo_factor >=  64 )
    {
        if(threadIdx.x < modulo_factor) {
            for( unsigned k=0; k<N; k++)
            {
                T x = a[k*THREADS_PER_BLOCK+modulo_factor];
                #pragma unroll
                for(unsigned i = 0; i != N; ++i) {
                    T s;
                    a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
                    x = s;
                }
                if (x != T(0)) {
                    *status = 2; //indicate overflow (here is where Nan is caught as well)
                }
            }
        }
        modulo_factor/=2;
        __syncthreads();
    }
    // We are now left with 64 FPEs in the block
    //Now merge sub-FPEs within each warp
    if( threadIdx.x < 32) warpReduce64<T, N, THREADS_PER_BLOCK>( l_fpe, threadIdx.x);
    if( threadIdx.x == 0)
    {
        for( unsigned i=0; i<N; i++)
            d_PartialFPEs[blockIdx.x + i*gridDim.x] = l_fpe[i*THREADS_PER_BLOCK];
    }

}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////

template<class T, unsigned N, unsigned NUM_FPES> //# of FPEs to merge (max 64)
__global__
void fpeDOTMerge(
     T *d_PartialFPEs,
     T *d_FPE
) {
    //d_a holds max 64 FPEs
    //There may only be one block with 32 threads
    //Merge sub-FPEs within each warp
    if( threadIdx.x < 32) warpReduce64<T, N, NUM_FPES>( d_PartialFPEs, threadIdx.x);
    if( threadIdx.x == 0)
    {
        for( unsigned i=0; i<N; i++)
            d_FPE[i] = d_PartialFPEs[i*NUM_FPES];
    }
}

static constexpr unsigned THREADS_PER_BLOCK           = 512; //# threads per block >= 64
static constexpr unsigned NUM_FPES                    = 64; //# of blocks

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
    static thrust::device_vector<T> d_PartialFPEsV( gpu::NUM_FPES*N, T(0));
    T *d_PartialFPEs = thrust::raw_pointer_cast( d_PartialFPEsV.data());
    thrust::device_vector<int> d_statusV(1, 0);
    int *d_status = thrust::raw_pointer_cast( d_statusV.data());
    gpu::fpeDOT<T, N, gpu::THREADS_PER_BLOCK, Functor, PointerOrValues...>
        <<<gpu::NUM_FPES, gpu::THREADS_PER_BLOCK, N*gpu::THREADS_PER_BLOCK*sizeof(T)>>>(
            d_status, size, d_PartialFPEs, f, xs_ptr...);
    gpu::fpeDOTMerge<T, N, gpu::NUM_FPES><<<1, 32>>>( d_PartialFPEs, fpe);
    *status = d_statusV[0];
}

}//namespace exblas
} //namespace dg
