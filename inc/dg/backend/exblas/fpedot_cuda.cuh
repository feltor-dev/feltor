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

template<uint NBFPE, uint  THREADS_PER_BLOCK>
__device__ void warpReduce( volatile double * a, unsigned int tid, volatile int * status)
{
    if( THREADS_PER_BLOCK >= 64)
    {
        #pragma unroll
        for(uint k = 0; k != NBFPE; ++k) {
            double x = a[k*THREADS_PER_BLOCK+tid+32];
            #pragma unroll
            for( uint i=0; i<NBFPE; i++)
            {
                double s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != 0.0) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 32)
    {
        #pragma unroll
        for(uint k = 0; k != NBFPE; ++k) {
            double x = a[k*THREADS_PER_BLOCK+tid+16];
            #pragma unroll
            for( uint i=0; i<NBFPE; i++)
            {
                double s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != 0.0) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 16)
    {
        #pragma unroll
        for(uint k = 0; k != NBFPE; ++k) {
            double x = a[k*THREADS_PER_BLOCK+tid+8];
            #pragma unroll
            for( uint i=0; i<NBFPE; i++)
            {
                double s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != 0.0) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 8)
    {
        #pragma unroll
        for(uint k = 0; k != NBFPE; ++k) {
            double x = a[k*THREADS_PER_BLOCK+tid+4];
            #pragma unroll
            for( uint i=0; i<NBFPE; i++)
            {
                double s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != 0.0) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 4)
    {
        #pragma unroll
        for(uint k = 0; k != NBFPE; ++k) {
            double x = a[k*THREADS_PER_BLOCK+tid+2];
            #pragma unroll
            for( uint i=0; i<NBFPE; i++)
            {
                double s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != 0.0) {
                *status = 2; //indicate overflow
            }
        }
    }
    if( THREADS_PER_BLOCK >= 2)
    {
        #pragma unroll
        for(uint k = 0; k != NBFPE; ++k) {
            double x = a[k*THREADS_PER_BLOCK+tid+1];
            #pragma unroll
            for( uint i=0; i<NBFPE; i++)
            {
                double s;
                a[i*THREADS_PER_BLOCK+tid] = KnuthTwoSum(a[i*THREADS_PER_BLOCK+tid], x, &s);
                x = s;
            }
            if (x != 0.0) {
                *status = 2; //indicate overflow
            }
        }
    }
}

template<uint NBFPE, uint THREADS_PER_BLOCK, class PointerOrValue1, class PointerOrValue2>
__global__ void fpeDOT(
    double *d_PartialFPEs,
    PointerOrValue1 d_a,
    PointerOrValue2 d_b,
    const uint NbElements,
    volatile int* status
) {
    __shared__ double l_fpe[NBFPE*THREADS_PER_BLOCK]; //shared variables live for a thread block
    double *a = l_fpe + threadIdx.x;
    //Initialize FPEs
    for (uint i = 0; i < NBFPE; i++)
        a[i * THREADS_PER_BLOCK] = 0;
    __syncthreads();

    //Read data from global memory and accumulate to sub-FPEs
    for(uint pos = blockIdx.x*THREADS_PER_BLOCK+threadIdx.x; pos < NbElements; pos += gridDim.x*THREADS_PER_BLOCK) {
        //double r = 0.0;
        //double x = TwoProductFMA(get_element(d_a,pos), get_element(d_b,pos), &r);
        double x = get_element(d_a,pos)*get_element(d_b,pos);
        //we do not accumulate the rest of this multiplication

        //Check if the input is sane
        if( !isfinite(x) ) *status = 1;

        #pragma unroll
        for(uint i = 0; i != NBFPE; ++i) {
            double s;
            a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
            x = s;
        }
        if (x != 0.0) {
            *status = 2; //indicate overflow
        }
    }
    __syncthreads();

    //Tree reduction
    int modulo_factor = THREADS_PER_BLOCK/2;
    while( modulo_factor >=  64 )
    {
        if(threadIdx.x < modulo_factor) {
            for( uint k=0; k<NBFPE; k++)
            {
                double x = a[k*THREADS_PER_BLOCK+modulo_factor];
                #pragma unroll
                for(uint i = 0; i != NBFPE; ++i) {
                    double s;
                    a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
                    x = s;
                }
                if (x != 0.0) {
                    *status = 2; //indicate overflow
                }
            }
        }
        modulo_factor/=2;
        __syncthreads();
    }
    //Now merge sub-FPEs within each warp
    if( threadIdx.x < 32) warpReduce<NBFPE, THREADS_PER_BLOCK>( l_fpe, threadIdx.x, status);
    if( threadIdx.x == 0)
    {
        for( uint i=0; i<NBFPE; i++)
            d_PartialFPEs[blockIdx.x + i*gridDim.x] = l_fpe[i*THREADS_PER_BLOCK];
    }
}
template<uint NBFPE, uint THREADS_PER_BLOCK, class PointerOrValue1, class PointerOrValue2, class PointerOrValue3>
__global__ void fpeDOT(
    double *d_PartialFPEs,
    PointerOrValue1 d_a,
    PointerOrValue2 d_b,
    PointerOrValue3 d_c,
    const uint NbElements,
    volatile int* status
) {
    __shared__ double l_fpe[NBFPE*THREADS_PER_BLOCK]; //shared variables live for a thread block
    double *a = l_fpe + threadIdx.x;
    //Initialize FPEs
    for (uint i = 0; i < NBFPE; i++)
        a[i * THREADS_PER_BLOCK] = 0;
    __syncthreads();

    //Read data from global memory and accumulate to sub-FPEs
    for(uint pos = blockIdx.x*THREADS_PER_BLOCK+threadIdx.x; pos < NbElements; pos += gridDim.x*THREADS_PER_BLOCK) {
        //double r = 0.0;
        //double x = TwoProductFMA(get_element(d_a,pos), get_element(d_b,pos), &r);
        double x1 = get_element(d_a,pos)*get_element(d_b,pos);
        double x = x1*get_element(d_c,pos);
        //we do not accumulate the rest of this multiplication

        //Check if the input is sane
        if( !isfinite(x) ) *status = 1;

        #pragma unroll
        for(uint i = 0; i != NBFPE; ++i) {
            double s;
            a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
            x = s;
        }
        if (x != 0.0) {
            *status = 2; //indicate overflow
        }
    }
    __syncthreads();

    //Tree reduction
    for(int modulo_factor = THREADS_PER_BLOCK/2; modulo_factor >= 64; modulo_factor/=2 )
    {
        if(threadIdx.x < modulo_factor) {
            for( uint k=0; k<NBFPE; k++)
            {
                double x = a[k*THREADS_PER_BLOCK+modulo_factor];
                #pragma unroll
                for(uint i = 0; i != NBFPE; ++i) {
                    double s;
                    a[i*THREADS_PER_BLOCK] = KnuthTwoSum(a[i*THREADS_PER_BLOCK], x, &s);
                    x = s;
                }
                if (x != 0.0) {
                    *status = 2; //indicate overflow
                }
            }
        }
        __syncthreads();
    }
    //Now merge sub-FPEs within each warp
    if( threadIdx.x < 32) warpReduce<NBFPE, THREADS_PER_BLOCK>( l_fpe, threadIdx.x, status);
    if( threadIdx.x == 0)
    {
        for( uint i=0; i<NBFPE; i++)
            d_PartialFPEs[blockIdx.x + i*gridDim.x] = l_fpe[i*THREADS_PER_BLOCK];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////

template<uint NBFPE, uint NUM_FPES> //# of FPEs to merge (max 64)
__global__
void fpeDOTMerge(
     double *d_PartialFPEs,
     double *d_FPE,
     volatile int * status
) {
    //d_a holds max 64 FPEs
    //There may only be one block with 32 threads
    //Merge sub-FPEs within each warp
    if( threadIdx.x < 32) warpReduce<NBFPE, NUM_FPES>( d_PartialFPEs, threadIdx.x, status);
    if( threadIdx.x == 0)
    {
        for( uint i=0; i<NBFPE; i++)
            d_FPE[i] = d_PartialFPEs[i*NUM_FPES];
    }
}

static constexpr uint THREADS_PER_BLOCK           = 512; //# threads per block
static constexpr uint NUM_FPES                    = 64; //# of blocks

}//namespace gpu
///@endcond

/*!@brief gpu version of exact dot product
 *
 * Computes the exact sum \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param d_fpe pointer to an array of NBFPE doubles (the FPE) in device memory (contents are overwritten)
 * @param status 0 indicates success, 1 indicates an input value was NaN or Inf, 2 indicates an overflow of the FPE
 * @sa \c exblas::gpu::Round to convert the FPE into a double precision number
*/
template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE>
__host__
void fpedot_gpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, double* d_fpe, int* status)
{
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");

    static thrust::device_vector<double> d_PartialFPEsV( gpu::NUM_FPES*NBFPE, 0.0);
    double *d_PartialFPEs = thrust::raw_pointer_cast( d_PartialFPEsV.data());
    thrust::device_vector<int> d_statusV(1, 0);
    int *d_status = thrust::raw_pointer_cast( d_statusV.data());
    gpu::fpeDOT<NBFPE, gpu::THREADS_PER_BLOCK, PointerOrValue1, PointerOrValue2><<<gpu::NUM_FPES, gpu::THREADS_PER_BLOCK>>>(
        d_PartialFPEs, x1_ptr, x2_ptr,size, d_status);
    gpu::fpeDOTMerge<NBFPE,gpu::NUM_FPES><<<1, 32>>>( d_PartialFPEs, d_fpe, d_status);
    *status = d_statusV[0];
}

/*!@brief gpu version of exact triple dot product
 *
 * Computes the exact sum \f[ \sum_{i=0}^{N-1} x_i w_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param x3_ptr third array
 * @param d_fpe pointer to an array of NBFPE doubles (the FPE) in device memory (contents are overwritten)
 * @param status 0 indicates success, 1 indicates an input value was NaN or Inf, 2 indicates an overflow of the FPE
 * @sa \c exblas::gpu::Round to convert the FPE into a double precision number
 */
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3, size_t NBFPE>
__host__
void fpedot_gpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, PointerOrValue3 x3_ptr, double* d_fpe, int* status)
{
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue3>::value, "PointerOrValue3 needs to be T or T* with T one of (const) float or (const) double");
    static thrust::device_vector<double> d_PartialFPEsV( gpu::NUM_FPES*NBFPE, 0.0);
    double *d_PartialFPEs = thrust::raw_pointer_cast( d_PartialFPEsV.data());
    thrust::device_vector<int> d_statusV(1, 0);
    int *d_status = thrust::raw_pointer_cast( d_statusV.data());
    gpu::fpeDOT<NBFPE, gpu::THREADS_PER_BLOCK, PointerOrValue1, PointerOrValue2, PointerOrValue3><<<gpu::NUM_FPES, gpu::THREADS_PER_BLOCK>>>(
            d_PartialFPEs, x1_ptr, x2_ptr, x3_ptr, size, d_status);
    gpu::fpeDOTMerge<NBFPE,gpu::NUM_FPES><<<1, 32>>>( d_PartialFPEs, d_fpe, d_status);
    *status = d_statusV[0];
}

}//namespace exblas
} //namespace dg
