/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */

#include "thrust/device_vector.h"
#include "accumulate.cuh"

namespace exblas{



////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
__device__ 
double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

__device__ 
double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}


__global__ void ExDOT(
    int64_t *d_PartialSuperaccs,
    const double *d_a,
    const double *d_b,
    const uint NbElements
) {
    __shared__ int64_t l_sa[WARP_COUNT * BIN_COUNT]; //shared variables live for a thread block (39 rows, 16 columns!)
    int64_t *l_workingBase = l_sa + (threadIdx.x & (WARP_COUNT - 1)); //the bitwise & with 15 is a modulo operation: threadIdx.x % 16
    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    __syncthreads();

    //Read data from global memory and scatter it to sub-superaccs
    double a[NBFPE] = {0.0};
    for(uint pos = blockIdx.x*blockDim.x+threadIdx.x; pos < NbElements; pos += gridDim.x*blockDim.x) {
        //double r = 0.0;
        //double x = TwoProductFMA(d_a[pos], d_b[pos], &r);
        double x = d_a[pos]*d_b[pos];

        #pragma unroll
        for(uint i = 0; i != NBFPE; ++i) {
            double s;
            a[i] = KnuthTwoSum(a[i], x, &s);
            x = s;
        }
        if (x != 0.0) {
            AccumulateT(l_workingBase, x);
            // Flush FPEs to superaccs
            #pragma unroll
            for(uint i = 0; i != NBFPE; ++i) {
                AccumulateT(l_workingBase, a[i]);
                a[i] = 0.0;
            }
        }

        //if (r != 0.0) {//add the rest r in the same manner
        //    #pragma unroll
        //    for(uint i = 0; i != NBFPE; ++i) {
        //        double s;
        //        a[i] = KnuthTwoSum(a[i], r, &s);
        //        r = s;
        //    }
        //    if (r != 0.0) {
        //        AccumulateT(l_workingBase, r);
        //        // Flush FPEs to superaccs
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            AccumulateT(l_workingBase, a[i]);
        //            a[i] = 0.0;
        //        }
        //    }
        //}
    }
    //Flush FPEs to superaccs
    #pragma unroll
    for(uint i = 0; i != NBFPE; ++i)
        AccumulateT(l_workingBase, a[i]);
    __syncthreads();

    //Merge sub-superaccs into work-group partial-accumulator ( ATTENTION: PartialSuperacc is transposed!)
    uint pos = threadIdx.x;
    if(pos < WARP_COUNT) {
        NormalizeT( l_workingBase);
    }
    __syncthreads();

    if (pos < BIN_COUNT) {
        int64_t sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];

        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + pos] = sum;
    }

    __syncthreads();
    if (pos == 0) {
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT]);
    }
}

__global__ void ExDOT(
    int64_t *d_PartialSuperaccs,
    const double *d_a,
    const double *d_b,
    const double *d_c,
    const uint NbElements
) {
    __shared__ int64_t l_sa[WARP_COUNT * BIN_COUNT]; //shared variables live for a thread block (39 rows, 16 columns!)
    int64_t *l_workingBase = l_sa + (threadIdx.x & (WARP_COUNT - 1)); //the bitwise & with 15 is a modulo operation: threadIdx.x % 16
    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    __syncthreads();

    //Read data from global memory and scatter it to sub-superaccs
    double a[NBFPE] = {0.0};
    for(uint pos = blockIdx.x*blockDim.x+threadIdx.x; pos < NbElements; pos += gridDim.x*blockDim.x) {
        //double x2 = d_a[pos]*d_c[pos]*d_b[pos];
        //double r  = 0.0, r2 = 0.0;
        //double x  = TwoProductFMA(d_a[pos], d_b[pos], &r);
        //double x2 = TwoProductFMA(x , d_c[pos], &r2);
        double x2 = (d_a[pos]*d_b[pos])*d_c[pos];

        if( x2 != 0.0) {//accumulate x2
            #pragma unroll
            for(uint i = 0; i != NBFPE; ++i) {
                double s;
                a[i] = KnuthTwoSum(a[i], x2, &s);
                x2 = s;
            }
            if (x2 != 0.0) {
                AccumulateT(l_workingBase, x2);
                // Flush FPEs to superaccs
                #pragma unroll
                for(uint i = 0; i != NBFPE; ++i) {
                    AccumulateT(l_workingBase, a[i]);
                    a[i] = 0.0;
                }
            }
        }
        //if (r2 != 0.0) {//add the rest r2 
        //    #pragma unroll
        //    for(uint i = 0; i != NBFPE; ++i) {
        //        double s;
        //        a[i] = KnuthTwoSum(a[i], r2, &s);
        //        r2 = s; //error was here r = s
        //    }
        //    if (r2 != 0.0) { //error was here r != 0.0
        //        AccumulateT(l_workingBase, r2);
        //        // Flush FPEs to superaccs
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            AccumulateT(l_workingBase, a[i]);
        //            a[i] = 0.0;
        //        }
        //    }
        //}

        //if (r != 0.0) {//add the rest r*c in the same manner
        //    r2 = 0.0;
        //    x2 = TwoProductFMA(r , d_c[pos], &r2);
        //    if( x2 != 0.0) {//accumulate x2
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            double s;
        //            a[i] = KnuthTwoSum(a[i], x2, &s);
        //            x2 = s;
        //        }
        //        if (x2 != 0.0) {
        //            AccumulateT(l_workingBase, x2);
        //            // Flush FPEs to superaccs
        //            #pragma unroll
        //            for(uint i = 0; i != NBFPE; ++i) {
        //                AccumulateT(l_workingBase, a[i]);
        //                a[i] = 0.0;
        //            }
        //        }
        //    }
        //    if (r2 != 0.0) {//add the rest r2 
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            double s;
        //            a[i] = KnuthTwoSum(a[i], r2, &s);
        //            r2 = s; //error was here r = s
        //        }
        //        if (r2 != 0.0) { //error was here r != 0.0
        //            AccumulateT(l_workingBase, r2);
        //            // Flush FPEs to superaccs
        //            #pragma unroll
        //            for(uint i = 0; i != NBFPE; ++i) {
        //                AccumulateT(l_workingBase, a[i]);
        //                a[i] = 0.0;
        //            }
        //        }
        //    }
        //}
    }
	//Flush FPEs to superaccs
    #pragma unroll
    for(uint i = 0; i != NBFPE; ++i)
        AccumulateT(l_workingBase, a[i]);
    __syncthreads();

    //Merge sub-superaccs into work-group partial-accumulator ( ATTENTION: PartialSuperacc is transposed!)
    uint pos = threadIdx.x;
    if(pos<WARP_COUNT){
            NormalizeT(l_workingBase);
        }
    __syncthreads();
    if (pos < BIN_COUNT) {
        int64_t sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];

        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + pos] = sum;
    }

    __syncthreads();
    if (pos == 0) {
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////
__global__
void ExDOTComplete(
     int64_t *d_PartialSuperaccs,
     int64_t *d_superacc
) {
    uint lid = threadIdx.x;
    uint gid = blockIdx.x;

    if (lid < BIN_COUNT) {
        int64_t sum = 0;

        for(uint i = 0; i < MERGE_SUPERACCS_SIZE; i++)
            sum += d_PartialSuperaccs[(gid * MERGE_SUPERACCS_SIZE + i) * BIN_COUNT + lid];

        d_PartialSuperaccs[gid * BIN_COUNT + lid] = sum;
    }

    __syncthreads();
    if (lid == 0) {
        Normalize(&d_PartialSuperaccs[gid * BIN_COUNT]);
    }

    __syncthreads();
    if ((lid < BIN_COUNT) && (gid == 0)) {
        int64_t sum = 0;

        for(uint i = 0; i < gridDim.x; i++)
            sum += d_PartialSuperaccs[i * BIN_COUNT + lid];

        d_superacc[lid] = sum;
    }
}

////////////// parameters for Kernel execution            //////////////////////
//Kernel paramters for EXDOT
static constexpr uint WARP_SIZE                = 16 ; 
static constexpr uint WORKGROUP_SIZE           = (WARP_COUNT * WARP_SIZE); //# threads per block
static constexpr uint PARTIAL_SUPERACCS_COUNT  = 256; //# of groups; each has a partial SuperAcc (should not be larger than 512)
//Kernel paramters for EXDOTComplete
static constexpr uint MERGE_SUPERACCS_SIZE     = 64; //# of sa each block merges
static constexpr uint MERGE_WORKGROUP_SIZE     = 64;  //we need only 39 of those

//d_superacc must be a pointer to device memory with size at least BIN_COUNT 
__host__
void exdot_gpu(unsigned size, const double* x1_ptr, const double* x2_ptr, int64_t* d_superacc)
{
    static thrust::device_vector<int64_t> d_PartialSuperaccsV( PARTIAL_SUPERACCS_COUNT*BIN_COUNT, 0.0); //39 columns and PSC rows
    int64_t *d_PartialSuperaccs = thrust::raw_pointer_cast( d_PartialSuperaccsV.data());
    ExDOT<<<PARTIAL_SUPERACCS_COUNT, WORKGROUP_SIZE>>>( d_PartialSuperaccs, x1_ptr, x2_ptr,size);
    ExDOTComplete<<<PARTIAL_SUPERACCS_COUNT/MERGE_SUPERACCS_SIZE, MERGE_WORKGROUP_SIZE>>>( d_PartialSuperaccs, d_superacc );
}

__host__
void exdot_gpu(unsigned size, const double* x1_ptr, const double* x2_ptr, const double* x3_ptr, int64_t* d_superacc)
{
    static thrust::device_vector<int64_t> d_PartialSuperaccsV( PARTIAL_SUPERACCS_COUNT*BIN_COUNT, 0.0); //39 columns and PSC rows
    int64_t *d_PartialSuperaccs = thrust::raw_pointer_cast( d_PartialSuperaccsV.data());
    ExDOT<<<PARTIAL_SUPERACCS_COUNT, WORKGROUP_SIZE>>>( d_PartialSuperaccs, x1_ptr, x2_ptr, x3_ptr,size);
    ExDOTComplete<<<PARTIAL_SUPERACCS_COUNT/MERGE_SUPERACCS_SIZE, MERGE_WORKGROUP_SIZE>>>( d_PartialSuperaccs, d_superacc );
}

}//namespace exblas

