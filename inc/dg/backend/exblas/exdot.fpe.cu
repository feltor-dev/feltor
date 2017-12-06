/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */
#include "thrust/device_vector.h"

namespace exblas{

static constexpr uint BIN_COUNT     =  39; //size of superaccumulator
static constexpr uint NBFPE         =  3;  //size of floating point expansion
////////////// parameters for superaccumulator operations //////////////////////
static constexpr int KRX            =  8;  //High-radix carry-save bits
static constexpr int DIGITS         =  64 - KRX; //must be int because appears in integer expresssion
static constexpr int F_WORDS        =  20;
//static constexpr int TSAFE          =  0;
static constexpr double DELTASCALE = double(1ull << DIGITS); // Assumes KRX>0

////////////// parameters for Kernel execution            //////////////////////
//Kernel paramters for EXDOT
static constexpr uint WARP_COUNT               = 16 ; //# of sub superaccs
static constexpr uint WARP_SIZE                = 16 ; 
static constexpr uint WORKGROUP_SIZE           = (WARP_COUNT * WARP_SIZE); //# threads per block
static constexpr uint PARTIAL_SUPERACCS_COUNT  = 128; //# of groups; each has a partial SuperAcc (somehow does not work for 128???)
//Kernel paramters for EXDOTComplete
static constexpr uint MERGE_SUPERACCS_SIZE     = 128; //# of sa each block merges
static constexpr uint MERGE_WORKGROUP_SIZE     = 64;  //we need only 39 of those


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

//returns the original value at address
__device__ long long int atomicAdd( long long int* address, long long int val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old; //*address_as_ull might change during the time the CAS is reached
        old = atomicCAS(address_as_ull, assumed,
                          (unsigned long long int)(val + (long long int)old));
    } while( old != assumed);//try as often as necessary
    //assume that bit patterns don't change when casting
    //return the original value stored at address
    return (long long int)(old);
}
// signedcarry in {-1, 0, 1}
__device__ long long int xadd( long long int *sa, long long int x, unsigned char *of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    //long long int y = atom_add(sa, x);
    long long int y = atomicAdd(sa, x); 
    long long int z = y + x; // since the value sa->superacc[i] can be changed by another work item

    // TODO: cover also underflow
    *of = 0;
    if(x > 0 && y > 0 && z < 0)
        *of = 1;
    if(x < 0 && y < 0 && z > 0)
        *of = 1;

    return y;
}


////////////////////////////////////////////////////////////////////////////////
// Rounding functions
////////////////////////////////////////////////////////////////////////////////
__host__ __device__
double OddRoundSumNonnegative_gpu(double th, double tl) {
    union {
        double d;
        long long int l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

__device__
int Normalize( long long int *accumulator, int *imin, int *imax) {
    long long int carry_in = accumulator[*imin] >> DIGITS;
    accumulator[*imin] -= carry_in << DIGITS;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        long long int carry_out = accumulator[i] >> DIGITS;    // Arithmetic shift
        accumulator[i] -= (carry_out << DIGITS);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax] += carry_in << DIGITS;

    return carry_in < 0;
}
__device__
int NormalizeT( long long int *accumulator, int *imin, int *imax) {
    long long int carry_in = accumulator[(*imin)*WARP_COUNT] >> DIGITS;
    accumulator[(*imin)*WARP_COUNT] -= carry_in << DIGITS;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i*WARP_COUNT] += carry_in;
        long long int carry_out = accumulator[i*WARP_COUNT] >> DIGITS;    // Arithmetic shift
        accumulator[i*WARP_COUNT] -= (carry_out << DIGITS);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[(*imax)*WARP_COUNT] += carry_in << DIGITS;

    return carry_in < 0;
}

__device__
double Round( long long int *accumulator) {
    int imin = 0;
    int imax = 38;
    int negative = Normalize(accumulator, &imin, &imax);

    //Find leading word
    int i;
    //Skip zeroes
    for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
    }
    if (negative) {
        //Skip ones
        for(; (accumulator[i] & ((1l << DIGITS) - 1)) == ((1l << DIGITS) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
        return 0.0;

    long long int hiword = negative ? ((1l << DIGITS) - 1) - accumulator[i] : accumulator[i];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - F_WORDS) * DIGITS);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long long int) rint(rounded);
    double mid = ldexp((double) hiword, (i - F_WORDS) * DIGITS);

    //Compute sticky
    long long int sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << DIGITS) - accumulator[j] : accumulator[j];

    long long int loword = negative ? (1l << DIGITS) - accumulator[i - 1] : accumulator[i - 1];
    loword |= !!sticky;
    double lo = ldexp((double) loword, (i - 1 - F_WORDS) * DIGITS);

    //Now add3(hi, mid, lo)
    //No overlap, we have already normalized
    if (mid != 0)
        lo = OddRoundSumNonnegative_gpu(mid, lo);

    //Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial superaccs
////////////////////////////////////////////////////////////////////////////////
__device__
void AccumulateWord( long long int *sa, int i, long long int x) {
    // With atomic superacc updates
    // accumulation and carry propagation can happen in any order,
    // as long long int as addition is atomic
    // only constraint is: never forget an overflow bit
    unsigned char overflow;
    long long int carry = x;
    long long int carrybit;
    long long int oldword = xadd(&sa[i * WARP_COUNT], x, &overflow);

    // To propagate over- or underflow
    while (overflow) {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // superacc[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> DIGITS;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1l << KRX : -1l << KRX);

        // Cancel carry-save bits
        xadd(&sa[i * WARP_COUNT], (long long int) -(carry << DIGITS), &overflow);
        //if (TSAFE && (s ^ overflow))
        if (0 && (s ^ overflow)) //MW: TSAFE is always 0
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i * WARP_COUNT], carry, &overflow);
    }
}

__device__
void Accumulate( long long int *sa, double x) {
    if (x == 0)
        return;

    int e;
    frexp(x, &e); //extract the exponent of x (lies in -1024;1023 ?)
    int exp_word = e / DIGITS;  // Word containing MSbit
    int iup = exp_word + F_WORDS; //can be at most 18 + 20 

    double xscaled = ldexp(x, -DIGITS * exp_word);

    int i;
    for (i = iup; xscaled != 0; --i) {
        double xrounded = rint(xscaled);
        long long int xint = (long long int) xrounded;

        AccumulateWord(sa, i, xint);

        xscaled -= xrounded;
        xscaled *= DELTASCALE;
    }
}


__global__ void ExDOT(
    long long int *d_PartialSuperaccs,
    const double *d_a,
    const double *d_b,
    const uint NbElements
) {
    __shared__ long long int l_sa[WARP_COUNT * BIN_COUNT]; //shared variables live for a thread block (39 rows, 16 columns!)
    long long int *l_workingBase = l_sa + (threadIdx.x & (WARP_COUNT - 1)); //the bitwise & with 15 is a modulo operation: threadIdx.x % 16
    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    __syncthreads();

    //Read data from global memory and scatter it to sub-superaccs
    double a[NBFPE] = {0.0};
    for(uint pos = blockIdx.x*blockDim.x+threadIdx.x; pos < NbElements; pos += gridDim.x*blockDim.x) {
            double x = d_a[pos]*d_b[pos];
            #pragma unroll
            for(uint i = 0; i != NBFPE; ++i) {
                double s;
                a[i] = KnuthTwoSum(a[i], x, &s);
                x = s;
            }
            if (x != 0.0) {
                Accumulate(l_workingBase, x);
                // Flush FPEs to superaccs
                #pragma unroll
                for(uint i = 0; i != NBFPE; ++i) {
                    Accumulate(l_workingBase, a[i]);
                    a[i] = 0.0;
                }
            }
        //double r = 0.0;
        //double x = TwoProductFMA(d_a[pos], d_b[pos], &r);

        //#pragma unroll
        //for(uint i = 0; i != NBFPE; ++i) {
        //    double s;
        //    a[i] = KnuthTwoSum(a[i], x, &s);
        //    x = s;
        //}
        //if (x != 0.0) {
        //    Accumulate(l_workingBase, x);
        //    // Flush FPEs to superaccs
        //    #pragma unroll
        //    for(uint i = 0; i != NBFPE; ++i) {
        //        Accumulate(l_workingBase, a[i]);
        //        a[i] = 0.0;
        //    }
        //}

        //if (r != 0.0) {//add the rest r in the same manner
        //    #pragma unroll
        //    for(uint i = 0; i != NBFPE; ++i) {
        //        double s;
        //        a[i] = KnuthTwoSum(a[i], r, &s);
        //        r = s;
        //    }
        //    if (r != 0.0) {
        //        Accumulate(l_workingBase, r);
        //        // Flush FPEs to superaccs
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            Accumulate(l_workingBase, a[i]);
        //            a[i] = 0.0;
        //        }
        //    }
        //}
    }
	//Flush FPEs to superaccs
    #pragma unroll
    for(uint i = 0; i != NBFPE; ++i)
        Accumulate(l_workingBase, a[i]);
    __syncthreads();

    //Merge sub-superaccs into work-group partial-accumulator ( ATTENTION: PartialSuperacc is transposed!)
    uint pos = threadIdx.x;
//if(pos < WARP_COUNT) {
//        int imin = 0;
//        int imax = 38;
//    NormalizeT( l_workingBase, &imin, &imax);
//}
//    __syncthreads();

    if (pos < BIN_COUNT) {
        long long int sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];

        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + pos] = sum;
    }

    __syncthreads();
    if (pos == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT], &imin, &imax);
    }
}

__global__ void ExDOT(
    long long int *d_PartialSuperaccs,
    const double *d_a,
    const double *d_b,
    const double *d_c,
    const uint NbElements
) {
    __shared__ long long int l_sa[WARP_COUNT * BIN_COUNT]; //shared variables live for a thread block (39 rows, 16 columns!)
    long long int *l_workingBase = l_sa + (threadIdx.x & (WARP_COUNT - 1)); //the bitwise & with 15 is a modulo operation: threadIdx.x % 16
    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    __syncthreads();

    //Read data from global memory and scatter it to sub-superaccs
    double a[NBFPE] = {0.0};
    for(uint pos = blockIdx.x*blockDim.x+threadIdx.x; pos < NbElements; pos += gridDim.x*blockDim.x) {
            double x = d_a[pos]*d_c[pos]*d_b[pos];
            #pragma unroll
            for(uint i = 0; i != NBFPE; ++i) {
                double s;
                a[i] = KnuthTwoSum(a[i], x, &s);
                x = s;
            }
            if (x != 0.0) {
                Accumulate(l_workingBase, x);
                // Flush FPEs to superaccs
                #pragma unroll
                for(uint i = 0; i != NBFPE; ++i) {
                    Accumulate(l_workingBase, a[i]);
                    a[i] = 0.0;
                }
            }
        //double r  = 0.0, r2 = 0.0;
        //double x  = TwoProductFMA(d_a[pos], d_b[pos], &r);
        //double x2 = TwoProductFMA(x , d_c[pos], &r2);


        //if( x2 != 0.0) {//accumulate x2
        //    #pragma unroll
        //    for(uint i = 0; i != NBFPE; ++i) {
        //        double s;
        //        a[i] = KnuthTwoSum(a[i], x2, &s);
        //        x2 = s;
        //    }
        //    if (x2 != 0.0) {
        //        Accumulate(l_workingBase, x2);
        //        // Flush FPEs to superaccs
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            Accumulate(l_workingBase, a[i]);
        //            a[i] = 0.0;
        //        }
        //    }
        //}
        //if (r2 != 0.0) {//add the rest r2 
        //    #pragma unroll
        //    for(uint i = 0; i != NBFPE; ++i) {
        //        double s;
        //        a[i] = KnuthTwoSum(a[i], r2, &s);
        //        r2 = s; //error was here r = s
        //    }
        //    if (r2 != 0.0) { //error was here r != 0.0
        //        Accumulate(l_workingBase, r2);
        //        // Flush FPEs to superaccs
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            Accumulate(l_workingBase, a[i]);
        //            a[i] = 0.0;
        //        }
        //    }
        //}

        //if (r != 0.0) {//add the rest r*c in the same manner
        //    x2 = TwoProductFMA(r , d_c[pos], &r2);
        //    if( x2 != 0.0) {//accumulate x2
        //        #pragma unroll
        //        for(uint i = 0; i != NBFPE; ++i) {
        //            double s;
        //            a[i] = KnuthTwoSum(a[i], x2, &s);
        //            x2 = s;
        //        }
        //        if (x2 != 0.0) {
        //            Accumulate(l_workingBase, x2);
        //            // Flush FPEs to superaccs
        //            #pragma unroll
        //            for(uint i = 0; i != NBFPE; ++i) {
        //                Accumulate(l_workingBase, a[i]);
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
        //            Accumulate(l_workingBase, r2);
        //            // Flush FPEs to superaccs
        //            #pragma unroll
        //            for(uint i = 0; i != NBFPE; ++i) {
        //                Accumulate(l_workingBase, a[i]);
        //                a[i] = 0.0;
        //            }
        //        }
        //    }
        //}
    }
	//Flush FPEs to superaccs
    #pragma unroll
    for(uint i = 0; i != NBFPE; ++i)
        Accumulate(l_workingBase, a[i]);
    __syncthreads();

    //Merge sub-superaccs into work-group partial-accumulator ( ATTENTION: PartialSuperacc is transposed!)
    uint pos = threadIdx.x;
    if (pos < BIN_COUNT) {
        long long int sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];

        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + pos] = sum;
    }

    __syncthreads();
    if (pos == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT], &imin, &imax);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////
__global__
void ExDOTComplete(
     double *d_Res,
     long long int *d_PartialSuperaccs
) {
    uint lid = threadIdx.x;
    uint gid = blockIdx.x;

    if (lid < BIN_COUNT) {
        long long int sum = 0;

        for(uint i = 0; i < MERGE_SUPERACCS_SIZE; i++)
            sum += d_PartialSuperaccs[(gid * MERGE_SUPERACCS_SIZE + i) * BIN_COUNT + lid];

        d_PartialSuperaccs[gid * BIN_COUNT + lid] = sum;
    }

    __syncthreads();
    if (lid == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[gid * BIN_COUNT], &imin, &imax);
    }

    __syncthreads();
    if ((lid < BIN_COUNT) && (gid == 0)) {
        long long int sum = 0;

        for(uint i = 0; i < gridDim.x; i++)
            sum += d_PartialSuperaccs[i * BIN_COUNT + lid];

        d_PartialSuperaccs[lid] = sum;

        //__syncthreads();
        //if (lid == 0)
        //    d_Res[0] = Round(d_PartialSuperaccs);
    }
}

__host__
std::vector<int64_t> exdot_gpu(unsigned size, const double* x1_ptr, const double* x2_ptr)
{
    thrust::device_vector<long long int> d_PartialSuperaccsV( PARTIAL_SUPERACCS_COUNT*BIN_COUNT); //39 columns and PSC rows
    long long int *d_PartialSuperaccs = thrust::raw_pointer_cast( d_PartialSuperaccsV.data());
    ExDOT<<<PARTIAL_SUPERACCS_COUNT, WORKGROUP_SIZE>>>( d_PartialSuperaccs, x1_ptr, x2_ptr,size);
    thrust::device_vector<double> r(1,0);
    double *r_ptr = thrust::raw_pointer_cast( r.data());
    ExDOTComplete<<<PARTIAL_SUPERACCS_COUNT/MERGE_SUPERACCS_SIZE, MERGE_WORKGROUP_SIZE>>>( r_ptr, d_PartialSuperaccs );
    std::vector<int64_t> h_Superacc(BIN_COUNT);
    cudaMemcpy( &h_Superacc[0], d_PartialSuperaccs, BIN_COUNT*sizeof(long long int), cudaMemcpyDeviceToHost);
    return h_Superacc;
}
__host__
std::vector<int64_t> exdot_gpu(unsigned size, const double* x1_ptr, const double* x2_ptr, const double* x3_ptr)
{
    thrust::device_vector<long long int> d_PartialSuperaccsV( PARTIAL_SUPERACCS_COUNT*BIN_COUNT); //39 columns and PSC rows
    long long int *d_PartialSuperaccs = thrust::raw_pointer_cast( d_PartialSuperaccsV.data());
    ExDOT<<<PARTIAL_SUPERACCS_COUNT, WORKGROUP_SIZE>>>( d_PartialSuperaccs, x1_ptr, x2_ptr, x3_ptr,size);
    thrust::device_vector<double> r(1,0);
    double *r_ptr = thrust::raw_pointer_cast( r.data());
    ExDOTComplete<<<PARTIAL_SUPERACCS_COUNT/MERGE_SUPERACCS_SIZE, MERGE_WORKGROUP_SIZE>>>( r_ptr, d_PartialSuperaccs );

    std::vector<int64_t> h_Superacc(BIN_COUNT, 1);
    cudaMemcpy( &h_Superacc[0], d_PartialSuperaccs, BIN_COUNT*sizeof(long long int), cudaMemcpyDeviceToHost);
    return h_Superacc;
}
}//namespace exblas

