/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

namespace exblas{

static constexpr uint BIN_COUNT      =  39;
static constexpr uint KRX            =  8;                 // High-radix carry-save bits
static constexpr uint DIGITS         =  56;
static constexpr double DELTASCALE   =  72057594037927936.0;  // Assumes K>0
static constexpr uint F_WORDS        =  20;
static constexpr uint TSAFE          =  0;

//Kernel paramters for EXDOT
static constexpr uint WARP_COUNT               = 16 ;
static constexpr uint WARP_SIZE                = 16 ;
static constexpr uint WORKGROUP_SIZE           = (WARP_COUNT * WARP_SIZE); //# threads per group
static constexpr uint PARTIAL_SUPERACCS_COUNT  = 512; //# of groups; each has a partial SuperAcc
//Kernel paramters for EXDOTComplete
static constexpr uint MERGE_SUPERACCS_SIZE     = 128;
static constexpr uint PARTIAL_WORKGORUP_SIZE   = 39;
static constexpr uint MERGE_WORKGROUP_SIZE     = 64;
static constexpr uint MERGE_SUPERACCS_SIZE     = 128;


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ 
double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

__host__ __device__ 
double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}
// signedcarry in {-1, 0, 1}
__host__ __device__ 
long xadd(__shared__ volatile long *sa, long x, uchar *of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    //long y = atom_add(sa, x);
    long y = atomicAdd(sa, x); //not sure if this is the correct CUDA function
    long z = y + x; // since the value sa->superacc[i] can be changed by another work item

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
double OddRoundSumNonnegative(double th, double tl) {
    union {
        double d;
        long l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

__host__ __device__
int Normalize(__device__ long *accumulator, int *imin, int *imax) {
    long carry_in = accumulator[*imin] >> DIGITS;
    accumulator[*imin] -= carry_in << DIGITS;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        long carry_out = accumulator[i] >> DIGITS;    // Arithmetic shift
        accumulator[i] -= (carry_out << DIGITS);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax] += carry_in << DIGITS;

    return carry_in < 0;
}

__host__ __device__
double Round(__device__ long *accumulator) {
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

    long hiword = negative ? ((1l << DIGITS) - 1) - accumulator[i] : accumulator[i];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - F_WORDS) * DIGITS);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long) rint(rounded);
    double mid = ldexp((double) hiword, (i - F_WORDS) * DIGITS);

    //Compute sticky
    long sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << DIGITS) - accumulator[j] : accumulator[j];

    long loword = negative ? (1l << DIGITS) - accumulator[i - 1] : accumulator[i - 1];
    loword |= !!sticky;
    double lo = ldexp((double) loword, (i - 1 - F_WORDS) * DIGITS);

    //Now add3(hi, mid, lo)
    //No overlap, we have already normalized
    if (mid != 0)
        lo = OddRoundSumNonnegative(mid, lo);

    //Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial superaccs
////////////////////////////////////////////////////////////////////////////////
__host__ __device__
void AccumulateWord(__shared__ volatile long *sa, int i, long x) {
    // With atomic superacc updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    uchar overflow;
    long carry = x;
    long carrybit;
    long oldword = xadd(&sa[i * WARP_COUNT], x, &overflow);

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
        xadd(&sa[i * WARP_COUNT], (long) -(carry << DIGITS), &overflow);
        if (TSAFE && (s ^ overflow))
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i * WARP_COUNT], carry, &overflow);
    }
}

__host__ __device__
void Accumulate(__shared__ volatile long *sa, double x) {
    if (x == 0)
        return;

    int e;
    frexp(x, &e);
    int exp_word = e / DIGITS;  // Word containing MSbit
    int iup = exp_word + F_WORDS;

    double xscaled = ldexp(x, -DIGITS * exp_word);

    int i;
    for (i = iup; xscaled != 0; --i) {
        double xrounded = rint(xscaled);
        long xint = (long) xrounded;

        AccumulateWord(sa, i, xint);

        xscaled -= xrounded;
        xscaled *= DELTASCALE;
    }
}


template<size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) //cuda performance hint macro, (max_threads_per_block, minBlocksPerMultiprocessor)
__global__ //__attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void ExDOT(
    __device__ long *d_PartialSuperaccs,
    __device__ double *d_a,
    const uint inca,
    const uint offseta,
    __device__ double *d_b,
    const uint incb,
    const uint offsetb,
    const uint NbElements
) {
    __shared__ long l_sa[WARP_COUNT * BIN_COUNT] __attribute__((aligned(8)));
    __shared__ long *l_workingBase = l_sa + (threadIdx.x & (WARP_COUNT - 1));
    __shared__ bool l_sa_check[WARP_COUNT];
    __shared__ bool *l_workingBase_check = l_sa_check + (threadIdx.x & (WARP_COUNT - 1));

    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    *l_workingBase_check = false;
    __synchthreads();

    //Read data from global memory and scatter it to sub-superaccs
    double a[NBFPE] = {0.0};
	if ((offseta == 0) && (inca == 1) && (offsetb == 0) && (incb == 1)) {
		for(uint pos = blockIdx.x*blockDim.x+threadIdx.x; pos < NbElements; pos += gridDim.x*blockDim.x) {
			double r = 0.0;
			double x = TwoProductFMA(d_a[pos], d_b[pos], &r);

			#ifdef NVIDIA
				#pragma unroll
			#endif
			for(uint i = 0; i != NBFPE; ++i) {
				double s;
				a[i] = KnuthTwoSum(a[i], x, &s);
				x = s;
			}
			if (x != 0.0) {
				Accumulate(l_workingBase, x);
				// Flush FPEs to superaccs
				#ifdef NVIDIA
					#pragma unroll
				#endif
				for(uint i = 0; i != NBFPE; ++i) {
					Accumulate(l_workingBase, a[i]);
					a[i] = 0.0;
				}
			}

			if (r != 0.0) {
				#ifdef NVIDIA
					#pragma unroll
				#endif
				for(uint i = NBFPE-3; i != NBFPE; ++i) {
					double s;
					a[i] = KnuthTwoSum(a[i], r, &s);
					r = s;
				}
				if (r != 0.0) {
					Accumulate(l_workingBase, r);
					// Flush FPEs to superaccs
					#ifdef NVIDIA
						#pragma unroll
					#endif
					for(uint i = 0; i != NBFPE; ++i) {
						Accumulate(l_workingBase, a[i]);
						a[i] = 0.0;
					}
				}
			}
		}
	} else {
		for(uint pos = blockIdx.x*blockDim.x+threadIdx.x; pos < NbElements; pos += gridDim.x*blockDim.x) {
			double r = 0.0;
			double x = TwoProductFMA(d_a[offseta + pos * inca], d_b[offsetb + pos * incb], &r);

			#ifdef NVIDIA
				#pragma unroll
			#endif
			for(uint i = 0; i != NBFPE; ++i) {
				double s;
				a[i] = KnuthTwoSum(a[i], x, &s);
				x = s;
			}
			if (x != 0.0) {
				Accumulate(l_workingBase, x);
				// Flush FPEs to superaccs
				#ifdef NVIDIA
					#pragma unroll
				#endif
				for(uint i = 0; i != NBFPE; ++i) {
					Accumulate(l_workingBase, a[i]);
					a[i] = 0.0;
				}
			}

			if (r != 0.0) {
				#ifdef NVIDIA
					#pragma unroll
				#endif
				for(uint i = NBFPE-3; i != NBFPE; ++i) {
					double s;
					a[i] = KnuthTwoSum(a[i], r, &s);
					r = s;
				}
				if (r != 0.0) {
					Accumulate(l_workingBase, r);
					// Flush FPEs to superaccs
					#ifdef NVIDIA
						#pragma unroll
					#endif
					for(uint i = 0; i != NBFPE; ++i) {
						Accumulate(l_workingBase, a[i]);
						a[i] = 0.0;
					}
				}
			}
		}
	}
	//Flush FPEs to superaccs
    #ifdef NVIDIA
        #pragma unroll
    #endif
    for(uint i = 0; i != NBFPE; ++i)
        Accumulate(l_workingBase, a[i]);
    __synchthreads();

    //Merge sub-superaccs into work-group partial-accumulator
    uint pos = threadIdx.x;
    if (pos < BIN_COUNT) {
        long sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];

        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + pos] = sum;
    }

    __synchthreads();
    if (pos == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT], &imin, &imax);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////
template<size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) //cuda performance hint macro, (max_threads_per_block, minBlocksPerMultiprocessor)
__global__
void ExDOTComplete(
    __device__ double *d_Res,
    __device__ long *d_PartialSuperaccs
) {
    uint lid = threadIdx.x;
    uint gid = blockIdx.x;

    if (lid < BIN_COUNT) {
        long sum = 0;

        for(uint i = 0; i < MERGE_SUPERACCS_SIZE; i++)
            sum += d_PartialSuperaccs[(gid * MERGE_SUPERACCS_SIZE + i) * BIN_COUNT + lid];

        d_PartialSuperaccs[gid * BIN_COUNT + lid] = sum;
    }

    __synchthreads();
    if (lid == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[gid * BIN_COUNT], &imin, &imax);
    }

    __synchthreads();
    if ((lid < BIN_COUNT) && (gid == 0)) {
        long sum = 0;

        for(uint i = 0; i < gridDim.x*blockDim.x / blockDim.x; i++)
            sum += d_PartialSuperaccs[i * BIN_COUNT + lid];

        d_PartialSuperaccs[lid] = sum;

        __synchthreads();
        if (lid == 0)
            d_Res[0] = Round(d_PartialSuperaccs);
    }
}

double exdot(const thrust::device_vector<double>& x1, const thrust::device_vector<double>& x2)
{
    static thrust::device_vector<long> d_PartialSuperaccsV( PARTIAL_SUPERACCS_COUNT*BIN_COUNT);
    static long *d_PartialSuperaccs = thrust::raw_pointer_cast( d_PartialSuperaccsV.data());

    const double *x1_ptr = thrust::raw_pointer_cast( x1.data());
    EXDOT<<<PARTIAL_SUPERACCS_COUNT, WORKGROUP_SIZE>>>( d_PartialSuperAccs, x1_ptr, 1,0, x2_ptr,1,0,x1.size());
    double result;
    ExDOTComplete<<<PARTIAL_SUPERACCS_COUNT/MERGE_SUPERACCS_SIZE, MERGE_WORKGROUP_SIZE>>>( &result, d_PartialSuperAccs );
    return result;
}
}//namespace exblas

