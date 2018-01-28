/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */

/**
 *  @file exdot_omp.h
 *  @brief OpenMP version of exdot
 *
 *  @authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk 
 */
#pragma once
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "accumulate.h"
#include "ExSUM.FPE.hpp"
#include <omp.h>

namespace exblas{
///@cond
namespace cpu{



/**
 * \brief Parallel reduction step
 *
 * \param step step among threads
 * \param tid1 id of the first thread
 * \param tid2 id of the second thread
 * \param acc1 superaccumulator of the first thread
 * \param acc2 superaccumulator of the second thread
 */
inline static void ReductionStep(int step, int tid1, int tid2, int64_t * acc1, int64_t * acc2,
    int volatile * ready1, int volatile * ready2)
{
    _mm_prefetch((char const*)ready2, _MM_HINT_T0);
    // Wait for thread 2
    while(*ready2 < step) {
        // wait
        _mm_pause();
    }
    int imin = IMIN, imax = IMAX;
    Normalize( acc1, imin, imax);
    imin = IMIN, imax = IMAX;
    Normalize( acc2, imin, imax);
    for(int i = IMIN; i <= IMAX; ++i) {
        acc1[i] += acc2[i];
    }
}

/**
 * \brief Final step of summation -- Parallel reduction among threads
 *
 * \param tid thread ID
 * \param tnum number of threads
 * \param acc superaccumulator
 */
inline static void Reduction(unsigned int tid, unsigned int tnum, std::vector<int32_t>& ready,
    std::vector<int64_t>& acc, int const linesize)
{
    // Custom reduction
    for(unsigned int s = 1; (unsigned)(1 << (s-1)) < tnum; ++s) 
    {
        int32_t volatile * c = &ready[tid * linesize];
        ++*c;
        if(tid % (1 << s) == 0) {
            unsigned int tid2 = tid | (1 << (s-1));
            if(tid2 < tnum) {
                //acc[tid2].Prefetch(); // No effect...
                ReductionStep(s, tid, tid2, &acc[tid*BIN_COUNT], &acc[tid2*BIN_COUNT],
                    &ready[tid * linesize], &ready[tid2 * linesize]);
            }
        }
    }
}

template<typename CACHE> 
void ExDOTFPE(int N, const double *a, const double *b, int64_t* h_superacc) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    std::vector<int64_t> acc(maxthreads*BIN_COUNT);
    std::vector<int32_t> ready(maxthreads * linesize);

    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int tnum = omp_get_num_threads();

        CACHE cache(&acc[tid*BIN_COUNT]);
        *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?

        int l = ((tid * int64_t(N)) / tnum) & ~7ul; // & ~3ul == round down to multiple of 4
        int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

        for(int i = l; i < r; i+=8) {
            asm ("# myloop");
            vcl::Vec8d r1 ;
            vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
            //vcl::Vec8d x  = vcl::mul_add( vcl::Vec8d().load(a+i),vcl::Vec8d().load(b+i),0);
            cache.Accumulate(x);
            cache.Accumulate(r1); //MW: exact product but halfs the speed
        }
        if( tid+1==tnum && r != N-1) {
            r+=1;
            //accumulate remainder
            vcl::Vec8d r1;
            vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
            //vcl::Vec8d x  = vcl::mul_add( vcl::Vec8d().load_partial(N-r,a+r),vcl::Vec8d().load_partial(N-r,b+r),0);
            cache.Accumulate(x);
            cache.Accumulate(r1);
        }
        cache.Flush();
        int imin=IMIN, imax=IMAX;
        Normalize(&acc[tid*BIN_COUNT], imin, imax);

        Reduction(tid, tnum, ready, acc, linesize);
    }
    for( int i=IMIN; i<=IMAX; i++)
        h_superacc[i] = acc[i];
}

template<typename CACHE> 
void ExDOTFPE(int N, const double *a, const double *b, const double *c, int64_t* h_superacc) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    std::vector<int64_t> acc(maxthreads*BIN_COUNT);
    std::vector<int32_t> ready(maxthreads * linesize);

    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int tnum = omp_get_num_threads();

        CACHE cache(&acc[tid*BIN_COUNT]);
        *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?

        int l = ((tid * int64_t(N)) / tnum) & ~7ul;// & ~3ul == round down to multiple of 4
        int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

        for(int i = l; i < r; i+=8) {
            asm ("# myloop");
            //vcl::Vec8d r1 , r2, cvec = vcl::Vec8d().load(c+i);
            //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
            //vcl::Vec8d x2 = TwoProductFMA(x , cvec, r2);
            vcl::Vec8d x1  = vcl::mul_add(vcl::Vec8d().load(a+i),vcl::Vec8d().load(b+i), 0);
            vcl::Vec8d x2  = vcl::mul_add( x1                   ,vcl::Vec8d().load(c+i), 0);
            cache.Accumulate(x2);
            //cache.Accumulate(r2);
            //x2 = TwoProductFMA(r1, cvec, r2);
            //cache.Accumulate(x2);
            //cache.Accumulate(r2);
        }
        if( tid+1 == tnum && r != N-1) {
            r+=1;
            //accumulate remainder
            //vcl::Vec8d r1 , r2, cvec = vcl::Vec8d().load_partial(N-r, c+r);
            //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
            //vcl::Vec8d x2 = TwoProductFMA(x , cvec, r2);
            vcl::Vec8d x1  = vcl::mul_add(vcl::Vec8d().load_partial(N-r, a+r),vcl::Vec8d().load_partial(N-r,b+r), 0);
            vcl::Vec8d x2  = vcl::mul_add( x1                   ,vcl::Vec8d().load_partial(N-r,c+r), 0);
            cache.Accumulate(x2);
            //cache.Accumulate(r2);
            //x2 = TwoProductFMA(r1, cvec, r2);
            //cache.Accumulate(x2);
            //cache.Accumulate(r2);
        }
        cache.Flush();
        int imin=IMIN, imax=IMAX;
        Normalize(&acc[tid*BIN_COUNT], imin, imax);

        Reduction(tid, tnum, ready, acc, linesize);
    }
    for( int i=IMIN; i<=IMAX; i++)
        h_superacc[i] = acc[i];
}
}//namespace cpu
///@endcond

/*!@brief OpenMP parallel version of exact dot product
 *
 * Computes the exact sum \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param h_superacc pointer to an array of 64 bit integers (the superaccumulator) in host memory with size at least \c exblas::BIN_COUNT (39) (contents are overwritten)
 * @sa \c exblas::cpu::Round  to convert the superaccumulator into a double precision number
*/
void exdot_omp(unsigned size, const double* x1_ptr, const double* x2_ptr, int64_t* h_superacc){
    assert( vcl::instrset_detect() >= 7);
    //assert( vcl::hasFMA3() );
    cpu::ExDOTFPE<cpu::FPExpansionVect<vcl::Vec8d, 8, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc);
}
/*!@brief OpenMP parallel version of exact triple dot product
 *
 * Computes the exact sum \f[ \sum_{i=0}^{N-1} x_i w_i y_i \f]
 * @ingroup highlevel
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param x3_ptr third array
 * @param h_superacc pointer to an array of 64 bit integegers (the superaccumulator) in host memory with size at least \c exblas::BIN_COUNT (39) (contents are overwritten)
 * @sa \c exblas::cpu::Round  to convert the superaccumulator into a double precision number
 */
void exdot_omp(unsigned size, const double *x1_ptr, const double* x2_ptr, const double * x3_ptr, int64_t* h_superacc) {
    assert( vcl::instrset_detect() >= 7);
    //assert( vcl::hasFMA3() );
    cpu::ExDOTFPE<cpu::FPExpansionVect<vcl::Vec8d, 8, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc);
}

}//namespace exblas
