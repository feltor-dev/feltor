/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "superaccumulator.hpp"
#include "ExSUM.FPE.hpp"
#include <omp.h>

namespace exblas{



/**
 * \brief Parallel reduction step
 *
 * \param step step among threads
 * \param tid1 id of the first thread
 * \param tid2 id of the second thread
 * \param acc1 superaccumulator of the first thread
 * \param acc2 superaccumulator of the second thread
 */
inline static void ReductionStep(int step, int tid1, int tid2, Superaccumulator * acc1, Superaccumulator * acc2,
    int volatile * ready1, int volatile * ready2)
{
    _mm_prefetch((char const*)ready2, _MM_HINT_T0);
    // Wait for thread 2
    while(*ready2 < step) {
        // wait
        _mm_pause();
    }
    acc1->Accumulate(*acc2);
}

/**
 * \brief Final step of summation -- Parallel reduction among threads
 *
 * \param tid thread ID
 * \param tnum number of threads
 * \param acc superaccumulator
 */
inline static void Reduction(unsigned int tid, unsigned int tnum, std::vector<int32_t>& ready,
    std::vector<Superaccumulator>& acc, int const linesize)
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
                ReductionStep(s, tid, tid2, &acc[tid], &acc[tid2],
                    &ready[tid * linesize], &ready[tid2 * linesize]);
            }
        }
    }
}

template<typename CACHE> 
Superaccumulator ExDOTFPE(int N, const double *a, const double *b) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    std::vector<Superaccumulator> acc(maxthreads);
    std::vector<int32_t> ready(maxthreads * linesize);

    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int tnum = omp_get_num_threads();

        CACHE cache(acc[tid]);
        *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?

        int l = ((tid * int64_t(N)) / tnum) & ~7ul; // & ~3ul == round down to multiple of 4
        int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

        for(int i = l; i < r; i+=8) {
            asm ("# myloop");
            vcl::Vec8d r1 ;
            vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
            cache.Accumulate(x);
            //cache.Accumulate(r1);
        }
        if( tid+1==tnum && r != N-1) {
            r+=1;
            //accumulate remainder
            vcl::Vec8d r1;
            vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
            cache.Accumulate(x);
            //cache.Accumulate(r1);
        }
        cache.Flush();
        acc[tid].Normalize();

        Reduction(tid, tnum, ready, acc, linesize);
    }
    return acc[0];
}
template<typename CACHE> 
Superaccumulator ExDOTFPE(int N, const double *a, const double *b, const double *c) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    std::vector<Superaccumulator> acc(maxthreads);
    std::vector<int32_t> ready(maxthreads * linesize);

    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int tnum = omp_get_num_threads();

        CACHE cache(acc[tid]);
        *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?

        int l = ((tid * int64_t(N)) / tnum) & ~7ul;// & ~3ul == round down to multiple of 4
        int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

        for(int i = l; i < r; i+=8) {
            asm ("# myloop");
            vcl::Vec8d r1 , r2, cvec = vcl::Vec8d().load(c+i);
            vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
            vcl::Vec8d x2 = TwoProductFMA(x , cvec, r2);
            cache.Accumulate(x2);
            //cache.Accumulate(r2);
            //x2 = TwoProductFMA(r1, cvec, r2);
            //cache.Accumulate(x2);
            //cache.Accumulate(r2);
        }
        if( tid+1 == tnum && r != N-1) {
            r+=1;
            //accumulate remainder
            vcl::Vec8d r1 , r2, cvec = vcl::Vec8d().load_partial(N-r, c+r);
            vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
            vcl::Vec8d x2 = TwoProductFMA(x , cvec, r2);
            cache.Accumulate(x2);
            //cache.Accumulate(r2);
            //x2 = TwoProductFMA(r1, cvec, r2);
            //cache.Accumulate(x2);
            //cache.Accumulate(r2);
        }
        cache.Flush();
        acc[tid].Normalize();

        Reduction(tid, tnum, ready, acc, linesize);
    }
    return acc[0];
}
/*
 * Parallel summation using our algorithm
 * Otherwise, use floating-point expansions of size FPE with superaccumulators when needed
 * early_exit corresponds to the early-exit technique
 */
Superaccumulator exdot_omp(int N, const double *a, const double* b, int fpe, bool early_exit) {
    assert( vcl::instrset_detect() >= 7);
    assert( vcl::hasFMA3() );
    if (fpe < 2) {
        fprintf(stderr, "Size of floating-point expansion must be in the interval [2, 8]\n");
        exit(1);
    }
    Superaccumulator acc;
    if (early_exit) {
        if (fpe <= 4)
            acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 4, FPExpansionTraits<true> > >)(N,a,b);
        if (fpe <= 6)
            acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 6, FPExpansionTraits<true> > >)(N,a,b);
        if (fpe <= 8)
            acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 8, FPExpansionTraits<true> > >)(N,a,b);
    } else { // ! early_exit
        if (fpe == 2) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 2> >)(N, a,b);
        if (fpe == 3) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 3> >)(N, a,b);
        if (fpe == 4) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 4> >)(N, a,b);
        if (fpe == 5) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 5> >)(N, a,b);
        if (fpe == 6) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 6> >)(N, a,b);
        if (fpe == 7) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 7> >)(N, a,b);
        if (fpe == 8) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 8> >)(N, a,b);
    }
    return acc;
}
Superaccumulator exdot_omp(int N, const double *a, const double* b, const double * c, int fpe, bool early_exit) {
    assert( vcl::instrset_detect() >= 7);
    assert( vcl::hasFMA3() );
    if (fpe < 2) {
        fprintf(stderr, "Size of floating-point expansion must be in the interval [2, 8]\n");
        exit(1);
    }
    Superaccumulator acc;
    if (early_exit) {
        if (fpe <= 4)
            acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 4, FPExpansionTraits<true> > >)(N,a,b,c);
        if (fpe <= 6)
            acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 6, FPExpansionTraits<true> > >)(N,a,b,c);
        if (fpe <= 8)
            acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 8, FPExpansionTraits<true> > >)(N,a,b,c);
    } else { // ! early_exit
        if (fpe == 2) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 2> >)(N, a,b,c);
        if (fpe == 3) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 3> >)(N, a,b,c);
        if (fpe == 4) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 4> >)(N, a,b,c);
        if (fpe == 5) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 5> >)(N, a,b,c);
        if (fpe == 6) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 6> >)(N, a,b,c);
        if (fpe == 7) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 7> >)(N, a,b,c);
        if (fpe == 8) 
	    acc = (ExDOTFPE<FPExpansionVect<vcl::Vec8d, 8> >)(N, a,b,c);
    }
    return acc;
}

}//namespace exblas
