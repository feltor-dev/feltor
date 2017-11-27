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

#ifdef EXBLAS_MPI
    #include <mpi.h>
#endif

#ifdef EXBLAS_TIMING
    #define iterations 50
#endif


/*
 * Parallel summation using our algorithm
 * If fpe < 2, use superaccumulators only,
 * Otherwise, use floating-point expansions of size FPE with superaccumulators when needed
 * early_exit corresponds to the early-exit technique
 */
double exdot(int N, double *a, double* b, int fpe, bool early_exit) {
    if (fpe < 2) {
        fprintf(stderr, "Size of floating-point expansion must be in the interval [2, 8]\n");
        exit(1);
    }
    if (early_exit) {
        if (fpe <= 4)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 4, FPExpansionTraits<true> > >)(N,a,b);
        if (fpe <= 6)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 6, FPExpansionTraits<true> > >)(N,a,b);
        if (fpe <= 8)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 8, FPExpansionTraits<true> > >)(N,a,b);
    } else { // ! early_exit
        if (fpe == 2) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 2> >)(N, a,b);
        if (fpe == 3) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 3> >)(N, a,b);
        if (fpe == 4) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 4> >)(N, a,b);
        if (fpe == 5) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 5> >)(N, a,b);
        if (fpe == 6) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 6> >)(N, a,b);
        if (fpe == 7) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 7> >)(N, a,b);
        if (fpe == 8) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 8> >)(N, a,b);
    }

    return 0.0;
}
double exdot(int N, double *a, double* b, double * c, int fpe, bool early_exit) {
    if (fpe < 2) {
        fprintf(stderr, "Size of floating-point expansion must be in the interval [2, 8]\n");
        exit(1);
    }
    if (early_exit) {
        if (fpe <= 4)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 4, FPExpansionTraits<true> > >)(N,a,b,c);
        if (fpe <= 6)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 6, FPExpansionTraits<true> > >)(N,a,b,c);
        if (fpe <= 8)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 8, FPExpansionTraits<true> > >)(N,a,b,c);
    } else { // ! early_exit
        if (fpe == 2) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 2> >)(N, a,b,c);
        if (fpe == 3) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 3> >)(N, a,b,c);
        if (fpe == 4) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 4> >)(N, a,b,c);
        if (fpe == 5) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 5> >)(N, a,b,c);
        if (fpe == 6) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 6> >)(N, a,b,c);
        if (fpe == 7) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 7> >)(N, a,b,c);
        if (fpe == 8) 
	    return (ExSUMFPE<FPExpansionVect<Vec4d, 8> >)(N, a,b,c);
    }

    return 0.0;
}


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
    for(unsigned int s = 1; (1 << (s-1)) < tnum; ++s) 
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
double ExDOTFPE(int N, double *a, double *b) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    double dacc;
#ifdef EXBLAS_TIMING
    double t, mint = 10000;
    uint64_t tstart, tend;
    for(int iter = 0; iter != iterations; ++iter) {
        tstart = rdtsc();
#endif
        std::vector<Superaccumulator> acc(maxthreads);
        std::vector<int32_t> ready(maxthreads * linesize);
    
        #pragma omp parallel
        {
            unsigned int tid = omp_get_thread_num();
            unsigned int tnum = omp_get_num_threads();

            CACHE cache(acc[tid]);
            *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?

            int l = ((tid * int64_t(N)) / tnum) & ~7ul;
            int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

            for(int i = l; i < r; i+=4) {
                asm ("# myloop");
                Vec4d r1 ;
                Vec4d x  = TwoProductFMA(Vec4d().load(a+i), Vec4d().load(b+i), r1);
                cache.Accumulate(x);
                cache.Accumulate(r1);
            }
            cache.Flush();
            acc[tid].Normalize();

            Reduction(tid, tnum, ready, acc, linesize);
        }
#ifdef EXBLAS_MPI
        acc[0].Normalize();
        std::vector<int64_t> result(acc[0].get_f_words() + acc[0].get_e_words(), 0);
        MPI_Allreduce(&(acc[0].get_accumulator()[0]), &(result[0]), acc[0].get_f_words() + acc[0].get_e_words(), MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        Superaccumulator acc_fin(result);
        dacc = acc_fin.Round();
#else
        dacc = acc[0].Round();
#endif    

#ifdef EXBLAS_TIMING
        tend = rdtsc();
        t = double(tend - tstart) / N;
        mint = std::min(mint, t);
    }
    fprintf(stderr, "%f ", mint);
#endif

    return dacc;
}
template<typename CACHE> double ExDOTFPE(int N, double *a, double *b, double *c) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    double dacc;
#ifdef EXBLAS_TIMING
    double t, mint = 10000;
    uint64_t tstart, tend;
    for(int iter = 0; iter != iterations; ++iter) {
        tstart = rdtsc();
#endif
        std::vector<Superaccumulator> acc(maxthreads);
        std::vector<int32_t> ready(maxthreads * linesize);
    
        #pragma omp parallel
        {
            unsigned int tid = omp_get_thread_num();
            unsigned int tnum = omp_get_num_threads();

            CACHE cache(acc[tid]);
            *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?

            int l = ((tid * int64_t(N)) / tnum) & ~7ul;
            int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;

            for(int i = l; i < r; i+=4) {
                asm ("# myloop");
                Vec4d r1 , r2, cvec = Vec4d().load(c+i);
                Vec4d x  = TwoProductFMA(Vec4d().load(a+i), Vec4d().load(b+i), r1);
                Vec4d x2 = TwoProductFMA(x , cvec, r2);
                cache.Accumulate(x2);
                cache.Accumulate(r2);
                x2 = TwoProductFMA(r1, cvec, r2);
                cache.Accumulate(x2);
                cache.Accumulate(r2);
            }
            cache.Flush();
            acc[tid].Normalize();

            Reduction(tid, tnum, ready, acc, linesize);
        }
#ifdef EXBLAS_MPI
        acc[0].Normalize();
        std::vector<int64_t> result(acc[0].get_f_words() + acc[0].get_e_words(), 0);
        MPI_Allreduce(&(acc[0].get_accumulator()[0]), &(result[0]), acc[0].get_f_words() + acc[0].get_e_words(), MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        Superaccumulator acc_fin(result);
        dacc = acc_fin.Round();
#else
        dacc = acc[0].Round();
#endif    

#ifdef EXBLAS_TIMING
        tend = rdtsc();
        t = double(tend - tstart) / N;
        mint = std::min(mint, t);
    }
    fprintf(stderr, "%f ", mint);
#endif

    return dacc;
}

