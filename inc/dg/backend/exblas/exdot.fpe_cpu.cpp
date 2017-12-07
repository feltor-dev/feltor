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

namespace exblas{

template<typename CACHE> 
Superaccumulator ExDOTFPE_cpu(int N, const double *a, const double *b) {
    assert( vcl::instrset_detect() >= 7);
    assert( vcl::hasFMA3() );
    Superaccumulator acc;
    CACHE cache(acc);

    int r = (( int64_t(N) ) & ~3ul);
    for(int i = 0; i < r; i+=4) {
        asm ("# myloop");
        //vcl::Vec8d r1 ;
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
        vcl::Vec8d x  = vcl::Vec8d().load(a+i)*vcl::Vec8d().load(b+i);
        cache.Accumulate(x);
        //cache.Accumulate(r1);
    }
    if( r != N) {
        //accumulate remainder
        //vcl::Vec8d r1;
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
        vcl::Vec8d x  = vcl::Vec8d().load_partial(N-r, a+r)*vcl::Vec8d().load_partial(N-r,b+r);
        cache.Accumulate(x);
        //cache.Accumulate(r1);
    }
    cache.Flush();
    return acc;
}
template<typename CACHE> 
Superaccumulator ExDOTFPE_cpu(int N, const double *a, const double *b, const double *c) {
    assert( vcl::instrset_detect() >= 7);
    assert( vcl::hasFMA3() );
    Superaccumulator acc;
    CACHE cache(acc);
    int r = (( int64_t(N))  & ~3ul);
    for(int i = 0; i < r; i+=4) {
        asm ("# myloop");
        //vcl::Vec8d r1 , r2, cvec = vcl::Vec8d().load(c+i);
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
        //vcl::Vec8d x2 = TwoProductFMA(x , cvec, r2);
        vcl::Vec8d x2  = (vcl::Vec8d().load(a+i)*vcl::Vec8d().load(b+i))*vcl::Vec8d().load(c+i);
        cache.Accumulate(x2);
        //cache.Accumulate(r2);
        //x2 = TwoProductFMA(r1, cvec, r2);
        //cache.Accumulate(x2);
        //cache.Accumulate(r2);
    }
    if( r != N) {
        //accumulate remainder
        //vcl::Vec8d r1 , r2, cvec = vcl::Vec8d().load_partial(N-r, c+r);
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
        //vcl::Vec8d x2 = TwoProductFMA(x , cvec, r2);
        vcl::Vec8d x2  = (vcl::Vec8d().load_partial(N-r,a+r)*vcl::Vec8d().load_partial(N-r,b+r))*vcl::Vec8d().load_partial(N-r,c+r);
        cache.Accumulate(x2);
        //cache.Accumulate(r2);
        //x2 = TwoProductFMA(r1, cvec, r2);
        //cache.Accumulate(x2);
        //cache.Accumulate(r2);
    }
    cache.Flush();
    return acc;
}
Superaccumulator exdot_cpu(int N, const double *a, const double* b, int fpe, bool early_exit) {
    if (fpe < 2) {
        fprintf(stderr, "Size of floating-point expansion must be in the interval [2, 8]\n");
        exit(1);
    }
    Superaccumulator acc;
    if (early_exit) {
        if (fpe <= 4)
            acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 4, FPExpansionTraits<true> > >)(N,a,b);
        if (fpe <= 6)
            acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 6, FPExpansionTraits<true> > >)(N,a,b);
        if (fpe <= 8)
            acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 8, FPExpansionTraits<true> > >)(N,a,b);
    } else { // ! early_exit
        if (fpe == 2) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 2> >)(N, a,b);
        if (fpe == 3) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 3> >)(N, a,b);
        if (fpe == 4) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 4> >)(N, a,b);
        if (fpe == 5) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 5> >)(N, a,b);
        if (fpe == 6) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 6> >)(N, a,b);
        if (fpe == 7) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 7> >)(N, a,b);
        if (fpe == 8) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 8> >)(N, a,b);
    }
    return acc;
}
Superaccumulator exdot_cpu(int N, const double *a, const double* b, const double * c, int fpe, bool early_exit) {
    if (fpe < 2) {
        fprintf(stderr, "Size of floating-point expansion must be in the interval [2, 8]\n");
        exit(1);
    }
    Superaccumulator acc;
    if (early_exit) {
        if (fpe <= 4)
            acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 4, FPExpansionTraits<true> > >)(N,a,b,c);
        if (fpe <= 6)
            acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 6, FPExpansionTraits<true> > >)(N,a,b,c);
        if (fpe <= 8)
            acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 8, FPExpansionTraits<true> > >)(N,a,b,c);
    } else { // ! early_exit
        if (fpe == 2) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 2> >)(N, a,b,c);
        if (fpe == 3) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 3> >)(N, a,b,c);
        if (fpe == 4) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 4> >)(N, a,b,c);
        if (fpe == 5) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 5> >)(N, a,b,c);
        if (fpe == 6) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 6> >)(N, a,b,c);
        if (fpe == 7) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 7> >)(N, a,b,c);
        if (fpe == 8) 
	    acc = (ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 8> >)(N, a,b,c);
    }
    return acc;
}


}//namespace exblas
