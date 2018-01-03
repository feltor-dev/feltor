#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "accumulate.h"
#include "ExSUM.FPE.hpp"

namespace exblas{

template<typename CACHE> 
void ExDOTFPE_cpu(int N, const double *a, const double *b, int64_t* acc) {
    assert( vcl::instrset_detect() >= 7);
    //assert( vcl::hasFMA3() );
    CACHE cache(acc);

    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
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
}

template<typename CACHE> 
void ExDOTFPE_cpu(int N, const double *a, const double *b, const double *c, int64_t* acc) {
    assert( vcl::instrset_detect() >= 7);
    //assert( vcl::hasFMA3() );
    CACHE cache(acc);
    int r = (( int64_t(N))  & ~7ul);
    for(int i = 0; i < r; i+=8) {
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
}

void exdot_cpu(unsigned size, const double* x1_ptr, const double* x2_ptr, int64_t* h_superacc){
    assert( vcl::instrset_detect() >= 7);
    //assert( vcl::hasFMA3() );
    ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 8, FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc);
}

void exdot_cpu(unsigned size, const double *x1_ptr, const double* x2_ptr, const double * x3_ptr, int64_t* h_superacc) {
    assert( vcl::instrset_detect() >= 7);
    //assert( vcl::hasFMA3() );
    ExDOTFPE_cpu<FPExpansionVect<vcl::Vec8d, 8, FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc);
}



}//namespace exblas
