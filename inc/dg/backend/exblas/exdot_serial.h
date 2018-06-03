/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */

/**
 *  @file exdot_serial.h
 *  @brief Serial version of exdot
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

namespace exblas{
///@cond
namespace cpu{

template<typename CACHE>
void ExDOTFPE_cpu(int N, const double *a, const double *b, int64_t* acc) {
    CACHE cache(acc);
#ifndef _WITHOUT_VCL
    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
        vcl::Vec8d r1 ;
        vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
        //vcl::Vec8d x  = vcl::Vec8d().load(a+i)*vcl::Vec8d().load(b+i);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
    if( r != N) {
        //accumulate remainder
        vcl::Vec8d r1;
        vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
        //vcl::Vec8d x  = vcl::Vec8d().load_partial(N-r, a+r)*vcl::Vec8d().load_partial(N-r,b+r);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        double r1;
        double x = TwoProductFMA(a[i],b[i],r1);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
#endif// _WITHOUT_VCL
    cache.Flush();
}

template<typename CACHE>
void ExDOTFPE_cpu(int N, const double *a, const double *b, const double *c, int64_t* acc) {
    CACHE cache(acc);
#ifndef _WITHOUT_VCL
    int r = (( int64_t(N))  & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
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
    if( r != N) {
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
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        double x1 = a[i]*b[i];
        double x2 = x1*c[i];
        cache.Accumulate(x2);
    }
#endif// _WITHOUT_VCL
    cache.Flush();
}
}//namespace cpu
///@endcond

/*!@brief serial version of exact dot product
 *
 * Computes the exact sum \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param h_superacc pointer to an array of 64 bit integers (the superaccumulator) in host memory with size at least \c exblas::BIN_COUNT (39) (contents are overwritten)
 * @sa \c exblas::cpu::Round  to convert the superaccumulator into a double precision number
*/
template<size_t NBFPE=8>
void exdot_cpu(unsigned size, const double* x1_ptr, const double* x2_ptr, int64_t* h_superacc){
#ifndef _WITHOUT_VCL
    for( int i=0; i<exblas::BIN_COUNT; i++)
        h_superacc[i] = 0;
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc);
#else
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc);
#endif//_WITHOUT_VCL
}

/*!@brief gpu version of exact triple dot product
 *
 * Computes the exact sum \f[ \sum_{i=0}^{N-1} x_i w_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param x3_ptr third array
 * @param h_superacc pointer to an array of 64 bit integegers (the superaccumulator) in host memory with size at least \c exblas::BIN_COUNT (39) (contents are overwritten)
 * @sa \c exblas::cpu::Round  to convert the superaccumulator into a double precision number
 */
template<size_t NBFPE=8>
void exdot_cpu(unsigned size, const double *x1_ptr, const double* x2_ptr, const double * x3_ptr, int64_t* h_superacc) {
#ifndef _WITHOUT_VCL
    for( int i=0; i<exblas::BIN_COUNT; i++)
        h_superacc[i] = 0;
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc);
#else
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc);
#endif//_WITHOUT_VCL
}



}//namespace exblas
