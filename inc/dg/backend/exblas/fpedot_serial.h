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

namespace dg
{
namespace exblas{
///@cond
namespace cpu{

template<typename PointerOrValue1, typename PointerOrValue2, int NBFPE>
void fpeDOT(int N, PointerOrValue1 a, PointerOrValue2 b, std::array<double, NBFPE>& fpe, int * status) {
    // declare fpe for accumulating errors
    for( int i=0; i<NBFPE; i++)
        fpe[i] = 0;
#ifndef _WITHOUT_VCL
    std::array<vcl::Vec8d, NBFPE> fpe_;
    for( int i=0; i<NBFPE; i++)
        fpe_[i] = vcl::Vec8d(0.);
    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
        //vcl::Vec8d r1 ;
        //vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,i), make_vcl_vec8d(b,i), r1);
        vcl::Vec8d x  = make_vcl_vec8d(a,i)* make_vcl_vec8d(b,i);
        Accumulate(x, fpe_, status);
        //Accumulate(r1, fpe_, status);
        vcl::Vec8db finite = vcl::is_finite( x);
        if( !vcl::horizontal_and( finite) ) *status = 1;
    }
    if( r != N) {
        //accumulate remainder
        //vcl::Vec8d r1;
        //vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,r,N-r), make_vcl_vec8d(b,r,N-r), r1);
        vcl::Vec8d x  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);
        Accumulate(x, fpe_, status);
        //Accumulate(r1, fpe_, status);
        vcl::Vec8db finite = vcl::is_finite( x);
        if( !vcl::horizontal_and( finite) ) *status = 1;
    }
    //now sum everything into output fpe
    for( int k =0; k<NBFPE; k++)
        for(unsigned int i = 0; i != 8; ++i)
            Accumulate( fpe_[k][i], fpe, status);
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        //double r1;
        //double x = TwoProductFMA(get_element(a,i),get_element(b,i),r1);
        double x = get_element(a,i)*get_element(b,i);
        Accumulate(x, fpe, status);
        //Accumulate(r1, fpe, status);
        if( !std::isfinite(x) ) *status = 1;
    }
#endif// _WITHOUT_VCL
}

template<typename PointerOrValue1, typename PointerOrValue2, typename PointerOrValue3, int NBFPE>
void fpeDOT(int N, PointerOrValue1 a, PointerOrValue2 b, PointerOrValue3 c, std::array<double, NBFPE>& fpe, int * status) {
    for( int i=0; i<NBFPE; i++)
        fpe[i] = 0;
#ifndef _WITHOUT_VCL
    std::array<vcl::Vec8d, NBFPE> fpe_;
    for( int i=0; i<NBFPE; i++)
        fpe_[i] = 0.;
    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
        vcl::Vec8d x1  = make_vcl_vec8d(a,i)* make_vcl_vec8d(b,i);
        vcl::Vec8d x2  = x1*make_vcl_vec8d(c,i);
        Accumulate(x2, fpe_, status);
        vcl::Vec8db finite = vcl::is_finite( x2);
        if( !vcl::horizontal_and( finite) ) *status = 1;
    }
    if( r != N) {
        vcl::Vec8d x1  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);
        vcl::Vec8d x2  = x1*make_vcl_vec8d(c,r,N-r);
        Accumulate(x2, fpe_, status);
        vcl::Vec8db finite = vcl::is_finite( x2);
        if( !vcl::horizontal_and( finite) ) *status = 1;
    }
    //now sum everything into output fpe
    for( int k =0; k<NBFPE; k++)
        for(unsigned int i = 0; i != 8; ++i)
            Accumulate( fpe_[k][i], fpe, status);
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        double x1 = get_element(a,i)*get_element(b,i);
        double x2 = x1*get_element(c,i);
        Accumulate(x2, fpe, status);
        if( !std::isfinite(x2) ) *status = 1;
    }
#endif// _WITHOUT_VCL
}

}//namespace cpu
///@endcond

/*!@brief serial version of exact dot product
 *
 * Computes the exact dot \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param fpe the FPE holding the result (write-only)
 * @sa \c exblas::cpu::Round  to convert the FPE into a double precision number
*/
template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE>
void fpedot_cpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, std::array<double, NBFPE>& fpe, int * status){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");

    cpu::fpeDOT<PointerOrValue1,PointerOrValue2,NBFPE>(
            (int)size,x1_ptr,x2_ptr, fpe, status);
}

/*!@brief serial version of exact dot product
 *
 * Computes the exact dot \f[ \sum_{i=0}^{N-1} x_i w_i y_i\f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param x3_ptr third array
 * @param fpe the FPE holding the result (write-only)
 * @sa \c exblas::cpu::Round  to convert the FPE into a double precision number
*/
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3, size_t NBFPE>
void fpedot_cpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, PointerOrValue3 x3_ptr, std::array<double, NBFPE>& fpe, int * status){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue3>::value, "PointerOrValue3 needs to be T or T* with T one of (const) float or (const) double");

    cpu::fpeDOT<PointerOrValue1,PointerOrValue2,PointerOrValue3,NBFPE>(
            (int)size,x1_ptr,x2_ptr,x3_ptr, fpe, status);
}


}//namespace exblas
} //namespace dg
