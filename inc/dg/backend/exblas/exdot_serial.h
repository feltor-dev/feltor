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

namespace dg
{
namespace exblas{

/*! @brief Utility union to display all bits of a double (using <a href="https://en.wikipedia.org/wiki/Type_punning">type-punning</a>)
@code
double result; // = ...
udouble res;
res.d = result;
std::cout << "Result as double "<<res.d<<"  as integer "<<res.i<<std::endl;
@endcode
*/
union udouble{
    double d; //!< a double
    int64_t i; //!< a 64 bit integer
};

/*! @brief Utility union to display all bits of a float (using <a href="https://en.wikipedia.org/wiki/Type_punning">type-punning</a>)
@code
float result; // = ...
ufloat res;
res.f = result;
std::cout << "Result as float "<<res.f<<"  as integer "<<res.i<<std::endl;
@endcode
*/
union ufloat{
    float f; //!< a float
    int32_t i; //!< a 32 bit integer
};

///@cond
namespace cpu{

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2>
void ExDOTFPE_cpu(int N, PointerOrValue1 a, PointerOrValue2 b, int64_t* acc, bool* error) {
    CACHE cache(acc);
#ifndef _WITHOUT_VCL
    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
        //vcl::Vec8d r1 ;
        //vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,i), make_vcl_vec8d(b,i), r1);
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
        vcl::Vec8d x  = make_vcl_vec8d(a,i)* make_vcl_vec8d(b,i);
        vcl::Vec8db finite = vcl::is_finite( x);
        if( !vcl::horizontal_and( finite) ) *error = true;
        cache.Accumulate(x);
        //cache.Accumulate(r1);
    }
    if( r != N) {
        //accumulate remainder
        //vcl::Vec8d r1;
        //vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,r,N-r), make_vcl_vec8d(b,r,N-r), r1);
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
        vcl::Vec8d x  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);
        vcl::Vec8db finite = vcl::is_finite( x);
        if( !vcl::horizontal_and( finite) ) *error = true;
        cache.Accumulate(x);
        //cache.Accumulate(r1);
    }
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        //double r1;
        //double x = TwoProductFMA(get_element(a,i),get_element(b,i),r1);
        double x = (double)get_element(a,i)*(double)get_element(b,i);
        if( !std::isfinite(x) ) *error = true;
        cache.Accumulate(x);
        //cache.Accumulate(r1);
    }
#endif// _WITHOUT_VCL
    cache.Flush();
}

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2, typename PointerOrValue3>
void ExDOTFPE_cpu(int N, PointerOrValue1 a, PointerOrValue2 b, PointerOrValue3 c, int64_t* acc, bool* error) {
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
        //vcl::Vec8d x1  = vcl::mul_add(vcl::Vec8d().load(a+i),vcl::Vec8d().load(b+i), 0);
        //vcl::Vec8d x2  = vcl::mul_add( x1                   ,vcl::Vec8d().load(c+i), 0);
        vcl::Vec8d x1  = vcl::mul_add(make_vcl_vec8d(a,i),make_vcl_vec8d(b,i), 0);
        vcl::Vec8d x2  = vcl::mul_add( x1                ,make_vcl_vec8d(c,i), 0);
        vcl::Vec8db finite = vcl::is_finite( x2);
        if( !vcl::horizontal_and( finite) ) *error = true;
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
        vcl::Vec8d x1  = vcl::mul_add(make_vcl_vec8d(a,r,N-r),make_vcl_vec8d(b,r,N-r), 0);
        vcl::Vec8d x2  = vcl::mul_add( x1                    ,make_vcl_vec8d(c,r,N-r), 0);
        vcl::Vec8db finite = vcl::is_finite( x2);
        if( !vcl::horizontal_and( finite) ) *error = true;
        cache.Accumulate(x2);
        //cache.Accumulate(r2);
        //x2 = TwoProductFMA(r1, cvec, r2);
        //cache.Accumulate(x2);
        //cache.Accumulate(r2);
    }
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        double x1 = (double)get_element(a,i)*(double)get_element(b,i);
        double x2 = x1*(double)get_element(c,i);
        if( !std::isfinite(x2) ) *error = true;
        cache.Accumulate(x2);
    }
#endif// _WITHOUT_VCL
    cache.Flush();
}
}//namespace cpu
///@endcond

/*!@class hide_exdot2
 *
 * Accumulate the exact sum \f[ \sum_{i=0}^{N-1} x_i y_i \f] into a superaccumulator.
 * The superaccumulator is an array of \c exblas::BIN_COUNT (39) 64 bit integers that
 * represents a large fixed point number such that the summation is computed with virtually infinite precision and is thus bitwise reproducible even in a parallel environment.
 * @note The superaccumulator can be converted to
 * a double precision number using the Round function:
 * \ref dg::exblas::cpu::Round for cpu and omp version or \ref dg::exblas::gpu::Round for gpu version
 * @attention the product \f$ x_iy_i\f$ of numbers is **not** computed with infinite precision only the sum is (this does not break the reproducibility)
 * @note The algorithm is described in the paper
 <a href = "https://hal.archives-ouvertes.fr/hal-00949355v3">Sylvain Collange, David Defour, Stef Graillat, Roman Iakymchuk. "Numerical Reproducibility for the Parallel Reduction on Multi- and Many-Core Architectures", 2015. </a>
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 */

/*!@class hide_exdot3
 *
 * Accumulate the exact sum \f[ \sum_{i=0}^{N-1} x_i w_i y_i \f] into a superaccumulator.
 * The superaccumulator is an array of \c exblas::BIN_COUNT (39) 64 bit integers that
 * represents a large fixed point number such that the summation is computed with virtually infinite precision and is thus bitwise reproducible even in a parallel environment.
 * @note The superaccumulator can be converted to
 * a double precision number using the Round function:
 * \ref dg::exblas::cpu::Round for cpu and omp version or \ref dg::exblas::gpu::Round for gpu version
 * @attention the product \f$ x_iw_iy_i\f$ of numbers is **not** computed with infinite precision only the sum is (this does not break the reproducibility)
 * @note The algorithm is described in the paper
 <a href = "https://hal.archives-ouvertes.fr/hal-00949355v3">Sylvain Collange, David Defour, Stef Graillat, Roman Iakymchuk. "Numerical Reproducibility for the Parallel Reduction on Multi- and Many-Core Architectures", 2015. </a>
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param x3_ptr third array
 */
/*!@class hide_hostacc
 * @param h_superacc pointer to an array of 64 bit integegers (the
 * superaccumulator) in **host memory** with size at least \c exblas::BIN_COUNT
 * (39) (contents are overwritten, the function does not allocate memory i.e.
 * the memory needs to be allocated **before** calling the function)
 * @param status 0 indicates success, 1 indicates an input value was NaN or Inf
 */
/*!@class hide_deviceacc
 * @param d_superacc pointer to an array of 64 bit integegers (the
 * superaccumulator) in **device memory** with size at least \c
 * exblas::BIN_COUNT (39) (contents are overwritten, the function does not
 * allocate memory i.e. the memory needs to be allocated **before** calling the
 * function)
 * @param status 0 indicates success, 1 indicates an input value was NaN or Inf
 */

///@brief Serial version of exact dot product
///@copydoc hide_exdot2
///@copydoc hide_hostacc
template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE=8>
void exdot_cpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, int64_t* h_superacc, int* status){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
    for( int i=0; i<exblas::BIN_COUNT; i++)
        h_superacc[i] = 0;
    bool error = false;
#ifndef _WITHOUT_VCL
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc, &error);
#else
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, h_superacc, &error);
#endif//_WITHOUT_VCL
    *status = 0;
    if( error ) *status = 1;
}

///@brief Serial version of exact dot product
///@copydoc hide_exdot3
///@copydoc hide_hostacc
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3, size_t NBFPE=8>
void exdot_cpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, PointerOrValue3 x3_ptr, int64_t* h_superacc, int* status) {
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue3>::value, "PointerOrValue3 needs to be T or T* with T one of (const) float or (const) double");
    for( int i=0; i<exblas::BIN_COUNT; i++)
        h_superacc[i] = 0;
    bool error = false;
#ifndef _WITHOUT_VCL
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc, &error);
#else
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, x3_ptr, h_superacc, &error);
#endif//_WITHOUT_VCL
    *status = 0;
    if( error ) *status = 1;
}



}//namespace exblas
} //namespace dg
