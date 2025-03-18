/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */
/**
 *  @file accumulate.h
 *  @brief Primitives for accumulation into superaccumulator
 *
 *  @authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include "config.h"
#include "mylibm.hpp"
//this file has a direct correspondance to gpu code accumulate.cuh

namespace dg
{
namespace exblas {
namespace cpu {
///@cond
///////////////////////////////////////////////////////////////////////////
//********* Here, the change from float to double happens ***************//
///////////////////////////////////////////////////////////////////////////
#ifndef _WITHOUT_VCL
inline vcl::Vec8d make_vcl_vec8d( double x, int i){
    return vcl::Vec8d(x);
}
inline vcl::Vec8d make_vcl_vec8d( const double* x, int i){
    return vcl::Vec8d().load( x+i);
}
inline vcl::Vec8d make_vcl_vec8d( double x, int i, int num){
    return vcl::Vec8d(x);
}
inline vcl::Vec8d make_vcl_vec8d( const double* x, int i, int num){
    return vcl::Vec8d().load_partial( num, x+i);
}
inline vcl::Vec8d make_vcl_vec8d( float x, int i){
    return vcl::Vec8d((double)x);
}
inline vcl::Vec8d make_vcl_vec8d( const float* x, int i){
    return vcl::Vec8d( x[i], x[i+1], x[i+2], x[i+3], x[i+4], x[i+5], x[i+6], x[i+7]);
}
inline vcl::Vec8d make_vcl_vec8d( float x, int i, int num){
    return vcl::Vec8d((double)x);
}
inline vcl::Vec8d make_vcl_vec8d( const float* x, int i, int num){
    double tmp[8];
    for(int j=0; j<num; j++)
        tmp[j] = (double)x[i+j];
    return vcl::Vec8d().load_partial( num, tmp);
}
#endif//_WITHOUT_VCL
template<class T>
inline T get_element( T x, int i){
	return x;
}
template<class T>
inline T get_element( T* x, int i){
	return *(x+i);
}
////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////

// Knuth 2Sum.
template<typename T>
inline std::enable_if_t<!std::is_integral_v<T>,T> KnuthTwoSum(T a, T b, T & s)
{
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
}
template<typename T> // for unsigned, int, char etc.
inline std::enable_if_t<std::is_integral_v<T>,T> KnuthTwoSum(T a, T b, T & s)
{
    s = 0;
    return a + b;
}

template<typename T>
inline T TwoProductFMA(T a, T b, T &d) {
    T p = a * b;
#ifdef _WITHOUT_VCL
    d = a*b-p;
#else
    d = vcl::mul_sub_x(a, b, p); //extra precision even if FMA is not available
#endif//_WITHOUT_VCL
    return p;
}

////////////////////////////////////////////////////////////////////////////////
// FPE Accumulate and Round function for leightweight implementation
////////////////////////////////////////////////////////////////////////////////
//does not check for NaN
template<typename T, size_t N> UNROLL_ATTRIBUTE
void Accumulate(T x, std::array<T,N>& fpe , int* status)
{
    if( x == T(0) )
        return;
    for(unsigned int i = 0; i != N; ++i) {
        T s;
        fpe[i] = KnuthTwoSum(fpe[i], x, s);
        x = s;
        if( x == T(0)) //early exit
	        return;
    }

    if (x != T(0) && *status != 1) {
        *status = 2;
    }
}
/**
* @brief Convert a fpe to the nearest number (CPU version)
*
* @param fpe a pointer to N doubles on the CPU (representing the fpe)
* @return the double precision number nearest to the fpe
*/
template<class T, size_t N>
inline T Round( const std::array<T,N>& fpe ) {
    // Our own implementation
    // Just accumulate to a FPE of size 2 and return sum;
    std::array<T, 2> fpe_red{T(0),T(0)};
    int status_red;
    for( unsigned u = 0; u<N; u++)
        Accumulate( fpe[u], fpe_red, &status_red);
    return fpe_red[0] + fpe_red[1];

    // The problem with the following is to get it to work for complex

    //// Sylvie Boldo, and Guillaume Melquiond. "Emulation of a FMA and correctly rounded sums: proved algorithms using rounding to odd." IEEE Transactions on Computers, 57, no. 4 (2008): 462-471.
    //// Listing 1 CorrectRoundedSum3
    //// xh = fpe[0], xm = fpe[1], xl = fpe[2]
    //static_assert( N > 2, "FPE size must be greater than 2");
    //union {
    //    double d;
    //    int64_t l;
    //} thdb;

    //double tl;
    //double th = KnuthTwoSum(fpe[1], fpe[2], tl);

    //if (tl != 0.0) {
    //    thdb.d = th;
    //    // if the mantissa of th is odd, there is nothing to do
    //    if (!(thdb.l & 1)) {
    //        // choose the rounding direction
    //        // depending of the signs of th and tl
    //        if ((tl > 0.0) ^ (th < 0.0))
    //            thdb.l++;
    //        else
    //            thdb.l--;
    //        th = thdb.d;
    //    }
    //}

    //// final addition rounded to nearest
    //return fpe[0] + th;
}

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial superaccs
////////////////////////////////////////////////////////////////////////////////
inline void AccumulateWord( int64_t *accumulator, int i, int64_t x) {
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(i >= 0 && i < BIN_COUNT);
    unsigned char overflow;
    int64_t carry = x;
    int64_t carrybit;
    int64_t oldword = cpu::xadd(accumulator[i], x, overflow);
    // To propagate over- or underflow
    while(unlikely(overflow)) {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // accumulator[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> DIGITS;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1ll << KRX : (unsigned long long)(-1ll) << KRX); //MW left shift of negative number is undefined so convert to unsigned

        // Cancel carry-save bits
        cpu::xadd(accumulator[i], (int64_t) -(carry << DIGITS), overflow);
        if(TSAFE && unlikely(s ^ overflow)) {
            // (Another) overflow of sign S
            carrybit *= 2;
        }
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT){
            //status = Overflow;
            return;
        }
        oldword = cpu::xadd(accumulator[i], carry, overflow);
    }
}
///@endcond

/**
* @brief Accumulate a double to the superaccumulator
*
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the CPU (representing the superaccumulator)
* @param x the double to add to the superaccumulator
*/
inline void Accumulate( int64_t* accumulator, double x) {
    if (x == 0)
        return;
    //assert( !std::isnan(x) && "Detected NaN in dot product!!");


    int e = cpu::exponent(x);
    int exp_word = e / DIGITS;  // Word containing MSbit (upper bound)
    int iup = exp_word + F_WORDS;

    double xscaled = cpu::myldexp(x, -DIGITS * exp_word);

    int i;
    for (i = iup; i>=0 && xscaled != 0; --i) { //MW: i>=0 protects against NaN
        double xrounded = cpu::myrint(xscaled);
        int64_t xint = cpu::myllrint(xscaled);
        AccumulateWord(accumulator, i, xint);

        xscaled -= xrounded;
        xscaled *= DELTASCALE;
    }
}
#ifndef _WITHOUT_VCL
inline void Accumulate( int64_t* accumulator, vcl::Vec8d x) {
    double v[8];
    x.store(v);

#if INSTRSET >= 7
    _mm256_zeroupper();
#endif
    for(unsigned int j = 0; j != 8; ++j) {
        exblas::cpu::Accumulate(accumulator, v[j]);
    }
}
#endif //_WITHOUT_VCL
////////////////////////////////////////////////////////////////////////////////
// Normalize functions
////////////////////////////////////////////////////////////////////////////////
// Returns sign
// Does not really normalize! MW: what does that mean?
//
/**
* @brief Normalize a superaccumulator
*
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the CPU (representing the superaccumulator)
* @param imin the first index in the accumulator
* @param imax the last index in the accumulator
*
* @return  carry in bit (sign)
*/
inline bool Normalize( int64_t *accumulator, int& imin, int& imax) {
    int64_t carry_in = accumulator[imin] >> DIGITS;
    accumulator[imin] -= carry_in << DIGITS;
    int i;
    // Sign-extend all the way
    for (i = imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        int64_t carry_out = accumulator[i] >> DIGITS;    // Arithmetic shift
        accumulator[i] -= (carry_out << DIGITS);
        carry_in = carry_out;
    }
    imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[imax] += carry_in << DIGITS;

    return carry_in < 0;
}

////////////////////////////////////////////////////////////////////////////////
// Rounding functions
////////////////////////////////////////////////////////////////////////////////


/**
* @brief Convert a superaccumulator to the nearest double precision number (CPU version)
*
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the CPU (representing the superaccumulator)
* @return the double precision number nearest to the superaccumulator
*/
inline double Round( int64_t * accumulator) {
    int imin = IMIN;
    int imax = IMAX;
    bool negative = Normalize(accumulator, imin, imax);

    // Find leading word
    int i;
    // Skip zeroes
    for(i = imax; i >= imin && accumulator[i] == 0; --i) {
        // MW: note that i >= imin has to come *before* accumulator[i]
        // else it is possible that accumulator[-1] is accessed
    }
    if (negative) {
        // Skip ones
        for(; i >= imin && (accumulator[i] & ((1ll << DIGITS) - 1)) == ((1ll << DIGITS) - 1); --i) {
        }
    }
    if (i < 0) {
        return 0.0;
    }

    int64_t hiword = negative ? ((1ll << DIGITS) - 1) - accumulator[i] : accumulator[i];
    double rounded = (double)hiword;
    double hi = ldexp(rounded, (i - F_WORDS) * DIGITS);
    if (i == 0) {
        return negative ? -hi : hi;  // Correct rounding achieved
    }
    hiword -= std::llrint(rounded);
    double mid = ldexp((double) hiword, (i - F_WORDS) * DIGITS);

    // Compute sticky
    int64_t sticky = 0;
    for (int j = imin; j != i - 1; ++j) {
        sticky |= negative ? ((1ll << DIGITS) - accumulator[j]) : accumulator[j];
    }

    int64_t loword = negative ? ((1ll << DIGITS) - accumulator[i - 1]) : accumulator[i - 1];
    loword |= !!sticky;
    double lo = ldexp((double) loword, (i - 1 - F_WORDS) * DIGITS);

    // Now add3(hi, mid, lo)
    // No overlap, we have already normalized
    if (mid != 0) {
        lo = cpu::OddRoundSumNonnegative(mid, lo);
    }
    // Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}

}//namespace cpu
} //namespace exblas
} //namespace dg
