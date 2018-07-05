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

namespace exblas {
namespace cpu {
///////////////////////////////////////////////////////////////////////////
//********* Here, the change from float to double happens ***************//
///////////////////////////////////////////////////////////////////////////
#ifndef _WITHOUT_VCL
static inline vcl::Vec8d make_vcl_vec8d( double x, int i){
    return vcl::Vec8d(x);
}
static inline vcl::Vec8d make_vcl_vec8d( const double* x, int i){
    return vcl::Vec8d().load( x+i);
}
static inline vcl::Vec8d make_vcl_vec8d( double x, int i, int num){
    return vcl::Vec8d(x);
}
static inline vcl::Vec8d make_vcl_vec8d( const double* x, int i, int num){
    return vcl::Vec8d().load_partial( num, x+i);
}
static inline vcl::Vec8d make_vcl_vec8d( float x, int i){
    return vcl::Vec8d((double)x);
}
static inline vcl::Vec8d make_vcl_vec8d( const float* x, int i){
    double tmp[8];
    for(int i=0; i<8; i++)
        tmp[i] = (double)x[i];
    return vcl::Vec8d().load( tmp);
}
static inline vcl::Vec8d make_vcl_vec8d( float x, int i, int num){
    return vcl::Vec8d((double)x);
}
static inline vcl::Vec8d make_vcl_vec8d( const float* x, int i, int num){
    double tmp[8];
    for(int i=0; i<num; i++)
        tmp[i] = (double)x[i];
    return vcl::Vec8d().load_partial( num, tmp);
}
#endif//_WITHOUT_VCL
template<class T>
inline double get_element( T x, int i){
	return (double)x;
}
template<class T>
inline double get_element( const T* x, int i){
	return (double)(*(x+i));
}


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial superaccs
////////////////////////////////////////////////////////////////////////////////
///@cond
static inline void AccumulateWord( int64_t *accumulator, int i, int64_t x) {
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
        carrybit = (s ? 1ll << KRX : -1ll << KRX);

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
* @ingroup lowlevel
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the CPU (representing the superaccumulator)
* @param x the double to add to the superaccumulator
*/
static inline void Accumulate( int64_t* accumulator, double x) {
    if (x == 0)
        return;


    int e = cpu::exponent(x);
    int exp_word = e / DIGITS;  // Word containing MSbit (upper bound)
    int iup = exp_word + F_WORDS;

    double xscaled = cpu::myldexp(x, -DIGITS * exp_word);

    int i;
    for (i = iup; xscaled != 0; --i) {
        double xrounded = cpu::myrint(xscaled);
        int64_t xint = cpu::myllrint(xscaled);
        AccumulateWord(accumulator, i, xint);

        xscaled -= xrounded;
        xscaled *= DELTASCALE;
    }
}
#ifndef _WITHOUT_VCL
static inline void Accumulate( int64_t* accumulator, vcl::Vec8d x) {
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
* @ingroup lowlevel
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the CPU (representing the superaccumulator)
* @param imin the first index in the accumulator
* @param imax the last index in the accumulator
*
* @return  carry in bit (sign)
*/
static inline bool Normalize( int64_t *accumulator, int& imin, int& imax) {
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
* @ingroup highlevel
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the CPU (representing the superaccumulator)
* @return the double precision number nearest to the superaccumulator
*/
static inline double Round( int64_t * accumulator) {
    int imin = IMIN;
    int imax = IMAX;
    bool negative = Normalize(accumulator, imin, imax);

    // Find leading word
    int i;
    // Skip zeroes
    for(i = imax; accumulator[i] == 0 && i >= imin; --i) {
    }
    if (negative) {
        // Skip ones
        for(; (accumulator[i] & ((1ll << DIGITS) - 1)) == ((1ll << DIGITS) - 1) && i >= imin; --i) {
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
