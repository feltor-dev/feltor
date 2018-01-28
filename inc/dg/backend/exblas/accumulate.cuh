/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */
/**
 *  @file accumulate.cuh
 *  @brief The CUDA version of accumulate.h
 *
 *  @authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk 
 */
#pragma once
#include "config.h"
#include "mylibm.cuh"
//this file has a direct correspondance to cpu code accumulate.h

namespace exblas
{
namespace gpu
{

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial superaccs
////////////////////////////////////////////////////////////////////////////////
///@cond
__device__
inline void AccumulateWord( int64_t *accumulator, int i, int64_t x, int stride = 1) {
    // With atomic superacc updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(i >= 0 && i < BIN_COUNT);
    unsigned char overflow;
    int64_t carry = x;
    int64_t carrybit;
    int64_t oldword = xadd(accumulator[i * stride], x, overflow);
    // To propagate over- or underflow
    while (overflow) {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // accumulator[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> DIGITS;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1l << KRX : -1l << KRX);

        // Cancel carry-accumulatorve bits
        xadd(accumulator[i * stride], (int64_t) -(carry << DIGITS), overflow);
        if (TSAFE && (s ^ overflow)){ //MW: TSAFE is always 0
            // (Another) overflow of sign S
            carrybit *= 2;
        }
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT){
            //status = Overflow;
            return;
        }
        oldword = xadd(accumulator[i * stride], carry, overflow);
    }
}
///@endcond

/**
* @brief Accumulate a double to the superaccumulator (GPU version)
*
* @ingroup lowlevel
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the GPU (representing the superaccumulator)
* @param x the double to add to the superaccumulator
* @param stride stride in which accumulator is to be accessed
*/
__device__
inline void Accumulate( int64_t* accumulator, double x, int stride = 1) { //transposed accumulation
    if (x == 0)
        return;

    int e;
    frexp(x, &e); //extract the exponent of x (lies in -1024;1023 ?)
    int exp_word = e / DIGITS;  // Word containing MSbit
    int iup = exp_word + F_WORDS; //can be at most 18 + 20 

    double xscaled = ldexp(x, -DIGITS * exp_word);

    int i;
    for (i = iup; xscaled != 0; --i) {
        double xrounded = rint(xscaled);
        int64_t xint = (int64_t) xrounded;
        AccumulateWord(accumulator, i, xint, stride);

        xscaled -= xrounded;
        xscaled *= DELTASCALE;
    }
}
////////////////////////////////////////////////////////////////////////////////
// Normalize functions
////////////////////////////////////////////////////////////////////////////////
// Returns sign
// Does not really normalize! MW: what does that mean?
/**
* @brief Normalize a superaccumulator (GPU version)
*
* @ingroup lowlevel
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the GPU (representing the superaccumulator)
* @param imin the first index in the accumulator
* @param imax the last index in the accumulator
* @param stride strid in which the superaccumulator is accessed
*
* @return  carry in bit (sign)
*/
__device__
int Normalize( int64_t *accumulator, int& imin, int& imax, int stride = 1) {
    int64_t carry_in = accumulator[(imin)*stride] >> DIGITS;
    accumulator[(imin)*stride] -= carry_in << DIGITS;
    int i;
    // Sign-extend all the way
    for (i = imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i*stride] += carry_in;
        int64_t carry_out = accumulator[i*stride] >> DIGITS;    // Arithmetic shift
        accumulator[i*stride] -= (carry_out << DIGITS);
        carry_in = carry_out;
    }
    imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[imax*stride] += carry_in << DIGITS;

    return carry_in < 0;
}

////////////////////////////////////////////////////////////////////////////////
// Rounding functions
////////////////////////////////////////////////////////////////////////////////

/**
* @brief Convert a superaccumulator to the nearest double precision number (GPU version)
*
* @ingroup highlevel
* @param accumulator a pointer to at least \c BIN_COUNT 64 bit integers on the GPU (representing the superaccumulator)
* @return the double precision number nearest to the superaccumulator
*/
__device__
double Round( int64_t * accumulator) {
    int imin = IMIN;
    int imax = IMAX;
    int negative = Normalize(accumulator, imin, imax);

    // Find leading word
    int i;
    // Skip zeroes
    for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
    }
    if (negative) {
        // Skip ones
        for(; (accumulator[i] & ((1ll << DIGITS) - 1)) == ((1ll << DIGITS) - 1) && i >= imin;--i) {
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
    hiword -= (int64_t) rint(rounded);
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
        lo = OddRoundSumNonnegative(mid, lo);
    }
    // Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}

} //namespace gpu
} //namespace exblas
