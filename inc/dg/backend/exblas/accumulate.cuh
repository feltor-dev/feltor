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

__device__
inline void AccumulateWordT( int64_t *accumulator, int i, int64_t x) {
    // With atomic superacc updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(i >= 0 && i < BIN_COUNT);
    unsigned char overflow;
    int64_t carry = x;
    int64_t carrybit;
    int64_t oldword = xadd(accumulator[i * WARP_COUNT], x, overflow);
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
        xadd(accumulator[i * WARP_COUNT], (int64_t) -(carry << DIGITS), overflow);
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
        oldword = xadd(accumulator[i * WARP_COUNT], carry, overflow);
    }
}

__device__
inline void AccumulateT( int64_t* accumulator, double x) { //transposed accumulation
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
        AccumulateWordT(accumulator, i, xint);

        xscaled -= xrounded;
        xscaled *= DELTASCALE;
    }
}
////////////////////////////////////////////////////////////////////////////////
// Normalize functions
////////////////////////////////////////////////////////////////////////////////
// Returns sign
// Does not really normalize! MW: what does that mean?
__device__
bool Normalize( int64_t *accumulator, int& imin, int& imax) {
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
__device__
int NormalizeT( int64_t *accumulator, int& imin, int& imax) {
    int64_t carry_in = accumulator[(imin)*WARP_COUNT] >> DIGITS;
    accumulator[(imin)*WARP_COUNT] -= carry_in << DIGITS;
    int i;
    // Sign-extend all the way
    for (i = imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i*WARP_COUNT] += carry_in;
        int64_t carry_out = accumulator[i*WARP_COUNT] >> DIGITS;    // Arithmetic shift
        accumulator[i*WARP_COUNT] -= (carry_out << DIGITS);
        carry_in = carry_out;
    }
    imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[imax*WARP_COUNT] += carry_in << DIGITS;

    return carry_in < 0;
}

////////////////////////////////////////////////////////////////////////////////
// Rounding functions
////////////////////////////////////////////////////////////////////////////////

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
