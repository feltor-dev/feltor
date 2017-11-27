/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include "superaccumulator.hpp"
#include "mylibm.hpp"
#include <ostream>
#include <cassert>
#include <cmath>

#include <iostream>

Superaccumulator::Superaccumulator(int e_bits, int f_bits) :
    f_words((f_bits + digits - 1) / digits),   // Round up
    e_words((e_bits + digits - 1) / digits),
    accumulator(f_words + e_words, 0),
    imin(0), imax(f_words + e_words - 1),
    status(Exact),
    overflow_counter((1ll<<K)-1)
{
}

Superaccumulator::Superaccumulator(std::vector<int64_t> acc, int e_bits, int f_bits) :
    f_words((f_bits + digits - 1) / digits),   // Round up
    e_words((e_bits + digits - 1) / digits),
    accumulator(acc),
    imax(f_words + e_words - 1), imin(0),
    status(Exact),
    overflow_counter((1ll<<K)-1)
{
}

void Superaccumulator::Accumulate(int64_t x, int exp)
{
    Normalize();
    // Count from lsb to avoid signed arithmetic
    unsigned int exp_abs = exp + f_words * digits;
    int i = exp_abs / digits;
    int shift = exp_abs % digits;

    imin = std::min(imin, i);
    imax = std::max(imax, i+2);

    if(shift == 0) {
        // ignore carry
        AccumulateWord(x, i);
        return;
    }
    //        xh      xm    xl
    //        |-   ------   -|shift
    // |XX-----+|XX++++++|XX+-----|
    //   a[i+1]    a[i]

    int64_t xl = (x << shift) & ((1ll << digits) - 1);
    AccumulateWord(xl, i);
    x >>= digits - shift;
    if(x == 0) return;
    int64_t xm = x & ((1ll << digits) - 1);
    AccumulateWord(xm, i + 1);
    x >>= digits;
    if(x == 0) return;
    int64_t xh = x & ((1ll << digits) - 1);
    AccumulateWord(xh, i + 2);
}


void Superaccumulator::Accumulate(Superaccumulator & other)
{
    // Naive impl
    Normalize();
    other.Normalize();
    imin = std::min(imin, other.imin);
    imax = std::max(imax, other.imax);
    for(int i = imin; i <= imax; ++i) {
        accumulator[i] += other.accumulator[i];
    }
}

double Superaccumulator::Round()
{
    assert(digits >= 52);
    if(imin > imax) {
        return 0;
    }
    bool negative = Normalize();
    
    // Find leading word
    int i;
    // Skip zeroes
    for(i = imax;
        accumulator[i] == 0 && i >= imin;
        --i) {
    }
    if(negative) {
        // Skip ones
        for(;
            (accumulator[i] & ((1ll << digits) - 1)) == ((1ll << digits) - 1) && i >= imin;
            --i) {
        }
    }
    if(i < 0) {
        return 0.;
    }
    
    int64_t hiword = negative ? ((1ll << digits) - 1) - accumulator[i] : accumulator[i];
    double rounded = double(hiword);
    double hi = ldexp(rounded, (i - f_words) * digits);
    if(i == 0) {
        return negative ? -hi : hi;  // Correct rounding achieved
    }
    hiword -= llrint(rounded);
    double mid = ldexp(double(hiword), (i - f_words) * digits);
    
    // Compute sticky
    int64_t sticky = 0;
    for(int j = imin; j != i - 1; ++j) {
        sticky |= negative ? (1ll << digits) - accumulator[j] : accumulator[j];
    }
    
    int64_t loword = negative ? (1ll << digits) - accumulator[i-1] : accumulator[i-1];
    loword |=!! sticky;
    double lo = ldexp(double(loword), (i - 1 - f_words) * digits);
    
 
    // Now add3(hi, mid, lo)
    // No overlap, we have already normalized
    if(mid != 0) {
        lo = OddRoundSumNonnegative(mid, lo);
    }
    // Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}

// Returns sign
// Does not really normalize!
bool Superaccumulator::Normalize()
{
    if(imin > imax) {
        return false;
    }
    overflow_counter = 0;
    int64_t carry_in = accumulator[imin] >> digits;
    accumulator[imin] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for(i = imin + 1;
        i < f_words + e_words;
        ++i)
    {
        accumulator[i] += carry_in;
        int64_t carry_out = accumulator[i] >> digits;    // Arithmetic shift
        accumulator[i] -= (carry_out << digits);
        carry_in = carry_out;
    }
    imax = i - 1;
    // Do not cancel the last carry to avoid losing information
    accumulator[imax] += carry_in << digits;
    
    return carry_in < 0;
}

void Superaccumulator::Dump(std::ostream & os)
{
    switch(status) {
    case Exact:
        os << "Exact "; break;
    case Inexact:
        os << "Inexact "; break;
    case Overflow:
        os << "Overflow "; break;
    default:
        os << "??";
    }
    os << std::hex;
    for(int i = f_words + e_words - 1; i >= 0; --i) {
        int64_t hi = accumulator[i] >> digits;
        int64_t lo = accumulator[i] - (hi << digits);
        os << "+" << hi << " " << lo;
    }
    os << std::dec;
    os << std::endl;
}

