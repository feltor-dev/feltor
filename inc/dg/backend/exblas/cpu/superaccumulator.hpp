/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file cpu/blas1/superaccumulator.hpp
 *  \brief Provides a class to work with superaccumulators
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef SUPERACCUMULATOR_HPP_INCLUDED
#define SUPERACCUMULATOR_HPP_INCLUDED

#include <vector>
#include <stdint.h>
#include <iosfwd>
#include "mylibm.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

/**
 * \struct Superaccumulator
 * \ingroup ExSUM
 * \brief This class is meant to provide functionality for working with superaccumulators
 */
struct Superaccumulator
{
    /** 
     * Construction 
     * \param e_bits maximum exponent 
     * \param f_bits maximum exponent with significand
     */
    Superaccumulator(int e_bits = 1023, int f_bits = 1023 + 52);
   
    /** 
     * Construction 
     * \param acc another superaccumulator represented as a vector 
     * \param e_bits maximum exponent 
     * \param f_bits maximum exponent with significand
     */
    Superaccumulator(std::vector<int64_t> acc, int e_bits = 1023, int f_bits = 1023 + 52);

    /**
     * Function for accumulating values into superaccumulator
     * \param x value
     * \param exp exponent
     */ 
    void Accumulate(int64_t x, int exp);

    /**
     * Function for accumulating values into superaccumulator
     * \param x double-precision value
     */ 
    void Accumulate(double x);

    /**
     * Function for adding another supperaccumulator into the current
     * \param other superaccumulator
     */ 
    void Accumulate(Superaccumulator & other);   // May modify (normalize) other member

    /**
     * Function to perform correct rounding
     */
    double Round();
    
    /**< Characterizes the result of summation */
    enum Status
    {
        Exact, /**< Reproducible and accurate */
        Inexact, /**< non-accurate */
        MinusInfinity, /**< minus infinity */
        PlusInfinity, /**< plus infinity */
        Overflow, /**< overflow occurred */
        sNaN, /**< not-a-number */
        qNaN /**< not-a-number */
    };
 
    /**
     * Function to normalize the superaccumulator
     */
    bool Normalize();

    /**
     * Function to print the superaccumulator
     */
    void Dump(std::ostream & os);

    /**
     * Returns f_words
     */
    int get_f_words();

    /**
     * Returns e_words
     */
    int get_e_words();

    /**
     * Returns the superaccumulator, actually an array with results of summation
     */
    std::vector<int64_t> get_accumulator();

    /**
     * Sets the superaccumulator, actually an array of summation
     */
    void set_accumulator(std::vector<int64_t> other);

private:
    void AccumulateWord(int64_t x, int i);

    static constexpr unsigned int K = 8;    // High-radix carry-save bits
    static constexpr int digits = 64 - K;
    static constexpr double deltaScale = double(1ull << digits); // Assumes K>0


    int f_words, e_words;
    std::vector<int64_t> accumulator;
    int imin, imax;
    Status status;
    
    int64_t overflow_counter;
};


inline void Superaccumulator::AccumulateWord(int64_t x, int i)
{
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(i >= 0 && i < e_words + f_words);
    int64_t carry = x;
    int64_t carrybit;
    unsigned char overflow;
    int64_t oldword = xadd(accumulator[i], x, overflow);
    while(unlikely(overflow))
    {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // accumulator[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> digits;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1ll << K : -1ll << K);
        
        // Cancel carry-save bits
        xadd(accumulator[i], -(carry << digits), overflow);
        if(TSAFE && unlikely(s ^ overflow)) {
            // (Another) overflow of sign S
            carrybit *= 2;
        }
        
        carry += carrybit;

        ++i;
        if(i >= f_words + e_words) {
            status = Overflow;
            return;
        }
        oldword = xadd(accumulator[i], carry, overflow);
    }
}

inline void Superaccumulator::Accumulate(double x)
{
    if(x == 0) return;
    
    
    int e = exponent(x);
    int exp_word = e / digits;  // Word containing MSbit (upper bound)
    int iup = exp_word + f_words;
    
    double xscaled = myldexp(x, -digits * exp_word);

    int i;
    for(i = iup; xscaled != 0; --i) {

        double xrounded = myrint(xscaled);
        int64_t xint = myllrint(xscaled);
        AccumulateWord(xint, i);
        
        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
}

inline int Superaccumulator::get_f_words() {
    return f_words;
}

inline int Superaccumulator::get_e_words() {
    return e_words;
}

inline std::vector<int64_t> Superaccumulator::get_accumulator(){
    return accumulator;
}

inline void Superaccumulator::set_accumulator(std::vector<int64_t> other){
    accumulator = other;
}

#endif
