#pragma once

#include <stdint.h> //definition of int64_t

namespace exblas
{
static constexpr uint NBFPE         =  3;  //size of floating point expansion
////////////// parameters for superaccumulator operations //////////////////////
static constexpr int KRX            =  8;  //High-radix carry-save bits
static constexpr int DIGITS         =  64 - KRX; //must be int because appears in integer expresssion
static constexpr int F_WORDS        =  20;
static constexpr int E_WORDS        =  19;
static constexpr int BIN_COUNT     =  F_WORDS+E_WORDS; //size of superaccumulator
static constexpr int IMIN           = 0;
static constexpr int IMAX           = BIN_COUNT-1;
//static constexpr int TSAFE          =  0;
static constexpr double DELTASCALE = double(1ull << DIGITS); // Assumes KRX>0

static constexpr uint WARP_COUNT     = 16 ; //# of sub superaccs in CUDA kernels

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

}//namespace exblas
