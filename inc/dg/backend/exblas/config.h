/**
 *  @file config.h
 *  @brief Configuration of superaccumulators
 *
 *  @authors
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk 
 */
#pragma once

#include <stdint.h> //definition of int64_t

namespace exblas
{
////////////// parameters for superaccumulator operations //////////////////////
///High radix carray-save bits
static constexpr int KRX            =  8; //!< High-radix carry-save bits
static constexpr int DIGITS         =  64 - KRX; //!< number of nonoverlapping digits
static constexpr int F_WORDS        =  20;  //!< number of uper exponent words (64bits)
static constexpr int E_WORDS        =  19;  //!< number of lower exponent words (64bits)
static constexpr int BIN_COUNT     =  F_WORDS+E_WORDS; //!< size of superaccumulator (in 64 bit units)
static constexpr int IMIN           = 0; //!< first index in a superaccumulator
static constexpr int IMAX           = BIN_COUNT-1; //!< last index in a superaccumulator
static constexpr double DELTASCALE = double(1ull << DIGITS); //!< Assumes KRX>0

///@brief Characterizes the result of summation 
enum Status
{
    Exact, /*!< Reproducible and accurate */
    Inexact, /*!< non-accurate */
    MinusInfinity, /*!< minus infinity */
    PlusInfinity, /*!< plus infinity */
    Overflow, /*!< overflow occurred */
    sNaN, /*!< not-a-number */
    qNaN /*!< not-a-number */
};

/*! @brief Utility union to display all bits of a double (using "type-punning")
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


}//namespace exblas
