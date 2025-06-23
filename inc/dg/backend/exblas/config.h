/**
 *  @file config.h
 *  @brief Configuration of superaccumulators
 *
 *  @authors
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once

#include <cstdint> //definition of int64_t
#include <cmath>
#include <cassert>
////////////////////////////////////////////////////////////////////////
//nvcc does not compile the avx512 instruction set, so do not include it
#ifdef __NVCC__
#define _WITHOUT_VCL
#endif //__NVCC__
#ifdef __APPLE__ //Condition for mac users to define without VCL
#define _WITHOUT_VCL
#endif//__APPLE__
#ifdef WITHOUT_VCL
#define _WITHOUT_VCL
#endif//WITHOUT_VCL

////////////////////////////////////////////////////////////////////////
//include vcl if available
#ifndef _WITHOUT_VCL

#define MAX_VECTOR_SIZE 512 //configuration of vcl
#define VCL_NAMESPACE vcl
// The vcl folder does not exist by default:
#ifdef VCL_NO_INCLUDE_PREFIX // used by cmake
#include "vectorclass.h"
#else
#include "vcl/vectorclass.h" //vcl by Agner Fog, may also include immintrin.h e.g.
#endif
#if INSTRSET <5
#define _WITHOUT_VCL
#pragma message("WARNING: Instruction set below SSE4.1! Deactivating vectorization!")
#elif INSTRSET <7
#pragma message( "NOTE: If available, it is recommended to activate AVX instruction set (-mavx) or higher")
#endif//INSTRSET

#endif//_WITHOUT_VCL

#if defined __INTEL_COMPILER
#define UNROLL_ATTRIBUTE
#elif defined __GNUC__

#ifdef __APPLE__ //MAC does not know "unroll-loops"
#define UNROLL_ATTRIBUTE
#elif defined __clang__
#define UNROLL_ATTRIBUTE
#else
#define UNROLL_ATTRIBUTE __attribute__((optimize("unroll-loops")))
#endif // __APPLE__

#else
#define UNROLL_ATTRIBUTE
#endif

#ifdef ATT_SYNTAX
#define ASM_BEGIN ".intel_syntax;"
#define ASM_END ";.att_syntax"
#else
#define ASM_BEGIN
#define ASM_END
#endif

// Debug mode
#define paranoid_assert(x) assert(x)
// Making C code less readable in an attempt to make assembly more readable
#if not defined _MSC_VER //there is no builtin_expect on msvc:
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

namespace dg
{
namespace exblas
{
////////////// parameters for superaccumulator operations //////////////////////
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
    //MW: not used anywhere but it would probably be useful to work it in?
    Exact, /*!< Reproducible and accurate */
    Inexact, /*!< non-accurate */
    MinusInfinity, /*!< minus infinity */
    PlusInfinity, /*!< plus infinity */
    Overflow, /*!< overflow occurred */
    sNaN, /*!< not-a-number */
    qNaN /*!< not-a-number */
};

///@cond
template<class T>
struct ValueTraits
{
    using value_type = T;
};
template<class T>
struct ValueTraits<T*>
{
    using value_type = T;
};
template<class U>
using has_floating_value = std::conditional_t< std::is_floating_point<typename ValueTraits<U>::value_type>::value, std::true_type, std::false_type>;
///@endcond

}//namespace exblas
} //namespace dg
