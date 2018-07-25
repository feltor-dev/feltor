/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 */

/**
 *  \file mylibm.hpp
 *  \brief Provides a set of auxiliary functions to superaccumulation.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */

#ifndef MYLIBM_HPP_INCLUDED
#define MYLIBM_HPP_INCLUDED

#include "config.h"

namespace exblas{
namespace cpu{

static inline int64_t myllrint(double x) {
#ifndef _WITHOUT_VCL
    return _mm_cvtsd_si64(_mm_set_sd(x));
#else
    return std::llrint(x);
#endif
}

static inline double myrint(double x)
{
#ifndef _WITHOUT_VCL
#if defined __GNUG__ || _MSC_VER
    // Workaround gcc bug 51033
    union {
        __m128d v;
        double d[2];
    } r;
    //_mm_round_sd(_mm_undefined_pd(), _mm_set_sd(x));
    //__m128d undefined;
    //r.v = _mm_round_sd(_mm_setzero_pd(), _mm_set_sd(x), _MM_FROUND_TO_NEAREST_INT);
    //r.v = _mm_round_sd(undefined, _mm_set_sd(x), _MM_FROUND_TO_NEAREST_INT);
    r.v = _mm_round_pd(_mm_set_sd(x), _MM_FROUND_TO_NEAREST_INT);
    return r.d[0];
#else
    double r;
    //asm("roundsd $0, %1, %0" : "=x" (r) : "x" (x));
    asm(ASM_BEGIN "roundsd %0, %1, 0" ASM_END : "=x" (r) : "x" (x));
    return r;
#endif
#else
    return std::rint(x);
#endif//_WITHOUT_VCL
}

//inline double uint64_as_double(uint64_t i)
//{
//    double d;
//    asm("movsd %0, %1" : "=x" (d) : "g" (i) :);
//    return d;
//}

static inline int exponent(double x)
{
    // simpler frexp
    union {
        double d;
        uint64_t i;
    } caster;
    caster.d = x;
    uint64_t e = ((caster.i >> 52) & 0x7ff) - 0x3ff;
    return e;
}

static inline int biased_exponent(double x)
{
    union {
        double d;
        uint64_t i;
    } caster;
    caster.d = x;
    uint64_t e = (caster.i >> 52) & 0x7ff;
    return e;
}

static inline double myldexp(double x, int e)
{
    // Scale x by e
    union {
        double d;
        uint64_t i;
    } caster;

    caster.d = x;
    caster.i += (uint64_t)e << 52;
    return caster.d;
}

static inline double exp2i(int e)
{
    // simpler ldexp
    union {
        double d;
        uint64_t i;
    } caster;

    caster.i = (uint64_t)(e + 0x3ff) << 52;
    return caster.d;
}

// Assumptions: th>tl>=0, no overlap between th and tl
static inline double OddRoundSumNonnegative(double th, double tl)
{
    // Adapted from:
    // Sylvie Boldo, and Guillaume Melquiond. "Emulation of a FMA and correctly rounded sums: proved algorithms using rounding to odd." IEEE Transactions on Computers, 57, no. 4 (2008): 462-471.
    union {
        double d;
        int64_t l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

#ifdef THREADSAFE
#define TSAFE 1
#define LOCK_PREFIX "lock "
#else
#define TSAFE 0
#define LOCK_PREFIX
#endif

// signedcarry in {-1, 0, 1}
inline static int64_t xadd(int64_t & memref, int64_t x, unsigned char & of)
{

#if defined _MSC_VER && !TSAFE //non-atomic load-ADDC-store
	int64_t y = memref;
	memref = y + x;
	int64_t x63 = (x >> 63) & 1;
	int64_t y63 = (y >> 63) & 1;
	int64_t r63 = (memref >> 63) & 1;
	int64_t c62 = r63 ^ x63 ^ y63;
	int64_t c63 = (x63 & y63) | (c62 & (x63 | y63));
	of = c63 ^ c62;
	return y;
#else
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    int64_t oldword = x;
#ifdef ATT_SYNTAX
    asm volatile (LOCK_PREFIX"xaddq %1, %0\n"
        "setob %2"
     : "+m" (memref), "+r" (oldword), "=q" (of) : : "cc", "memory");
#else
    asm volatile (LOCK_PREFIX"xadd %1, %0\n"
        "seto %2"
     : "+m" (memref), "+r" (oldword), "=q" (of) : : "cc", "memory");
#endif //ATT_SYNTAX
    return oldword;
#endif //_MSC_VER
}

//static inline vcl::Vec8d clear_significand(vcl::Vec8d x) {
//    return x & vcl::Vec8d(_mm512_castsi256_pd(_mm512_set1_epi64x(0xfff0000000000000ull)));
//}

//static inline double horizontal_max(vcl::Vec8d x) {
//    vcl::Vec4d h = x.get_high();
//    vcl::Vec4d l = x.get_low();
//    vcl::Vec4d m1 = max(h, l);
//    vcl::Vec4d m2 = vcl::permute4d<1, 0, 1, 0>(m1);
//    vcl::Vec4d m = vcl::max(m1, m2);
//    return m[0];    // Why is it so hard to convert from vector xmm register to scalar xmm register?
//}
//
//inline static void horizontal_twosum(vcl::Vec8d & r, vcl::Vec8d & s)
//{
//    //r = KnuthTwoSum(r, s, s);
//    transpose1(r, s);
//    r = KnuthTwoSum(r, s, s);
//    transpose2(r, s);
//    r = KnuthTwoSum(r, s, s);
//}
//
//static inline bool sign_horizontal_or (vcl::Vec8db const & a) {
//    //effectively tests if any element in a is non zero
//    return vcl::horizontal_or( a);
//    //return !_mm512_testz_pd(a,a);
//}
#ifndef _WITHOUT_VCL
inline static bool horizontal_or(vcl::Vec8d const & a) {
    //return _mm512_movemask_pd(a) != 0;
    vcl::Vec8db p = a != 0;
    return vcl::horizontal_or( p);
    //return !_mm512_testz_pd(p, p);
}
#else
inline static bool horizontal_or( const double & a){
    return a!= 0;
}
#endif//_WITHOUT_VCL


}//namespace cpu
}//namespace exblas

#endif
