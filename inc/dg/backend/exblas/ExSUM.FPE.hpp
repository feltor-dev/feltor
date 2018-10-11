/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file ExSUM.FPE.hpp
 *  \brief A set of routines concerning floating-point expansions
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#ifndef EXSUM_FPE_HPP_
#define EXSUM_FPE_HPP_
#include "accumulate.h"

namespace exblas
{
namespace cpu
{

/**
 * \struct FPExpansionTraits
 * \ingroup lowlevel
 * \brief This struct is meant to specify optimization or other technique used
 */
template<bool EX=false, bool FLUSHHI=false, bool H2SUM=false, bool CRF=false, bool CSWAP=false, bool B2SUM=true, bool SORT=false, bool VICT=false>
struct FPExpansionTraits
{
    static bool constexpr EarlyExit = EX;
    static bool constexpr FlushHi = FLUSHHI;
    static bool constexpr Horz2Sum = H2SUM;
    static bool constexpr CheckRangeFirst = CRF;
    static bool constexpr ConditionalSwap = CSWAP;
    static bool constexpr Biased2Sum = B2SUM;
    static bool constexpr Sort = SORT;
    static bool constexpr Victimcache = VICT;
};

/**
 * \struct FPExpansionVect
 * \ingroup lowlevel
 * \brief This struct is meant to introduce functionality for working with
 *  floating-point expansions in conjuction with superaccumulators
 */
template<typename T, int N, typename TRAITS=FPExpansionTraits<false,false> >
struct FPExpansionVect
{
    /**
     * Constructor
     * \param sa superaccumulator
     */
    FPExpansionVect(int64_t* sa);

    /**
     * This function accumulates value x to the floating-point expansion
     * \param x input value
     */
    void Accumulate(T x);

    ////**
    // * This function accumulates two values x to the floating-point expansion
    // * \param x1 input value
    // * \param x2 input value
    // */
    //void Accumulate(T x1, T x2);

    /**
     * This function is used to flush the floating-point expansion to the superaccumulator
     */
    void Flush();

private:
    void FlushVector(T x) const;
    void Insert(T & x);
    void Insert(T & x1, T & x2);
    static void Swap(T & x1, T & x2);
    static T twosum(T a, T b, T & s);

    int64_t* superacc;

    // Most significant digits first!
#ifdef _MSC_VER
	_declspec(align(64)) T a[N];
#else
    T a[N] __attribute__((aligned(64)));
#endif
    T victim;
};

template<typename T, int N, typename TRAITS>
FPExpansionVect<T,N,TRAITS>::FPExpansionVect(int64_t * sa) :
    superacc(sa),
    victim(0)
{
    std::fill(a, a + N, 0);
}

// Knuth 2Sum.
template<typename T>
inline static T KnuthTwoSum(T a, T b, T & s)
{
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
}
template<typename T>
inline static T TwoProductFMA(T a, T b, T &d) {
    T p = a * b;
#ifdef _WITHOUT_VCL
    d = a*b-p;
#else
    d = vcl::mul_sub_x(a, b, p); //extra precision even if FMA is not available
#endif//_WITHOUT_VCL
    return p;
}

// Knuth 2Sum with FMAs
template<typename T>
inline static T FMA2Sum(T a, T b, T & s)
{
#ifndef _WITHOUT_VCL
    T r = a + b;
    T z = vcl::mul_sub(1., r, a);
    s = vcl::mul_add(1., a - vcl::mul_sub(1., r, z), b - z);
    return r;
#else
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
#endif//_WITHOUT_VCL
}

template<typename T, int N, typename TRAITS> UNROLL_ATTRIBUTE
void FPExpansionVect<T,N,TRAITS>::Accumulate(T x)
{
    // Experimental
    if(TRAITS::CheckRangeFirst && horizontal_or(abs(x) < abs(a[N-1]))) {
        FlushVector(x);
        return;
    }
    T s;
    for(unsigned int i = 0; i != N; ++i) {
        a[i] = twosum(a[i], x, s);
        x = s;
        if(TRAITS::EarlyExit && i != 0 && !horizontal_or(x)) return;
    }
    if(TRAITS::EarlyExit || horizontal_or(x)) {
        FlushVector(x);
    }
}

template<typename T, int N, typename TRAITS>
T FPExpansionVect<T,N,TRAITS>::twosum(T a, T b, T & s)
{
//#if INSTRSET > 7                       // AVX2 and later
	// Assume Haswell-style architecture with parallel Add and FMA pipelines
	return FMA2Sum(a, b, s);
//#else
    //if(TRAITS::Biased2Sum) {
    //    return BiasedSIMD2Sum(a, b, s);
    //}
    //else {
    //    return KnuthTwoSum(a, b, s);
    //}
//#endif
}

template<typename T, int N, typename TRAITS>
void FPExpansionVect<T,N,TRAITS>::Swap(T & x1, T & x2)
{
    //if(TRAITS::ConditionalSwap) {
    //    swap_if_nonzero(x1, x2);
    //}
    //else {
        std::swap(x1, x2);
    //}
}

template<typename T, int N, typename TRAITS> UNROLL_ATTRIBUTE
void FPExpansionVect<T,N,TRAITS>::Insert(T & x)
{
    if(TRAITS::Sort) {
        // Insert at tail. Unconditional version.
        // Rotate accumulators:
        // x <= a[0]
        // a[0] <= a[1]
        // a[1] <= a[2]
        // ...
        // a[N-2] <= a[N-1]
        // a[N-1] <= x
        //T xb = a[0];
        T xb = T().load_a((double*)&a[0]);
        for(int i = 0; i != N-1; ++i)
        {
            //a[i] = a[i+1];
            T v;
            v.load_a((double*)&a[i+1]);
            v.store_a((double*)&a[i]);
        }
        //a[N-1] = x;
        x.store_a((double*)&a[N-1]);
        x = xb;
    }
    else {
        // Insert at head
        // Conditional or unconditional
        Swap(x, a[0]);
    }
}

template<typename T, int N, typename TRAITS> UNROLL_ATTRIBUTE
void FPExpansionVect<T,N,TRAITS>::Insert(T & x1, T & x2)
{
    if(TRAITS::Sort) {
        // x1 <= a[0]
        // x2 <= a[1]
        // a[0] <= a[2]
        // a[1] <= a[3]
        // a[i] <= a[i+2]
        // a[N-3] <= a[N-1]
        // a[N-2] <= x1
        // a[N-1] <= x2
        T x1b = a[0];
        T x2b = a[1];
        for(int i = 0; i != N-2; ++i) {
            a[i] = a[i+2];
        }
        a[N-2] = x1;
        a[N-1] = x2;
        x1 = x1b;
        x2 = x2b;
    }
    else {
        Swap(x1, a[0]);
        Swap(x2, a[1]);
    }
}

#undef IACA
#undef IACA_START
#undef IACA_END

template<typename T, int N, typename TRAITS>
void FPExpansionVect<T,N,TRAITS>::Flush()
{
    for(unsigned int i = 0; i != N; ++i)
    {
        FlushVector(a[i]);
        a[i] = 0;
    }
    if(TRAITS::Victimcache) {
        FlushVector(victim);
    }
}

template<typename T, int N, typename TRAITS> inline
void FPExpansionVect<T,N,TRAITS>::FlushVector(T x) const
{
    // TODO: update status, handle Inf/Overflow/NaN cases
    // TODO: make it work for other values of 4
    exblas::cpu::Accumulate(superacc, x);
}

}//namespace cpu
}//namespace exblas
#endif // EXSUM_FPE_HPP_
