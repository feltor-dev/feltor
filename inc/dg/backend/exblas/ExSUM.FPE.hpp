/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file cpu/blas1/ExSUM.FPE.hpp
 *  \brief Provides a set of routines concerning floating-point expansions
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk 
 */
#ifndef EXSUM_FPE_HPP_
#define EXSUM_FPE_HPP_
namespace exblas
{

/**
 * \struct FPExpansionTraits
 * \ingroup ExSUM
 * \brief This struct is meant ot specify optimization or other technique used
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
 * \ingroup ExSUM
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
    FPExpansionVect(Superaccumulator & sa);

    /** 
     * This function accumulates value x to the floating-point expansion
     * \param x input value
     */
    void Accumulate(T x);

    /** 
     * This function accumulates two values x to the floating-point expansion
     * \param x1 input value
     * \param x2 input value
     */
    void Accumulate(T x1, T x2);

    /**
     * This function is used to flush the floating-point expansion to the superaccumulator
     */
    void Flush();

    /**
     * This function is meant to be used for printing the floating-point expansion
     */
    void Dump() const;
private:
    void FlushVector(T x) const;
    void DumpVector(T x) const;
    void Insert(T & x);
    void Insert(T & x1, T & x2);
    static void Swap(T & x1, T & x2);
    static T twosum(T a, T b, T & s);

    Superaccumulator & superacc;

    // Most significant digits first!
    T a[N] __attribute__((aligned(32)));
    T victim;
};

template<typename T, int N, typename TRAITS>
FPExpansionVect<T,N,TRAITS>::FPExpansionVect(Superaccumulator & sa) :
    superacc(sa),
    victim(0)
{
    std::fill(a, a + N, 0);
}

// Knuth 2Sum.
template<typename T>
inline static T Knuth2Sum(T a, T b, T & s)
{
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
}
template<typename T>
inline static T TwoProductFMA(T a, T b, T &d) {
    T p = a * b;
    d = vcl::mul_sub_x(a, b, p); //extra precision even if FMA is not available
    return p;
}

//// Vector impl with test for fast path (MW: doesn't compile with g++ without -mavx and new vcl) 
//template<typename T>
//inline static T BiasedSIMD2Sum(T a, T b, T & s)
//{
//    T r = a + b;
//    auto doswap = abs(b) > abs(a);
//    //if(unlikely(!_mm512_testz_pd(doswap, doswap)))
//    //asm("nop");
//    if(/*unlikely*/(!_mm512_testz_si256(_mm512_castpd_si256(doswap), _mm512_castpd_si256(b))))  // any(doswap && b != +0)
//    {
//        // Slow path
//        T a2 = select(doswap, b, a);
//        T b2 = select(doswap, a, b);
//        a = a2;
//        b = b2;
//    }
//    s = (a - r) + b;
//    return r;
//}

// Knuth 2Sum with FMAs
template<typename T>
inline static T FMA2Sum(T a, T b, T & s)
{
    T r = a + b;
    T z = vcl::mul_sub(1., r, a);
    s = vcl::mul_add(1., a - vcl::mul_sub(1., r, z), b - z);
    return r;
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

static inline bool sign_horizontal_or (vcl::Vec8db const & a) {
    return vcl::horizontal_or( a);
    //return !_mm512_testz_pd(a,a);
}

//MW: did not get it to compile with the new Vec8d vectorclass
//// Input:
//// a3 a2 a1 a0
//// b3 b2 b1 b0
//// Output:
//// a3 b3 a1 b1
//// a2 b2 a0 b0
//inline static void transpose1(vcl::Vec8d & a, vcl::Vec8d & b)
//{
//    // a3 -- a1 --
//    // -- b3 -- b1
//    vcl::Vec8d a2 = vcl::blend8d<4|1,0|1,4|3,0|3>(a, b);
//    // a2 -- a0 --
//    // -- b2 -- b0
//    vcl::Vec8d b2 = vcl::blend8d<4|0,0|0,4|2,0|2>(a, b);
//    a = a2;
//    b = b2;
//}
//
//// Input:
//// a3 a2 a1 a0
//// b3 b2 b1 b0
//// Output:
//// a3 a2 b3 b2
//// a1 a0 b1 b0
//inline static void transpose2(vcl::Vec8d & a, vcl::Vec8d & b)
//{
//    // a3 a2 -- --
//    // -- -- b3 b2
//    vcl::Vec8d a2 = vcl::blend4d<4|2,4|3,0|2,0|3>(a, b);
//    // a1 a0 -- --
//    // -- -- b1 b0
//    vcl::Vec8d b2 = vcl::blend4d<4|0,4|1,0|0,0|1>(a, b);
//    a = a2;
//    b = b2;
//}
//
//inline static void horizontal_twosum(vcl::Vec8d & r, vcl::Vec8d & s)
//{
//    //r = Knuth2Sum(r, s, s);
//    transpose1(r, s);
//    r = Knuth2Sum(r, s, s);
//    transpose2(r, s);
//    r = Knuth2Sum(r, s, s);
//}

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
    //    return Knuth2Sum(a, b, s);
    //}
//#endif
}

//inline static void swap_if_nonzero(vcl::Vec8d & a, vcl::Vec8d & b)
//{
//    // if(a_i != 0) { a'_i = b_i; b'_i = a_i; }
//    // else {         a'_i = 0;   b'_i = b_i; }
//    vcl::Vec8db swapmask = (a != 0);
//    vcl::Vec8d b2 = select(swapmask, a, b);
//    a = b & vcl::Vec8d(swapmask);
//    b = b2;
//}

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



template<typename T, int N, typename TRAITS> UNROLL_ATTRIBUTE //INLINE_ATTRIBUTE //MW: is ignored?
void FPExpansionVect<T,N,TRAITS>::Accumulate(T x1, T x2)
{
    if(TRAITS::CheckRangeFirst) {
        auto p = abs(x1) < abs(a[N-1]);
        if(sign_horizontal_or(p)) {
            FlushVector(x1 & T(p));
            x1 = T(andnot(vcl::Vec8db(x1), p));
        }
        p = abs(x2) < abs(a[N-1]);
        if(sign_horizontal_or(p)) {
            FlushVector(x2 & T(p));
            x2 = T(andnot(vcl::Vec8db(x2), p));
        }
    }
    
    T s1, s2;
    for(unsigned int i = 0; i != N; ++i) {
        T ai = vcl::Vec8d().load_a((double*)(a+i));
        //T ai = a[i];
        ai = twosum(ai, x1, s1);
        ai = twosum(ai, x2, s2);
        ai.store_a((double*)(a+i));
        //a[i] = ai;
        x1 = s1;
        x2 = s2;
        if(TRAITS::EarlyExit && i != 0 && !horizontal_or(x1|x2)) return;
    }

    
    if(TRAITS::EarlyExit || (TRAITS::Horz2Sum && !TRAITS::Victimcache)) {
        // 1 check for both numbers
        if(TRAITS::EarlyExit || unlikely(horizontal_or(x1|x2))) {
            if(TRAITS::FlushHi) {
                Insert(x1, x2);
            }
            //if(TRAITS::Horz2Sum) {
            //    horizontal_twosum(x1, x2);
            //}
            FlushVector(x1);
            if(!TRAITS::Horz2Sum || horizontal_or(x2)) {
                FlushVector(x2);
            }
        }
    }
    else {
        // Separate checks
        if(unlikely(horizontal_or(x1))) {
            if(TRAITS::FlushHi) {
                Insert(x1);
            }
            // Compact if we can
            //if(TRAITS::Victimcache && TRAITS::Horz2Sum) {
            //    horizontal_twosum(victim, x1);
            //}
            if(!TRAITS::Horz2Sum || horizontal_or(x1)) {
                FlushVector(x1);
            }
        }
        if(unlikely(horizontal_or(x2))) {
            if(false && TRAITS::FlushHi) {  // Alternate flush low/high
                Insert(x2);
            }
            //if(TRAITS::Victimcache && TRAITS::Horz2Sum) {
            //    horizontal_twosum(victim, x2);
            //}
            if(!TRAITS::Horz2Sum || horizontal_or(x2)) {
                FlushVector(x2);
            }
        }
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
    double v[8];
    x.store(v);
    
#if INSTRSET >= 7
    _mm256_zeroupper();
#endif
    for(unsigned int j = 0; j != 8; ++j) {
        superacc.Accumulate(v[j]);
    }
}

template<typename T, int N, typename TRAITS>
void FPExpansionVect<T,N,TRAITS>::Dump() const
{
    for(unsigned int i = 0; i != N; ++i)
    {
        DumpVector(a[i]);
        std::cout << std::endl;
    }
}

template<typename T, int N, typename TRAITS>
void FPExpansionVect<T,N,TRAITS>::DumpVector(T x) const
{
    double v[8] __attribute__((aligned(32)));
    x.store_a(v);
    _mm256_zeroupper();
    
    for(unsigned int j = 0; j != 8; ++j) {
        printf("%a ", v[j]);
    }
}

}//namespace exblas
#endif // EXSUM_FPE_HPP_
