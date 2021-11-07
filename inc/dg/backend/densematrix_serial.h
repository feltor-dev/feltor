#pragma once
#include <vector>
#include "exblas/ExSUM.FPE.hpp"
#include "densematrix.h"
#include "config.h"
#include "predicate.h"
#include "blas1_serial.h"

namespace dg
{
namespace blas2
{
namespace detail
{

//(r,s) <- a*b+c
template<typename T>
inline static T KnuthTwoFMA(T a, T b, T c, T & s)
{
    T r = DG_FMA( a,b,c);
    T z = DG_FMA(1., r, -c);
    s = DG_FMA(1., c - DG_FMA(1., r, -z), DG_FMA( a, b, - z));
    return r;
}

template<class T, unsigned NBFPE>
void AccumulateFPE( T a, T b, T* fpe)
{
    // MW : let's make it consistent with exdot
    T s = 0;
    T x = a*b;
    for(unsigned i = 0; i != NBFPE-1; ++i) {
        if( fpe[i] == 0)
            fpe[i] += x;
        else
            fpe[i] = KnuthTwoFMA(1., x, fpe[i], s);
        x = s;
    }
    //fpe[0] = KnuthTwoFMA( a, b, fpe[0], s);
    //for( unsigned i = 1; i != NBFPE-1; ++i) {
    //    T x = s;
    //    fpe[i] = KnuthTwoFMA(1., x, fpe[i], s);
    //}
    fpe[NBFPE-1] += s; // we throw away the rest

}
template<class Vector0, class Vector1, class T>
void doDenseSymv_scalar( unsigned num_cols, T alpha,
        const std::vector<const Vector0*>& matrix, const Vector1& x,
        T beta, T& y)
{
    constexpr unsigned NBFPE = 2;
    T fpe [NBFPE] = {0};
    for( unsigned k=0; k<num_cols; k++)
    {
        T a = (*matrix[k]);
        T b = x[k];
        AccumulateFPE<T,NBFPE>( a,b, fpe);
    }
    // multiply fpe with alpha
    T fpe2 [NBFPE] = {0};
    for( unsigned k=0; k<NBFPE; k++)
        AccumulateFPE<T,NBFPE>( alpha, fpe[k], fpe2);
    // Finally add beta*y
    AccumulateFPE<T,NBFPE>( beta, y, fpe2);
    // Finally sum up everything starting with smallest value
    y = 0;
    for( int k=(int)NBFPE-1; k>=0; k--)
        // round to nearest
        y = y + fpe2[k];
}

template<class Vector0, class Vector1, class T>
void doDenseSymv(SerialTag, unsigned num_rows, unsigned num_cols, T alpha,
        const std::vector<const Vector0*>& matrix, const Vector1& x,
        T beta, T* RESTRICT y)
{
    constexpr unsigned NBFPE = 2;
    for( unsigned i=0; i<num_rows; i++)
    {
        T fpe [NBFPE] = {0};
        for( unsigned k=0; k<num_cols; k++)
        {
            T a = (*matrix[k])[i];
            T b = x[k];
            AccumulateFPE<T,NBFPE>( a,b, fpe);
        }
        // multiply fpe with alpha
        T fpe2 [NBFPE] = {0};
        for( unsigned k=0; k<NBFPE; k++)
            AccumulateFPE<T,NBFPE>( alpha, fpe[k], fpe2);
        // Finally add beta*y
        AccumulateFPE<T,NBFPE>( beta, y[i], fpe2);
        // Finally sum up everything starting with smallest value
        y[i] = 0;
        for( int k=(int)NBFPE-1; k>=0; k--)
            // round to nearest
            y[i] = y[i] + fpe2[k];
    }
}

}//namespace detail
}//namespace blas2

} // namespace dg
