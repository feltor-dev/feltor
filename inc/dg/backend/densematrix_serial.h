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
    T z = r - c;
    s = (c - (r - z)) + DG_FMA( a,b, -z);
    return r;
}

template<class T, unsigned NBFPE>
void AccumulateFPE( T a, T b, T* fpe)
{
    T s;
    fpe[0] = KnuthTwoFMA( a, b, fpe[0], s);
    for(unsigned i = 1; i != NBFPE-1; ++i) {
        T x = s;
        fpe[i] = dg::exblas::cpu::KnuthTwoSum(x, fpe[i], s);
    }
    fpe[NBFPE-1] += s; // we throw away the rest
}

template<class T, class Vector1>
void doDenseSymv(SerialTag, unsigned num_rows, unsigned num_cols, T alpha, const
        std::vector<const T*>& m_ptr, const Vector1& x,
        T beta, T* RESTRICT y)
{
    constexpr unsigned NBFPE = 2;
    for( unsigned i=0; i<num_rows; i++)
    {
        T fpe [NBFPE] = {0};
        for( unsigned k=0; k<num_cols; k++)
        {
            T a = m_ptr[k][i];
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
