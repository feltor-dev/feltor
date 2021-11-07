#pragma once

#include <vector>
#include <omp.h>
#include "blas1_omp.h"
#include "densematrix_serial.h"

namespace dg{
namespace blas2{
namespace detail{
template<class Vector0, class Vector1, class T>
void doDenseSymv_omp(unsigned num_rows, unsigned num_cols, T alpha,
        const std::vector<const Vector0*>& matrix, const Vector1& x,
        T beta, T* RESTRICT y)
{
    constexpr unsigned NBFPE = 2;
#pragma omp for nowait
    for( unsigned i=0; i<num_rows; i++)
    {
        T fpe [NBFPE] = {0};
        for( unsigned k=0; k<num_cols; k++)
        {
            T a = (*matrix[k])[i];
            T b = x[ k];
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

template<class Vector0, class Vector1, class T>
void doDenseSymv(OmpTag, unsigned num_rows, unsigned num_cols, T alpha,
        const std::vector<const Vector0*>& m_ptr, const Vector1& x,
        T beta, T* RESTRICT y)
{
    if(omp_in_parallel())
    {
        doDenseSymv_omp( num_rows, num_cols, alpha, m_ptr, x, beta, y);
        return;
    }
    if(num_rows>dg::blas1::detail::MIN_SIZE)
    {
        #pragma omp parallel
        {
            doDenseSymv_omp( num_rows, num_cols, alpha, m_ptr, x, beta, y);
        }
    }
    else
        doDenseSymv( SerialTag(), num_rows, num_cols, alpha, m_ptr, x, beta, y);

}

} //namespace detail
} //namespace blas2
} //namespace dg
