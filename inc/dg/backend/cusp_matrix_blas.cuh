#ifndef _DG_BLAS_LAPLACE_CUH
#define _DG_BLAS_LAPLACE_CUH

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <cusp/multiply.h>
#include <cusp/convert.h>
#include <cusp/array1d.h>

#include "matrix_categories.h"


namespace dg{

namespace blas2
{
namespace detail
{
template<class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& x, Matrix2& y, CuspMatrixTag, CuspMatrixTag)
{
    cusp::convert(x,y);
}

template< class Matrix, class Vector>
inline void doSymv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    ThrustVectorTag,
                    ThrustVectorTag  )
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( m.num_rows == m.num_cols );
    assert( m.num_rows == x.size() );
#endif //DG_DEBUG
    cusp::array1d_view< typename Vector::const_iterator> cx( x.cbegin(), x.cend());
    cusp::array1d_view< typename Vector::iterator> cy( y.begin(), y.end());
    cusp::multiply( m, cx, cy);
}

template< class Matrix, class Vector>
inline void doGemv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    ThrustVectorTag,
                    ThrustVectorTag  )
{
#ifdef DG_DEBUG
    assert( m.num_rows == y.size() );
    assert( m.num_cols == x.size() );
#endif //DG_DEBUG
    cusp::array1d_view< typename Vector::const_iterator> cx( x.cbegin(), x.cend());
    cusp::array1d_view< typename Vector::iterator> cy( y.begin(), y.end());
    cusp::multiply( m, cx, cy);
}

} //namespace detail
} //namespace blas2
} //namespace dg

#endif //_DG_BLAS_LAPLACE_CUH
