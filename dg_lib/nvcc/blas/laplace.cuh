#ifndef _DG_BLAS_LAPLACE_CUH
#define _DG_BLAS_LAPLACE_CUH

#include <cusp/multiply.h>
#include <cusp/array1d.h>

#include "../blas2.h"
#include "../matrix_categories.h"


namespace dg{

namespace blas2
{
namespace detail
{

template< class Matrix, class Vector>
void blas2::doSymv( const Matrix& m, const Vector&x, Vector& y, CuspMatrixTag, ThrustVectorTag  )
{
    cusp::array1d_view< typename Vector::const_iterator> cx( x.cbegin(), x.cend());
    cusp::array1d_view< typename Vector::iterator> cy( y.begin(), y.end());
    cusp::multiply( m, cx, cy);
}

} //namespace detail
} //namespace blas2
} //namespace dg

#endif //_DG_BLAS_LAPLACE_CUH
