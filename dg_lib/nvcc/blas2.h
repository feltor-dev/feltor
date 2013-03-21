#ifndef _DG_BLAS2_
#define _DG_BLAS2_

#include "blas/laplace.cuh"
#include "blas/preconditioner.cuh"
#include "vector_traits.h"
#include "matrix_traits.h"

namespace dg{
namespace blas2{

template< class Matrix, class Vector>
inline typename Matrix::value_type dot( const Vector& x, const Matrix& m, const Vector& y)
{
    return dg::blas2::detail::doDot( x, m, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() )
}

template< class Matrix, class Vector>
inline typename Matrix::value_type dot( const Matrix& m, const Vector& x)
{
    return dg::blas2::detail::doDot( m, x, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

template< class Matrix, class Vector>
inline void symv( typename Matrix::value_type alpha, 
                  const Matrix& m, 
                  const Vector& x, 
                  typename Matrix::value_type beta, 
                  Vector& y)
{
    return dg::blas2::detail::doSymv( alpha, m, x, beta, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

template< class Matrix, class Vector>
inline void symv( const Matrix& m, 
                  const Vector& x, 
                  Vector& y)
{
    return dg::blas2::detail::doSymv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

} //namespace blas2
} //namespace dg

#endif //_DG_BLAS2_
