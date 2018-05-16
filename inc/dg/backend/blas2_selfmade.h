#ifndef _DG_BLAS_SELFMADE_
#define _DG_BLAS_SELFMADE_
#include "vector_traits.h"
#include "matrix_traits.h"
//
///@cond
namespace dg{
namespace blas2{
namespace detail{


template<class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& x, Matrix2& y, AnyMatrixTag, SelfMadeMatrixTag)
{
    y = (Matrix2)x; //try to invoke the explicit conversion construction
}

template<class Vector1, class Matrix, class Vector2>
inline get_value_type<Matrix> doDot( const Vector1& x, const Matrix& m, const Vector2& y, SelfMadeMatrixTag)
{
    return m.dot(x,y);
}
template< class Matrix, class Vector>
inline get_value_type<Matrix> doDot( const Matrix& m, const Vector& x, SelfMadeMatrixTag)
{
    return m.dot(x);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              SelfMadeMatrixTag)
{
    m.symv( x,y);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SelfMadeMatrixTag)
{
    m.symv( alpha, x, beta, y);
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
//
#endif//_DG_BLAS_SELFMADE_
