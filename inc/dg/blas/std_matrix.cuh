#pragma once


#include "std_vector.cuh"
#include "../matrix_traits.h"
#include "../matrix_categories.h"
namespace dg
{
namespace blas2
{
namespace detail
{

template< class Matrix, class Vector>
inline void doSymv( 
              const std::vector<Matrix>& m,
              const std::vector<Vector>& x, 
              std::vector<Vector>& y, 
              StdMatrixTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doSymv( m[i], x[i], y[i], 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
        
}
template< class Matrix, class Vector>
inline void doSymv( 
              const std::vector<Matrix*>& m,
              const std::vector<Vector>& x, 
              std::vector<Vector>& y, 
              StdMatrixPointerTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doSymv( *(m[i]), x[i], y[i],
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
        
}

template< class Matrix, class Vector>
inline void doSymv( 
              typename MatrixTraits<Matrix>::value_type alpha,
              const std::vector<Matrix>& m,
              const std::vector<Vector>& x, 
              typename MatrixTraits<Matrix>::value_type beta,
              std::vector<Vector>& y, 
              StdMatrixTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doSymv( alpha, m[i], x[i], beta, y[i],
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

template< class Matrix, class Vector>
inline void doSymv( 
              typename MatrixTraits<Matrix>::value_type alpha,
              const std::vector<Matrix*>& m,
              const std::vector<Vector>& x, 
              typename MatrixTraits<Matrix>::value_type beta,
              std::vector<Vector>& y, 
              StdMatrixPointerTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doSymv( alpha, *(m[i]), x[i], beta, y[i],
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

template< class Matrix, class Vector>
inline typename Vector::value_type  doDot( 
              const std::vector<Vector>& x, 
              const std::vector<Matrix>& m,
              const std::vector<Vector>& y, 
              StdMatrixTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    typename Vector::value_type sum = 0;
    for( unsigned i=0; i<x.size(); i++)
        sum += doDot( x[i], m[i], y[i],
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
    return sum;
}
template< class Matrix, class Vector>
inline typename VectorTraits<Vector>::value_type  doDot( 
              const std::vector<Matrix>& m,
              const std::vector<Vector>& y, 
              StdMatrixTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    typename Vector::value_type sum = 0;
    for( unsigned i=0; i<y.size(); i++)
        sum += doDot( y[i], m[i], y[i],
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
    return sum;
}
//POINTER TAGS
template< class Matrix, class Vector>
inline typename Vector::value_type  doDot( 
              const std::vector<Vector>& x, 
              const std::vector<Matrix*>& m,
              const std::vector<Vector>& y, 
              StdMatrixPointerTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    typename Vector::value_type sum = 0;
    for( unsigned i=0; i<x.size(); i++)
        sum += doDot( x[i], *m[i], y[i],
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
    return sum;
}
template< class Matrix, class Vector>
inline typename VectorTraits<Vector>::value_type  doDot( 
              const std::vector<Matrix*>& m,
              const std::vector<Vector>& y, 
              StdMatrixPointerTag,
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( m.size() == y.size() );
#endif //DG_DEBUG
    typename Vector::value_type sum = 0;
    for( unsigned i=0; i<y.size(); i++)
        sum += doDot( y[i], *m[i], y[i],
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
    return sum;
}


} //namespace detail
} //namespace blas1
} //namespace dg
