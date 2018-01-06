#pragma once


#include "std_vector_blas.cuh"
#include "matrix_categories.h"
///@cond
namespace dg
{
namespace blas2
{
namespace detail
{

template< class Matrix, class Vector>
inline void doSymv( 
              Matrix& m,
              const Vector& x, 
              Vector& y, 
              AnyMatrixTag,
              VectorVectorTag,
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    //assert( m.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doSymv( m, x[i], y[i], 
                       get_matrix_category<Matrix>(),
                       get_vector_category<typename Vector::value_type>(),
                       get_vector_category<typename Vector::value_type>() );
        
}

template< class Precon, class Vector>
inline void doSymv( 
              get_value_type<Vector> alpha,
              const Precon& m,
              const Vector& x, 
              get_value_type<Vector> beta,
              Vector& y, 
              AnyMatrixTag,
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    //assert( m.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doSymv( alpha, m, x[i], beta, y[i],
                       get_matrix_category<Precon>(),
                       get_vector_category<typename Vector::value_type >() );
}

template< class Matrix, class Vector>
inline get_value_type<Vector> doDot( 
              const Vector& x, 
              const Matrix& m,
              const Vector& y, 
              AnyMatrixTag,
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    std::vector<std::vector<int64_t>> acc( x.size());
    for( unsigned i=0; i<x.size(); i++)
        acc[i] = doDot_superacc( x[i], m, y[i],
                       get_matrix_category<Matrix>(),
                       get_vector_category<typename Vector::value_type>() );
    for( unsigned i=1; i<x.size(); i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return exblas::cpu::Round(&(acc[0][0]));
}
template< class Matrix, class Vector>
inline typename VectorTraits<Vector>::value_type  doDot( 
              const Matrix& m,
              const Vector& y, 
              AnyMatrixTag,
              VectorVectorTag)
{
    return doDot( y,m,y,AnyMatrixTag(),VectorVectorTag());
}


} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
