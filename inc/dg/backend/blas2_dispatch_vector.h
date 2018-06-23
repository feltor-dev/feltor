#pragma once

#include "blas1_dispatch_vector.h"
#include "matrix_categories.h"

///@cond
namespace dg
{
namespace blas2
{
//forward declare blas2 symv functions
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( MatrixType& M,
                  const ContainerType1& x,
                  ContainerType2& y);
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( get_value_type<ContainerType1> alpha,
                  MatrixType& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y);
namespace detail
{


template< class Vector1, class Matrix, class Vector2>
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, RecursiveVectorTag, RecursiveVectorTag)
{
    std::vector<std::vector<int64_t>> acc( m.size());
    for( unsigned i=0; i<m.size(); i++)
        acc[i] = doDot_superacc( x[i], m[i], y[i]);
    for( unsigned i=1; i<m.size(); i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return acc[0];
}

//In case the matrix is a RecursiveVector just do a recursive call
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              const Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              RecursiveVectorTag)
{
    for( unsigned i=0; i<m.size(); i++)
        dg::blas2::symv( alpha, m[i], x[i], beta, y[i]);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              RecursiveVectorTag)
{
    for( unsigned i=0; i<m.size(); i++)
        dg::blas2::symv( m[i], x[i], y[i]);
}



} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
