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
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, VectorVectorTag, VectorVectorTag)
{
    static_assert( std::is_base_of<VectorVectorTag,
        get_tensor_category<Vector2>>::value,
        "All data layouts must derive from the same vector category (VectorVectorTag in this case)!");
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    using inner_container1 = typename std::decay<Vector1>::type::value_type;
    using inner_container2 = typename std::decay<Matrix>::type::value_type;
    std::vector<std::vector<int64_t>> acc( x.size());
    for( unsigned i=0; i<x.size(); i++)
        acc[i] = doDot_superacc( x[i], m[i], y[i],
                       get_tensor_category<inner_container1>(),
                       get_tensor_category<inner_container2>() );
    for( unsigned i=1; i<x.size(); i++)
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
template< class Vector1, class Matrix, class Vector2>
inline get_value_type<Vector1> doDot( const Vector1& x, const Matrix& m, const Vector2& y, VectorVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,m,y,VectorVectorTag(), get_tensor_category<Vector1>());
    return exblas::cpu::Round(acc.data());
}
template< class Matrix, class Vector>
inline get_value_type<Matrix>  doDot(
              const Matrix& m,
              const Vector& y,
              VectorVectorTag
              )
{
    return doDot( y,m,y,VectorVectorTag());
}


//In case the matrix is a VectorVector just do a recursive call
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              const Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              VectorVectorTag)
{
    for( unsigned i=0; i<x.size(); i++)
        dg::blas2::symv( alpha, m[i], x[i], beta, y[i]);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              VectorVectorTag)
{
    for( unsigned i=0; i<x.size(); i++)
        dg::blas2::symv( m[i], x[i], y[i]);
}



} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
