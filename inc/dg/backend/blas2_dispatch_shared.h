#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include "tensor_traits.h"
#include "matrix_categories.h"
#include "vector_categories.h"
#include "blas1_dispatch_shared.h"
#include "blas1_dispatch_vector.h"

///@cond
namespace dg{
namespace blas1{
//forward declare used blas1 functions
template<class from_ContainerType, class to_ContainerType>
inline void transfer( const from_ContainerType& source, to_ContainerType& target);

template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y);
}//namespace blas1
namespace blas2{
namespace detail{
template< class ContainerType1, class MatrixType, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( const ContainerType1& x, const MatrixType& m, const ContainerType2& y);

//thrust vector preconditioner
template< class Vector1, class Vector2>
void doTransfer( const Vector1& in, Vector2& out, AnyVectorTag, AnyVectorTag)
{
    dg::blas1::transfer(in,out);
}

template< class Vector1, class Matrix, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag, SharedVectorTag)
{
    static_assert( std::is_convertible<get_value_type<Vector1>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_convertible<get_value_type<Matrix>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_convertible<get_value_type<Vector2>, double>::value, "We only support double precision dot products at the moment!");
    //find out which one is the SharedVector and determine category and policy
    using execution_policy = get_execution_policy<Matrix>;
    static_assert( all_true<
            dg::has_any_or_same_policy<Vector1, execution_policy>::value,
            dg::has_any_or_same_policy<Vector2, execution_policy>::value
            >::value,
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");

    return dg::blas1::detail::doDot_dispatch( execution_policy(), m.size(), get_iterator(x), get_iterator(m), get_iterator(y));
}

template< class Vector1, class Matrix, class Vector2>
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag, RecursiveVectorTag)
{
    //find out which one is the RecursiveVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    auto size = std::get<vector_idx>(std::forward_as_tuple(x,y)).size();
    std::vector<std::vector<int64_t>> acc( size);
    for( unsigned i=0; i<size; i++)
        acc[i] = doDot_superacc( get_element(x,i), m, get_element(y,i));
    for( unsigned i=1; i<size; i++)
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

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              const Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SharedVectorTag, SharedVectorTag)
{
    dg::blas1::pointwiseDot( alpha, m, x, beta, y);
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              SharedVectorTag, SharedVectorTag)
{
    dg::blas1::pointwiseDot( 1, m, x, 0, y);
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              const Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SharedVectorTag, RecursiveVectorTag)
{
    for(unsigned i=0; i<x.size(); i++)
        dg::blas1::pointwiseDot( alpha, m, x[i], beta, y[i]);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              SharedVectorTag, RecursiveVectorTag)
{
    for(unsigned i=0; i<x.size(); i++)
        dg::blas1::pointwiseDot( 1, m, x[i], 0, y[i]);
}


}//namespace detail
} //namespace blas2
} //namespace dg
///@endcond
#endif //_DG_BLAS_PRECONDITIONER_
