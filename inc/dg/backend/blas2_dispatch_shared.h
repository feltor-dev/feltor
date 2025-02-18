#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#include "tensor_traits.h"
#include "matrix_categories.h"
#include "vector_categories.h"
#include "blas1_dispatch_shared.h"
#include "blas1_dispatch_vector.h"

#include "blas2_stencil.h"

///@cond
namespace dg{

template<class from_ContainerType, class ContainerType, class ...Params>
inline void assign( const from_ContainerType& from, ContainerType& to, Params&& ... ps);

namespace blas1{
template< class ContainerType, class ContainerType1, class ContainerType2, class value_type, class value_type1>
inline void pointwiseDot( value_type alpha, const ContainerType1& x1, const ContainerType2& x2, value_type1 beta, ContainerType& y);
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( const ContainerType1& x1, const ContainerType2& x2, ContainerType& y);
}//namespace blas1

namespace blas2{
//forward declare blas2 functions
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void symv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y);
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void symv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y);
namespace detail{

template< class ContainerType1, class MatrixType, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( int * status, const ContainerType1& x, const MatrixType& m, const ContainerType2& y);

//thrust vector preconditioner
template< class Vector1, class Vector2>
void doTransfer( const Vector1& in, Vector2& out, AnyVectorTag, AnyVectorTag)
{
    dg::assign(in,out);
}

template< class Vector1, class Matrix, class Vector2>
std::vector<int64_t> doDot_superacc( int* status, const Vector1& x,
    const Matrix& m, const Vector2& y, SharedVectorTag, SharedVectorTag)
{
    static_assert( std::is_convertible_v<get_value_type<Vector1>, double>,
        "We only support double precision dot products at the moment!");
    static_assert( std::is_convertible_v<get_value_type<Matrix>, double>,
        "We only support double precision dot products at the moment!");
    static_assert( std::is_convertible_v<get_value_type<Vector2>, double>,
        "We only support double precision dot products at the moment!");
    //find out which one is the SharedVector and determine category and policy
    using execution_policy = get_execution_policy<Matrix>;
    static_assert( dg::has_any_or_same_policy<Vector1, execution_policy>::value &&
            dg::has_any_or_same_policy<Vector2, execution_policy>::value,
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");

    return dg::blas1::detail::doDot_dispatch( execution_policy(), status,
        m.size(),
        do_get_pointer_or_reference(x,get_tensor_category<Vector1>()),
        do_get_pointer_or_reference(m,get_tensor_category<Matrix>()),
        do_get_pointer_or_reference(y,get_tensor_category<Vector2>()));
}

template< class Vector1, class Matrix, class Vector2>
inline std::vector<int64_t> doDot_superacc( int* status, const Vector1& x,
    const Matrix& m, const Vector2& y, SharedVectorTag, RecursiveVectorTag)
{
    //find out which one is the RecursiveVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    auto size = get_idx<vector_idx>(x,y).size();
    std::vector<std::vector<int64_t>> acc( size);
    for( unsigned i=0; i<size; i++)
        acc[i] = doDot_superacc( status,
            do_get_vector_element(x,i,get_tensor_category<Vector1>()), m,
            do_get_vector_element(y,i,get_tensor_category<Vector2>()));
    for( unsigned i=1; i<size; i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<=exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return acc[0];
}

template< class Stencil, class ContainerType, class ...ContainerTypes>
inline void doParallelFor( SharedVectorTag, Stencil f, unsigned N, ContainerType&& x, ContainerTypes&&... xs)
{
    // a copy of doSubroutine ...

    using vector_type = find_if_t<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    using execution_policy = get_execution_policy<vector_type>;
    static_assert((
            dg::has_any_or_same_policy<ContainerType, execution_policy>::value &&
      ... &&dg::has_any_or_same_policy<ContainerTypes, execution_policy>::value),
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
    doParallelFor_dispatch(
            get_execution_policy<vector_type>(),
            N,
            f,
            do_get_pointer_or_reference(std::forward<ContainerType>(x),get_tensor_category<ContainerType>()) ,
            do_get_pointer_or_reference(std::forward<ContainerTypes>(xs),get_tensor_category<ContainerTypes>()) ...
            );
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SharedVectorTag, SharedVectorTag)
{
    dg::blas1::pointwiseDot( alpha, std::forward<Matrix>(m), x, beta, y);
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix&& m,
              const Vector1& x,
              Vector2& y,
              SharedVectorTag, SharedVectorTag)
{
    dg::blas1::pointwiseDot( std::forward<Matrix>(m), x, y);
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SharedVectorTag, RecursiveVectorTag)
{
    for(unsigned i=0; i<x.size(); i++)
        dg::blas2::symv( alpha, std::forward<Matrix>(m), do_get_vector_element(x,i,get_tensor_category<Vector1>()), beta, do_get_vector_element(y,i,get_tensor_category<Vector2>()));
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix&& m,
              const Vector1& x,
              Vector2& y,
              SharedVectorTag, RecursiveVectorTag)
{
    for(unsigned i=0; i<y.size(); i++)
        dg::blas2::symv( std::forward<Matrix>(m), do_get_vector_element(x,i,get_tensor_category<Vector1>()), do_get_vector_element(y,i,get_tensor_category<Vector2>()));
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SharedVectorTag, StdMapTag)
{
    for( auto el : y)
        dg::blas2::symv( alpha, std::forward<Matrix>(m), do_get_vector_element(x,el.first,get_tensor_category<Vector1>()), beta, do_get_vector_element(y,el.first,get_tensor_category<Vector2>()));
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix&& m,
              const Vector1& x,
              Vector2& y,
              SharedVectorTag, StdMapTag)
{
    for( auto el : y)
        dg::blas2::symv( std::forward<Matrix>(m), do_get_vector_element(x,el.first,get_tensor_category<Vector1>()), do_get_vector_element(y,el.first,get_tensor_category<Vector2>()));
}


}//namespace detail
} //namespace blas2
} //namespace dg
///@endcond
#endif //_DG_BLAS_PRECONDITIONER_
