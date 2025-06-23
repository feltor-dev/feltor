#ifndef _DG_BLAS_SCALAR_
#define _DG_BLAS_SCALAR_

#include "scalar_categories.h"
#include "tensor_traits.h"
#include "predicate.h"

#include "blas1_serial.h"

///@cond
namespace dg
{
namespace detail
{
template< class To, class From, class ...Params>
To doConstruct( const From& from, AnyScalarTag, AnyScalarTag, Params&& ...)
{
    return To( from);
}
template< class From, class To, class ...Params>
void doAssign( const From& from, To& to, AnyScalarTag, AnyScalarTag, Params&& ...)
{
    to = from;
}
}//namespace detail
namespace blas1
{
namespace detail
{
template< class T, size_t N, class Functor, class ContainerType, class ...ContainerTypes>
inline void doDot_fpe( AnyScalarTag, int* , std::array<T,N>& fpe, Functor f,
    const ContainerType& x, const ContainerTypes& ...xs)
{
    fpe[0] = f(x,xs...);
    for( unsigned u=1; u<N; u++)
        fpe[u] = T(0);
}

template< class Vector1, class Vector2>
std::vector<int64_t> doDot_superacc( int* status, const Vector1& x, const Vector2& y, AnyScalarTag)
{
    //both Vectors are scalars
    static_assert( std::is_convertible_v<get_value_type<Vector1>, double>, "We only support double precision dot products at the moment!");
    static_assert( std::is_convertible_v<get_value_type<Vector2>, double>, "We only support double precision dot products at the moment!");
    const get_value_type<Vector1>* x_ptr = &x;
    const get_value_type<Vector2>* y_ptr = &y;
    //since we only accumulate up to two values (multiplication and rest) reduce the size of the FPE
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu<const get_value_type<Vector1>*, const
        get_value_type<Vector2>*, 2>( 1, x_ptr,y_ptr, &h_superacc[0], status) ;

    return h_superacc;
}

template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void doSubroutine( AnyScalarTag, Subroutine f, ContainerType&& x, ContainerTypes&&... xs)
{
    f(x,xs...);
}

template<class T, class ContainerType, class BinaryOp, class UnaryOp>
inline T doReduce( AnyScalarTag, ContainerType x, T init, BinaryOp op, UnaryOp
        unary_op)
{
    init = op( init, unary_op(x));
    return init;
}
template< class BinarySubroutine, class Functor, class ContainerType, class ...ContainerTypes>
inline void doKronecker( AnyScalarTag, ContainerType& y, BinarySubroutine f, Functor g, const ContainerTypes&... xs)
{
    f( g(xs...),y);
}


} //namespace detail
} //namespace blas1
namespace detail
{
template<class ContainerType, class Functor, class ...ContainerTypes>
auto doKronecker( AnyScalarTag, Functor f, const ContainerType& x0, const ContainerTypes& ... xs)
{
    return f(x0,xs...);
}
} //namespace detail
} //namespace dg
///@endcond

#endif //_DG_BLAS_SCALAR_
