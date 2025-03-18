#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vector_categories.h"
#include "scalar_categories.h"
#include "tensor_traits.h"
#include "predicate.h"

#include "blas1_serial.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "blas1_cuda.cuh"
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include "blas1_omp.h"
#endif


///@cond
namespace dg
{
namespace detail
{
template< class To, class From, class ...Params>
To doConstruct( const From& from, ThrustVectorTag, ThrustVectorTag, Params&& ...ps)
{
    return To( from.begin(), from.end());
}
template< class From, class To, class ...Params>
void doAssign( const From& from, To& to, ThrustVectorTag, ThrustVectorTag, Params&& ...ps)
{
    to.assign( from.begin(), from.end());
}
template< class To, class From, class ...Params>
To doConstruct( const From& from, ArrayScalarTag, ArrayScalarTag, Params&& ...ps)
{
    return from;
}
template< class From, class To, class ...Params>
void doAssign( const From& from, To& to, ArrayScalarTag, ArrayScalarTag, Params&& ...ps)
{
    to = from;
}
}//namespace detail

namespace blas1
{
template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void subroutine( Subroutine f, ContainerType&& x, ContainerTypes&&... xs);

template< class ContainerType, class BinarySubroutine, class Functor, class ContainerType0, class ...ContainerTypes>
inline void kronecker( ContainerType& y, BinarySubroutine f, Functor g, const ContainerType0& x0, const ContainerTypes& ...xs);
namespace detail
{
template< class T, size_t N, class Functor, class ContainerType, class ...ContainerTypes>
inline void doDot_fpe( int* status,
    std::array<T,N>& fpe, Functor f, const ContainerType&, const ContainerTypes& ...xs);
template< class ContainerType1, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( int* status, const ContainerType1& x, const ContainerType2& y);
//we need to distinguish between Scalars and Vectors

///////////////////////////////////////////////////////////////////////////////////////////
template< class T, size_t N, class Functor, class ContainerType, class ...ContainerTypes>
inline void doDot_fpe( SharedVectorTag, int* status, std::array<T,N>& fpe, Functor f,
    const ContainerType& x, const ContainerTypes& ...xs)
{
    using vector_type = find_if_t<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    using execution_policy = get_execution_policy<vector_type>;
    static_assert( (dg::has_any_or_same_policy<ContainerType, execution_policy>::value &&
            ... &&  dg::has_any_or_same_policy<ContainerTypes, execution_policy>::value),
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>::value;
    doDot_fpe_dispatch(
            get_execution_policy<vector_type>(),
            status,
            get_idx<vector_idx>( x, xs...).size(),
            fpe,
            f,
            do_get_pointer_or_reference(x,get_tensor_category<ContainerType>()) ,
            do_get_pointer_or_reference(xs,get_tensor_category<ContainerTypes>()) ...
            );
}

template< class Vector1, class Vector2>
std::vector<int64_t> doDot_superacc( int* status, const Vector1& x, const Vector2& y, SharedVectorTag)
{
    static_assert( std::is_convertible_v<get_value_type<Vector1>, double>, "We only support double precision dot products at the moment!");
    static_assert( std::is_convertible_v<get_value_type<Vector2>, double>, "We only support double precision dot products at the moment!");
    //find out which one is the SharedVector and determine category and policy
    using vector_type = find_if_t<dg::is_not_scalar, Vector1, Vector1, Vector2>;
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    using execution_policy = get_execution_policy<vector_type>;
    static_assert(
            dg::has_any_or_same_policy<Vector1, execution_policy>::value &&
            dg::has_any_or_same_policy<Vector2, execution_policy>::value,
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
    //maybe assert size here?
    auto size = get_idx<vector_idx>(x,y).size();
    return dg::blas1::detail::doDot_dispatch( execution_policy(), status, size,
            do_get_pointer_or_reference(x, get_tensor_category<Vector1>()),
            do_get_pointer_or_reference(y, get_tensor_category<Vector2>()));
}

template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void doSubroutine( SharedVectorTag, Subroutine f, ContainerType&& x, ContainerTypes&&... xs)
{

    using vector_type = find_if_t<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    using execution_policy = get_execution_policy<vector_type>;
    static_assert( ( dg::has_any_or_same_policy<ContainerType, execution_policy>::value &&
            ... &&   dg::has_any_or_same_policy<ContainerTypes, execution_policy>::value),
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>::value;
    doSubroutine_dispatch(
            get_execution_policy<vector_type>(),
            get_idx<vector_idx>( std::forward<ContainerType>(x), std::forward<ContainerTypes>(xs)...).size(),
            f,
            do_get_pointer_or_reference(std::forward<ContainerType>(x),get_tensor_category<ContainerType>()) ,
            do_get_pointer_or_reference(std::forward<ContainerTypes>(xs),get_tensor_category<ContainerTypes>()) ...
            );
}

template<class T, class ContainerType, class BinaryOp, class UnaryOp>
inline T doReduce( SharedVectorTag, const ContainerType& x, T init, BinaryOp op, UnaryOp unary_op)
{
    return doReduce_dispatch( get_execution_policy<ContainerType>(), x.size(),
            thrust::raw_pointer_cast( x.data()), init, op, unary_op);
}

template<class T>
size_t do_get_size( const T& x, dg::SharedVectorTag){ return x.size();}
template<class T>
size_t do_get_size( const T& x, dg::AnyScalarTag){ return 1;}

template<class T>
size_t get_size( const T& x)
{
    return do_get_size( x, dg::get_tensor_category<T>());
}

template< class ContainerType, class BinarySubroutine, class Functor, class ...ContainerTypes>
inline void doKronecker( dg::SharedVectorTag, ContainerType& y, BinarySubroutine&& binary, Functor&& f, const ContainerTypes&... xs)
{
    constexpr size_t N = sizeof ...(ContainerTypes);
    std::array<size_t, N> sizes{ get_size(xs)...};
    unsigned size = 1;
    for( unsigned u=0; u<N; u++)
        size *= sizes[u];
    using vector_type = dg::find_if_t<dg::is_not_scalar_has_not_any_policy, dg::get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    using execution_policy = dg::get_execution_policy<vector_type>;
    static_assert(( dg::has_any_or_same_policy<ContainerType, execution_policy>::value &&
             ... && dg::has_any_or_same_policy<ContainerTypes, execution_policy>::value),
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
    doKronecker_dispatch(
            dg::get_execution_policy<vector_type>(),
            dg::do_get_pointer_or_reference(y,dg::get_tensor_category<ContainerType>()) ,
            size,
            std::forward<BinarySubroutine>(binary), std::forward<Functor>(f),
            sizes,
            dg::do_get_pointer_or_reference(xs,dg::get_tensor_category<ContainerTypes>()) ...
            );
}

} //namespace detail
} //namespace blas1

namespace detail
{
struct _equals
{
    template< class T1, class T2>
#ifdef __CUDACC__
__host__ __device__
#endif
    void operator()( T1 x, T2& y) const
    {
        y = x;
    }
};
template<class ContainerType, class Functor, class ...ContainerTypes>
auto doKronecker( SharedVectorTag, Functor&& f, const ContainerType& x0, const ContainerTypes& ... xs)
{
    constexpr size_t N = sizeof ...(ContainerTypes)+1;
    std::array<size_t, N> sizes{ dg::blas1::detail::get_size(x0), dg::blas1::detail::get_size(xs)...};
    unsigned size = 1;
    for( unsigned u=0; u<N; u++)
        size *= sizes[u];
    using vector_type = dg::find_if_t<dg::is_not_scalar_has_not_any_policy,
        dg::get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    using execution_policy = dg::get_execution_policy<vector_type>;
    using value_type = std::invoke_result_t<Functor,
        get_value_type<ContainerType>, get_value_type<ContainerTypes>...>;
    if constexpr (std::is_same_v<execution_policy, SerialTag>)
    {
        thrust::host_vector<value_type> y( size);
        dg::blas1::kronecker( y, _equals(), std::forward<Functor>(f), x0, xs ...);
        return y;
    }
    else
    {
        thrust::device_vector<value_type> y( size);
        dg::blas1::kronecker( y, _equals(), std::forward<Functor>(f), x0, xs ...);
        return y;
    }
}

} //namespace detail
} //namespace dg
///@endcond

#endif //_DG_BLAS_VECTOR_
