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
}//namespace detail

namespace blas1
{
template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void subroutine( Subroutine f, ContainerType&& x, ContainerTypes&&... xs);
template<class ContainerType, class BinaryOp>
inline get_value_type<ContainerType> reduce( const ContainerType& x, get_value_type<ContainerType> init, BinaryOp op);
namespace detail
{
template< class ContainerType1, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( const ContainerType1& x, const ContainerType2& y);
//we need to distinguish between Scalars and Vectors

///////////////////////////////////////////////////////////////////////////////////////////

template< class Vector1, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector1& x, const Vector2& y, SharedVectorTag)
{
    static_assert( std::is_convertible<get_value_type<Vector1>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_convertible<get_value_type<Vector2>, double>::value, "We only support double precision dot products at the moment!");
    //find out which one is the SharedVector and determine category and policy
    using vector_type = find_if_t<dg::is_not_scalar, Vector1, Vector1, Vector2>;
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    using execution_policy = get_execution_policy<vector_type>;
    static_assert( all_true<
            dg::has_any_or_same_policy<Vector1, execution_policy>::value,
            dg::has_any_or_same_policy<Vector2, execution_policy>::value
            >::value,
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
    //maybe assert size here?
    auto size = get_idx<vector_idx>(x,y).size();
    return dg::blas1::detail::doDot_dispatch( execution_policy(), size,
            do_get_pointer_or_reference(x, get_tensor_category<Vector1>()),
            do_get_pointer_or_reference(y, get_tensor_category<Vector2>()));
}

template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void doSubroutine( SharedVectorTag, Subroutine f, ContainerType&& x, ContainerTypes&&... xs)
{

    using vector_type = find_if_t<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    using execution_policy = get_execution_policy<vector_type>;
    static_assert( all_true<
            dg::has_any_or_same_policy<ContainerType, execution_policy>::value,
            dg::has_any_or_same_policy<ContainerTypes, execution_policy>::value...
            >::value,
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

template<class T, class ContainerType, class BinaryOp>
inline T doReduce( SharedVectorTag, const ContainerType& x, T init, BinaryOp op)
{
    return doReduce_dispatch( get_execution_policy<ContainerType>(), x.size(), thrust::raw_pointer_cast( x.data()), init, op);
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_VECTOR_
