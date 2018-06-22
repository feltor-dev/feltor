#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include "vector_categories.h"
#include "scalar_categories.h"
#include "tensor_traits.h"
#include "predicate.h"

#include "blas1_serial.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "blas1_cuda.cuh"
#else
#include "blas1_omp.h"
#endif


///@cond
namespace dg
{
namespace blas1
{
///@cond
template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void subroutine( Subroutine f, ContainerType&& x, ContainerTypes&&... xs);
///@endcond
namespace detail
{

template< class To, class From>
To doTransfer( const From& in, ThrustVectorTag, ThrustVectorTag)
{
    To t( in.begin(), in.end());
    return t;
}

template< class Vector, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector& x, const Vector2& y, SharedVectorTag)
{
    static_assert( std::is_same<get_value_type<Vector>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_same<get_value_type<Vector2>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_base_of<SharedVectorTag,
        get_tensor_category<Vector2>>::value,
        "All container types must share the same vector category (SharedVectorTag in this case)!");
    static_assert( std::is_same<get_execution_policy<Vector>,
        get_execution_policy<Vector2> >::value,
        "All container types must share the same execution policy!");
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    const double* x_ptr = thrust::raw_pointer_cast( x.data());
    const double* y_ptr = thrust::raw_pointer_cast( y.data());
    return doDot_dispatch( get_execution_policy<Vector>(), x.size(), x_ptr, y_ptr);
}

template<class Vector, class Vector2>
get_value_type<Vector> doDot( const Vector& x, const Vector2& y, SharedVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,y,SharedVectorTag());
    return exblas::cpu::Round(acc.data());
}

//we need to distinguish between Scalars and Vectors

template<class T>
auto get_iterator( T&& v, AnyVectorTag) -> decltype(v.begin()){
    return v.begin();
}
template<class T>
thrust::constant_iterator<T> get_iterator( T&& v, AnyScalarTag){
    return thrust::constant_iterator<T>(v);
}

template<class T>
auto get_iterator( T&& v ) -> decltype( get_iterator( std::forward<T>(v), get_tensor_category<T>())) {
    return get_iterator( std::forward<T>(v), get_tensor_category<T>());
}

template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void doSubroutine( SharedVectorTag, Subroutine f, ContainerType&& x, ContainerTypes&&... xs)
{
    using vector_type = find_if_t<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    static_assert( !is_scalar<vector_type>::value,
            "At least one ContainerType must have a non-trivial execution policy!"); //Actually, we know that this is true at this point
    using execution_policy = get_execution_policy<vector_type>;
    static_assert( all_true<
            dg::has_any_or_same_policy<ContainerType, execution_policy>::value,
            dg::has_any_or_same_policy<ContainerTypes, execution_policy>::value...
            >::value,
        "All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
    using vector_type = find_if_t<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>;
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar_has_not_any_policy, get_value_type<ContainerType>, ContainerType, ContainerTypes...>::value;

//#ifdef DG_DEBUG
    //is this possible?
    //assert( !x.empty());
    //assert( x.size() == xs.size() );
//#endif //DG_DEBUG
    doSubroutine_dispatch(
            get_execution_policy<vector_type>(),
            std::get<vector_idx>(std::forward_as_tuple(x,xs...)).size(),
            f,
            get_iterator(x) ,
            get_iterator(xs) ...
            );
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_VECTOR_
