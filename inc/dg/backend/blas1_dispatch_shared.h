#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vector_categories.h"
#include "scalar_categories.h"
#include "tensor_traits.h"

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



template< class T >
inline int get_vector_size( T&& x, AnyVectorTag) { return x.size();}
template< class T >
inline int get_vector_size( T&& x, AnyScalarTag) { return 1;}

template< class T>
inline int get_size( T&& x)
{
    return get_vector_size(std::forward<T>(x), get_tensor_category<T>());
}

template< class T, class... Ts>
inline int get_size( T&& x, Ts&&... xs)
{
    //bool test = T::nothin;
    if( is_scalar<T>::value)
        return get_size( std::forward<Ts>(xs)...);
    return get_vector_size(std::forward<T>(x), get_tensor_category<T>());
};

template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( SharedVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    using vector_type = find_first<dg::is_not_scalar, container, Containers...>;
    //static_assert( all_true<std::is_base_of<SharedVectorTag,
    //    get_tensor_category<Containers>>::value...>::value,
    //    "All container types must derive from the same vector category (SharedVectorTag in this case)!");
    //static_assert( all_true<std::is_same<get_execution_policy<container>,
    //    get_execution_policy<Containers> >::value...>::value,
    //    "All container types must share the same execution policy!");
#ifdef DG_DEBUG
    //is this possible?
    //assert( !x.empty());
    //assert( x.size() == xs.size() );
#endif //DG_DEBUG
    doSubroutine_dispatch(
            get_execution_policy<vector_type>(),
            get_size( x, xs...),
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
