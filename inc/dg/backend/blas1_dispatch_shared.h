#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vector_categories.h"
#include "vector_traits.h"

#include "blas1_serial.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "blas1_cuda.cuh"
#else
#include "blas1_omp.h"
#endif


namespace dg
{
///@addtogroup vec_list
///@{
template<class T>
struct VectorTraits<thrust::host_vector<T>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = ThrustVectorTag;
    using execution_policy  = SerialTag;
};
template<class T>
struct VectorTraits<thrust::host_vector<T>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using vector_category   = VectorVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

template<class T>
struct VectorTraits<thrust::device_vector<T>, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = ThrustVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ;  //!< enable if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#else
    using execution_policy  = OmpTag ;  //!< enable if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
#endif
};
///@}

///@cond
namespace blas1
{
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
    static_assert( std::is_base_of<SharedVectorTag,
        get_vector_category<Vector2>>::value,
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
get_value_type<Vector> doDot( const Vector& x, const Vector& y, SharedVectorTag)
{
    static_assert( std::is_same<get_value_type<Vector>, double>::value, "We only support double precision dot products at the moment!");
    std::vector<int64_t> acc = doDot_superacc( x,y,SharedVectorTag());
    return exblas::cpu::Round(acc.data());
}

template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( SharedVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    static_assert( all_true<std::is_base_of<SharedVectorTag,
        get_vector_category<Containers>>::value...>::value,
        "All container types must share the same vector category (SharedVectorTag in this case)!");
    static_assert( all_true<std::is_same<get_execution_policy<container>,
        get_execution_policy<Containers> >::value...>::value,
        "All container types must share the same execution policy!");
#ifdef DG_DEBUG
    //is this possible?
    //assert( !x.empty());
    //assert( x.size() == xs.size() );
#endif //DG_DEBUG
    doSubroutine_dispatch(
            get_execution_policy<container>(),
            x.size(),
            f,
            get_pointer_type<container>(  thrust::raw_pointer_cast(  x.data()) ),
            get_pointer_type<Containers>( thrust::raw_pointer_cast( xs.data()) )...
            );
}

} //namespace detail
} //namespace blas1
///@endcond
} //namespace dg

#endif //_DG_BLAS_VECTOR_
