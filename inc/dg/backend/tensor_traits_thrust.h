#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "vector_categories.h"
#include "tensor_traits.h"

namespace dg
{
///@addtogroup dispatch
///@{

///@brief prototypical Shared Vector with Serial Tag
template<class T>
struct TensorTraits<thrust::host_vector<T> >//, std::enable_if_t< std::is_arithmetic<T>::value>>
{
    using value_type        = T;
    using tensor_category   = ThrustVectorTag;
    using execution_policy  = SerialTag;
};
//template<class T>
//struct TensorTraits<thrust::host_vector<T>,
//    std::enable_if_t< !std::is_arithmetic<T>::value>>
//{
//    using value_type        = get_value_type<T>;
//    using tensor_category   = RecursiveVectorTag;
//    using execution_policy  = get_execution_policy<T>;
//};

///@brief prototypical Shared Vector with Cuda or Omp Tag
template<class T>
struct TensorTraits<thrust::device_vector<T> >//, std::enable_if_t<std::is_arithmetic<T>::value>>
{
    using value_type        = T;
    using tensor_category   = ThrustVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ;
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
    using execution_policy  = OmpTag ;
#else
    using execution_policy  = SerialTag ;
#endif
};
///@}
///@cond
//thrust::cpp is an alias for thrust::system::cpp
//tag is a class deriving from thrust::execution_policy<tag>
//raw pointers have the std::random_access_iterator_tag as iterator_category in thrust
template<class Tag>
struct ThrustTag { };
template <>
struct ThrustTag<SerialTag>
{
    using thrust_tag = thrust::cpp::tag;
};
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template <>
struct ThrustTag<CudaTag>
{
    using thrust_tag = thrust::cuda::tag;
};
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
template <>
struct ThrustTag<OmpTag>
{
    using thrust_tag = thrust::omp::tag;
};
#endif
template<class Vector>
using get_thrust_tag = typename ThrustTag<get_execution_policy<Vector>>::thrust_tag;

///@endcond
} //namespace dg
