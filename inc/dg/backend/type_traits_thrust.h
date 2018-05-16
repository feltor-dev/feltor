#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "vector_categories.h"
#include "type_traits.h"

namespace dg
{
///@addtogroup vec_list
///@{
template<class T>
struct TypeTraits<thrust::host_vector<T>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using data_layout       = ThrustVectorTag;
    using execution_policy  = SerialTag;
};
template<class T>
struct TypeTraits<thrust::host_vector<T>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using data_layout       = VectorVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

template<class T>
struct TypeTraits<thrust::device_vector<T>, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using data_layout       = ThrustVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ;  //!< enable if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#else
    using execution_policy  = OmpTag ;  //!< enable if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
#endif
};
///@}
///@cond
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
#else
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
