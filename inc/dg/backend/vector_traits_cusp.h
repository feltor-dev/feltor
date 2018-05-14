#pragma once

#include <cassert>
#include <cusp/array1d.h>

#include "vector_categories.h"
#include "vector_traits.h"

namespace dg
{

///@addtogroup vec_list
///@{
template<class T>
struct TypeTraits<cusp::array1d<T,cusp::host_memory>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using data_layout   = ThrustVectorTag;
    using execution_policy  = SerialTag;
};
template<class T>
struct TypeTraits<cusp::array1d<T,cusp::device_memory>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using data_layout   = ThrustVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ; //!< if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#else
    using execution_policy  = OmpTag ; //!< if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA

#endif
};
///@}

} //namespace dg
