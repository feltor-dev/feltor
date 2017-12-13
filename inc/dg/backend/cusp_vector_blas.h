#pragma once

#include <cassert>
#include <cusp/array1d.h>

#include "vector_categories.h"
#include "vector_traits.h"

namespace dg
{

template<class T>
struct VectorTraits<cusp::array1d<T,cusp::host_memory>, 
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = ThrustVectorTag;
    using execution_policy  = SerialTag;
};
template<class T>
struct VectorTraits<cusp::array1d<T,cusp::device_memory>, 
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = ThrustVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ; 
#else
    using execution_policy  = OmpTag ; 
#endif
};

} //namespace dg

