#pragma once

#include "scalar_categories.h"
#include "tensor_traits.h"

namespace dg
{
///@addtogroup traits
///@{

/// Enable double and float as a floating point
template<class T>
struct TensorTraits<T, std::enable_if_t< std::is_floating_point_v<T>>>
{
    using value_type        = T;
    using tensor_category   = FloatingPointTag;
    using execution_policy  = AnyPolicyTag;
};
/// Enable integers and anything promotable to integer (such as bool and char) as integral
template<class T>
struct TensorTraits<T, std::enable_if_t< std::is_integral_v<T>>>
{
    using value_type        = T;
    using tensor_category   = IntegralTag;
    using execution_policy  = AnyPolicyTag;
};

///@}
} //namespace dg
