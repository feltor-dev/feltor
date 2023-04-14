#pragma once

#include "scalar_categories.h"
#include "tensor_traits.h"

namespace dg
{
///@addtogroup traits
///@{

///@brief Recognize arithmetic types as scalars
template<class T>
struct TensorTraits<T, std::enable_if_t< std::is_arithmetic<T>::value>>
{
    using value_type        = T;
    using tensor_category   = ScalarTag;
    using execution_policy  = AnyPolicyTag;
};

///@}
} //namespace dg
