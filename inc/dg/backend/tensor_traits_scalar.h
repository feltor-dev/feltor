#pragma once

#include "scalar_categories.h"
#include "tensor_traits.h"

namespace dg
{
///@addtogroup dispatch
///@{

///@brief Recognize arithmetic types as scalars
template<class T>
struct TensorTraits<T, typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using tensor_category   = ScalarTag;
    using execution_policy  = AnyPolicyTag;
};

///@}
} //namespace dg
