#pragma once

#include <array>
#include <vector>
#include "vector_categories.h"
#include "tensor_traits.h"

namespace dg
{
///@addtogroup dispatch
///@{

///@brief Prototypical Recursive Vector
template<class T>
struct TensorTraits<std::vector<T>,
    std::enable_if_t< !std::is_arithmetic<T>::value>>
{
    using value_type        = get_value_type<T>;
    using tensor_category   = RecursiveVectorTag;
    using execution_policy  = get_execution_policy<T>;
};
template<class T>
struct TensorTraits<std::vector<T>,
    std::enable_if_t< std::is_arithmetic<T>::value>>
{
    using value_type        = T;
    using tensor_category   = RecursiveScalarTag;
    using execution_policy  = SerialTag;
};

template<class T, std::size_t N>
struct TensorTraits<std::array<T, N>,
    std::enable_if_t< !std::is_arithmetic<T>::value>>
{
    using value_type        = get_value_type<T>;
    using tensor_category   = ArrayVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

template<class T, std::size_t N>
struct TensorTraits<std::array<T, N>,
    std::enable_if_t< std::is_arithmetic<T>::value>>
{
    using value_type        = T;
    using tensor_category   = RecursiveScalarTag;
    using execution_policy  = SerialTag;
};
///@}
} //namespace dg
