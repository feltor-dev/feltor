#pragma once

#include <array>
#include <vector>
#include "vector_categories.h"
#include "tensor_traits.h"

namespace dg
{
///@addtogroup vec_list
///@{
template<class T>
struct TensorTraits<std::vector<T>>
{
    using value_type        = get_value_type<T>;
    using tensor_category   = RecursiveVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

/**
 * @brief There is a special implementation of blas1 functions for a \c std::array different from the one indicated by SerialTag
 *
 * @tparam T arithmetic value type
 * @tparam N size of the array
 */
template<class T, std::size_t N>
struct TensorTraits<std::array<T, N>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using tensor_category   = StdArrayTag;
    using execution_policy  = SerialTag;
};
template<class T, std::size_t N>
struct TensorTraits<std::array<T, N>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using tensor_category   = ArrayVectorTag;
    using execution_policy  = get_execution_policy<T>;
};
///@}
} //namespace dg
