#pragma once

#include <array>
#include <vector>
#include "vector_categories.h"
#include "type_traits.h"

namespace dg
{
///@addtogroup vec_list
///@{
template<class T>
struct TypeTraits<std::vector<T>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using data_layout   = ThrustVectorTag;
    using execution_policy  = OmpTag;
};
template<class T>
struct TypeTraits<std::vector<T>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using data_layout   = VectorVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

/**
 * @brief There is a special implementation of blas1 functions for a \c std::array different from the one indicated by SerialTag
 *
 * @tparam T arithmetic value type
 * @tparam N size of the array
 */
template<class T, std::size_t N>
struct TypeTraits<std::array<T, N>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using data_layout   = StdArrayTag;
    using execution_policy  = SerialTag;
};
template<class T, std::size_t N>
struct TypeTraits<std::array<T, N>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using data_layout   = ArrayVectorTag;
    using execution_policy  = get_execution_policy<T>;
};
///@}
} //namespace dg
