#pragma once

#include <array>
#include <vector>
#include <map>
#include <complex>

#include "vector_categories.h"
#include "tensor_traits.h"
#include "tensor_traits_scalar.h"
#include "predicate.h"

namespace dg
{
///@addtogroup traits
///@{

///@brief Prototypical Recursive Vector (unless vector of scalars)
template<class T>
struct TensorTraits<std::vector<T>,
    std::enable_if_t< !dg::is_scalar<T>::value>>
{
    using value_type        = get_value_type<T>;
    using tensor_category   = RecursiveVectorTag;
    using execution_policy  = get_execution_policy<T>;
};
template<class T>
struct TensorTraits<std::vector<T>,
    std::enable_if_t< dg::is_scalar<T>::value>>
{
    using value_type        = T;
    using tensor_category   = ThrustVectorTag;
    using execution_policy  = SerialTag;
};

template<class T, std::size_t N>
struct TensorTraits<std::array<T, N>,
    std::enable_if_t< !dg::is_scalar<T>::value>>
{
    using value_type        = get_value_type<T>;
    using tensor_category   = ArrayVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

template<class T, std::size_t N>
struct TensorTraits<std::array<T, N>,
    std::enable_if_t< dg::is_scalar<T>::value>>
{
    using value_type        = T;
    using tensor_category   = ArrayScalarTag;
    using execution_policy  = SerialTag;
};

/// Behaves like a RecursiveVector
template<class Key, class T>
struct TensorTraits<std::map<Key,T>>
{
    using value_type        = get_value_type<T>;
    using tensor_category   = StdMapTag;
    using execution_policy  = get_execution_policy<T>;
};

/// Enable std::complex as a Scalar
template<class T>
struct TensorTraits<std::complex<T>>
{
    using value_type        = std::complex<T>;
    using tensor_category   = ComplexTag;
    using execution_policy  = AnyPolicyTag;
};
///@}
} //namespace dg
