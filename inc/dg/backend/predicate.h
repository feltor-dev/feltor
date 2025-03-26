#pragma once

#include <type_traits>
#include <tuple>
#include "execution_policy.h"
#include "scalar_categories.h"
#include "tensor_traits.h"

namespace dg{
///@cond
namespace detail{
template<template <typename> class Predicate, unsigned n, class Default, class... Ts>
struct find_if_impl;

template<template <typename> class Predicate, unsigned n, class Default, class T>
struct find_if_impl<Predicate, n, Default, T>
{
    using type = std::conditional_t< Predicate<T>::value, T, Default>;
    static constexpr unsigned value = Predicate<T>::value ? n : n+1;
};

template<template <typename> class Predicate, unsigned n, class Default, class T, class... Ts>
struct find_if_impl<Predicate, n, Default, T, Ts...>
{
    using type = std::conditional_t< Predicate<T>::value, T, typename find_if_impl<Predicate, n+1, Default, Ts...>::type>;
    static constexpr unsigned value = Predicate<T>::value ? n : find_if_impl<Predicate, n+1, Default, Ts...>::value;
};
}//namespace detail

//access the element at position index
//we name it get_idx and not get so we do not get a conflict with std::get
template<size_t index, typename T, typename... Ts>
inline std::enable_if_t<index==0, T>
get_idx(T&& t, Ts&&... ts) {
    return std::forward<T>(t);
}

template<size_t index, typename T, typename... Ts>
inline std::enable_if_t<(index > 0) && index <= sizeof...(Ts),
          std::tuple_element_t<index, std::tuple<T, Ts...>>>
get_idx(T&& t, Ts&&... ts) {
    return get_idx<index-1>(std::forward<Ts>(ts)...);
}

//find first instance of a type that fulfills a predicate or false_type if non is found
template<template <typename> class Predicate, class Default, class T, class... Ts>
using find_if_t = typename detail::find_if_impl<Predicate,0, Default, T, Ts...>::type;
//find the corresponding element's index in the parameter pack
template<template <typename> class Predicate, class Default, class T, class... Ts>
using find_if_v = std::integral_constant<unsigned, detail::find_if_impl<Predicate,0, Default, T, Ts...>::value>;
///@endcond

/////////////////////////////////////////////////////////////////////////////////
/// @addtogroup dispatch
/// @{

/// @brief Does a type have a tensor_category derived from \c Tag
/// @sa dg::is_scalar_v
template< class T, class Tag = AnyScalarTag>
using is_scalar = typename std::is_base_of<Tag, get_tensor_category<T>>::type;
/// @brief Does a type have a tensor_category derived from \c Tag
/// @sa dg::is_vector_v
template< class T, class Tag = AnyVectorTag>
using is_vector = typename std::is_base_of<Tag, get_tensor_category<T>>::type;
/// @brief Does a type have a tensor_category derived from \c Tag
/// @sa dg::is_matrix_v
template< class T, class Tag = AnyMatrixTag>
using is_matrix = typename std::is_base_of<Tag, get_tensor_category<T>>::type;

/// Utility typedef
template< class T, class Tag = AnyScalarTag>
constexpr bool is_scalar_v = is_scalar<T, Tag>::value;
/// Utility typedef
template< class T, class Tag = AnyVectorTag>
constexpr bool is_vector_v = is_vector<T, Tag>::value;
/// Utility typedef
template< class T, class Tag = AnyMatrixTag>
constexpr bool is_matrix_v = is_matrix<T, Tag>::value;

/// @brief Does a type have an execution_policy equal to \c Tag
/// @sa dg::has_policy_v
template< class T, class Tag = AnyPolicyTag>
using has_policy = std::is_same<Tag, get_execution_policy<T>>;
/// Utility typedef
template< class T, class Tag = AnyPolicyTag>
constexpr bool has_policy_v = has_policy<T, Tag>::value;

/*! This is a utility class to get type information at compile time for
 * debugging purposes Use like
 * @code{.cpp}
 * dg::WhichType<T>{};
 * @endcode
 */
template< typename ...> struct WhichType;
/// @}
/// @cond

template< class T, class Tag = AnyScalarTag>
using is_not_scalar = std::conditional_t< !std::is_base_of<Tag, get_tensor_category<T>>::value, std::true_type, std::false_type>;

namespace detail
{
template<class Category>
using find_base_category = std::conditional_t< std::is_base_of<SharedVectorTag, Category>::value, SharedVectorTag,
        std::conditional_t< std::is_base_of<RecursiveVectorTag, Category>::value, RecursiveVectorTag, MPIVectorTag>>;
}//namesapce detail
//is scalar or same base vector category
template<class T, class Category>
using is_scalar_or_same_base_category = std::conditional_t< std::is_base_of<detail::find_base_category<Category>, get_tensor_category<T>>::value || is_scalar<T>::value , std::true_type, std::false_type>;

template< class T>
using has_any_policy = has_policy<T, AnyPolicyTag>;
template< class T>
using has_not_any_policy = std::conditional_t< !std::is_same<AnyPolicyTag, get_execution_policy<T>>::value, std::true_type, std::false_type>;
//has any or same policy tag
template<class U, class Policy>
using has_any_or_same_policy = std::conditional_t< std::is_same<get_execution_policy<U>, Policy>::value || has_any_policy<U>::value, std::true_type, std::false_type>;
//is not scalar and has a nontrivial policy
template< class T>
using is_not_scalar_has_not_any_policy = std::conditional_t< !is_scalar<T>::value && !has_any_policy<T>::value, std::true_type, std::false_type>;

/////////////////////////////////////////////////////////////////////////////////

/// @endcond


}//namespace dg
