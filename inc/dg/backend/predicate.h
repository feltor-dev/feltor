#pragma once

#include <type_traits>
#include "execution_policy.h"
#include "scalar_categories.h"
#include "tensor_traits.h"

namespace dg{
namespace detail{
template<template <typename> class Predicate, unsigned n, class... Ts>
struct find_if_impl;

template<template <typename> class Predicate, unsigned n, class T>
struct find_if_impl<Predicate, n, T>
{
    using type = typename std::conditional< Predicate<T>::value, T, std::false_type>::type;
    static constexpr unsigned value = Predicate<T>::value ? n : n+1;
};

template<template <typename> class Predicate, unsigned n, class T, class... Ts>
struct find_if_impl<Predicate, n, T, Ts...>
{
    using type = typename std::conditional< Predicate<T>::value, T, typename find_if_impl<Predicate, n+1, Ts...>::type>::type;
    static constexpr unsigned value = Predicate<T>::value ? n : find_if_impl<Predicate, n+1, Ts...>::value;
};
}//namespace detail

//find first instance of a type that fulfills a predicate
template<template <typename> class Predicate, class T, class... Ts>
using find_if_t = typename detail::find_if_impl<Predicate,0, T, Ts...>::type;
//find the corresponding element's index in the parameter pack
template<template <typename> class Predicate, class T, class... Ts>
using find_if_v = std::integral_constant<unsigned, detail::find_if_impl<Predicate,0, T, Ts...>::value>;

/////////////////////////////////////////////////////////////////////////////////
//is scalar
template< class T>
using is_scalar = typename std::conditional< std::is_base_of<AnyScalarTag, get_tensor_category<T>>::value, std::true_type, std::false_type>::type;
//is vector (or scalar)
template< class T>
using is_vector = typename std::conditional< std::is_base_of<AnyVectorTag, get_tensor_category<T>>::value, std::true_type, std::false_type>::type;
//is matrix (or vector or scalar)
template< class T>
using is_matrix = typename std::conditional< std::is_base_of<AnyMatrixTag, get_tensor_category<T>>::value, std::true_type, std::false_type>::type;

namespace detail
{
template<class Category>
using find_base_category = typename
    std::conditional< std::is_base_of<SharedVectorTag, Category>::value, SharedVectorTag,
    typename std::conditional< std::is_base_of<VectorVectorTag, Category>::value, VectorVectorTag, MPIVectorTag>::type>::type;
}//namesapce detail
//is scalar or same tensor category
template<class T, class Category>
using is_scalar_or_same_base_category = typename std::conditional< std::is_base_of<detail::find_base_category<Category>, get_tensor_category<T>>::value || is_scalar<T>::value , std::true_type, std::false_type>::type;


//has trivial policy
template< class T>
using has_any_policy = typename std::conditional< std::is_same<AnyPolicyTag, get_execution_policy<T>>::value, std::true_type, std::false_type>::type;
//has any or same policy tag
template<class U, class Policy>
using has_any_or_same_policy = typename std::conditional< std::is_same<get_execution_policy<U>, Policy>::value || has_any_policy<U>::value, std::true_type, std::false_type>::type;
//is not scalar and has a nontrivial policy
template< class T>
using is_not_scalar_has_not_any_policy = typename std::conditional< !is_scalar<T>::value && !has_any_policy<T>::value, std::true_type, std::false_type>::type;

/////////////////////////////////////////////////////////////////////////////////
//from stackoverflow implement Columbo's bool pack trick to check parameter packs
template < bool...> struct bool_pack;

template<bool... v>
using all_true = std::is_same<bool_pack<true,v...>, bool_pack<v..., true>>;



}//namespace dg
