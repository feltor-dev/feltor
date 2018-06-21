#pragma once

#include <type_traits>
#include "scalar_categories.h"
#include "tensor_traits.h"

namespace dg{

template<template <typename> class Predicate, class... Ts>
struct find_first_impl;

template<template <typename> class Predicate, class T>
struct find_first_impl<Predicate, T>
{
    using type = typename std::conditional< Predicate<T>::value, T, std::false_type>::type;
};

template<template <typename> class Predicate, class T, class... Ts>
struct find_first_impl<Predicate, T, Ts...>
{
    using type = typename std::conditional< Predicate<T>::value, T, typename find_first_impl<Predicate, Ts...>::type>::type;
};

template<template <typename> class Predicate, class T, class... Ts>
using find_first = typename find_first_impl<Predicate, T, Ts...>::type;

//is scalar
template<class T, class Enable = void>
struct is_scalar: std::false_type{};
template<class T>
struct is_scalar<T, typename std::enable_if< std::is_base_of<AnyScalarTag, get_tensor_category<T>>::value >::type > : std::true_type{};

template< class T>
using is_not_scalar = typename std::conditional< !is_scalar<T>::value, std::true_type, std::false_type>::type;
//is vector
template<class T, class Enable = void>
struct is_vector: std::false_type{};
template<class T>
struct is_vector<T, typename std::enable_if< std::is_base_of<AnyVectorTag, get_tensor_category<T>>::value >::type > : std::true_type{};
//is matrix
template<class T, class Enable = void>
struct is_matrix: std::false_type{};
template<class T>
struct is_matrix<T, typename std::enable_if< std::is_base_of<AnyMatrixTag, get_tensor_category<T>>::value >::type > : std::true_type{};





}//namespace dg
