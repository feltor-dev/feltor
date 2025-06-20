#ifndef _DG_VECTOR_TRAITS_
#define _DG_VECTOR_TRAITS_

#include <type_traits>
#include <thrust/memory.h>
#include "scalar_categories.h"
#include "vector_categories.h"
#include "matrix_categories.h"
#include "execution_policy.h"

// Q: Make inclusion of tensor_traits easier if all you want is to use the full traits system in feltor?
// A: Some subtle issues:
// - Do you want the thrust header? (And thus depend on thrust library?)
// - Do you want all matrix traits as well?
// - If so, do you want the sparsematrix.h which incurs -lcusparse dependency
// In fact, it is possible to include tensor_traits.h as is and include the specialisation later when it is needed

namespace dg{
///@addtogroup dispatch
///@{


/*! @brief The vector traits

Specialize this struct if you want to enable your own vector/container class for the use in blas1 functions.
The contained types are
- <tt> value_type </tt> the elementary data type of the contained data
- <tt> tensor_category </tt> the layout of the data (derives from \c AnyMatrixTag)
- <tt> execution_policy </tt> for \c SharedVectorTag the execution policy
    (derives from \c AnyPolicyTag)
    indicates the type of hardware memory is physically
allocated on in a vector class and therefore indicates the
possible parallelization and optimization strategies.
\see \ref dispatch
*/
template< class Vector, class Enable=void>
struct TensorTraits
{
    using value_type = void;
    using tensor_category = NotATensorTag;
    using execution_policy = NoPolicyTag;
};

template<class Vector>
using get_value_type = typename TensorTraits<std::decay_t<Vector>>::value_type;
template<class Vector>
using get_tensor_category = typename TensorTraits< std::decay_t<Vector>>::tensor_category;
template<class Vector>
using get_execution_policy = typename TensorTraits<std::decay_t<Vector>>::execution_policy;

///@}

///@cond
////////////get element, pointer and data
template<class T> //T = SharedVector
using get_pointer_type = std::conditional_t< std::is_const< std::remove_reference_t<T> >::value,
    const get_value_type<T>*, get_value_type<T>* >;

template<class T> //T = RecursiveVector
using get_element_type = std::conditional_t< std::is_const< std::remove_reference_t<T> >::value,
    const typename std::decay_t<T>::value_type&, typename std::decay_t<T>::value_type& >;
template<class T> //T = std::map
using get_mapped_type = std::conditional_t< std::is_const< std::remove_reference_t<T> >::value,
    const typename std::decay_t<T>::mapped_type&, typename std::decay_t<T>::mapped_type& >;

template<class T>//T = MPIVector
using get_data_type = std::conditional_t< std::is_const< std::remove_reference_t<T> >::value,
    const typename std::decay_t<T>::container_type&, typename std::decay_t<T>::container_type& >;

template<class T>
inline get_element_type<T> do_get_vector_element( T&& v, unsigned i, RecursiveVectorTag)//-> decltype(v[i]){
{
    return v[i];
}
template<class T, class Key>
inline get_mapped_type<T> do_get_vector_element( T&& v, const Key& key, StdMapTag)//-> decltype(v[i]){
{
    return v.at(key);
}
template<class T, class Key>
inline T&& do_get_vector_element( T&& v, const Key&, AnyScalarTag){
    return std::forward<T>(v);
}

template<class T>
inline get_data_type<T> do_get_data( T&& v, MPIVectorTag)//-> decltype(v.data())
{
    return v.data();
}
template<class T>
inline T&& do_get_data( T&& v, AnyScalarTag){
    return std::forward<T>(v);
}

template<class T>
inline get_pointer_type<T> do_get_pointer_or_reference( T&& v, AnyVectorTag)// -> decltype(thrust::raw_pointer_cast(v.data())) //nvcc-7.5 does not like decltype in this connection
{
    return thrust::raw_pointer_cast(v.data());
}
template<class T>
inline T&& do_get_pointer_or_reference( T&& v, AnyScalarTag){
    return std::forward<T>(v);
}

///@endcond

//template<class T>
//inline std::conditional_t<std::is_base_of<AnyScalarTag, get_tensor_category<T>>::value, T&&, get_element_type<T> > get_vector_element( T&& v, unsigned i )// -> decltype( do_get_vector_element( std::forward<T>(v), i, get_tensor_category<T>()) )
//{
//    return do_get_vector_element( std::forward<T>(v), i, get_tensor_category<T>());
//}
//
//template<class T>
//inline std::conditional_t<std::is_base_of<AnyScalarTag, get_tensor_category<T>>::value, T, get_data_type<T> > get_data( T&& v)//-> decltype(do_get_data( std::forward<T>(v), get_tensor_category<T>() ))
//{
//    return do_get_data( std::forward<T>(v), get_tensor_category<T>());
//}
//
//template<class T>
//std::conditional<std::is_base_of<AnyScalarTag, get_tensor_category<T>>::value, T, get_pointer_type<T> > get_pointer_or_reference( T&& v )// -> decltype( do_get_pointer_or_reference( std::forward<T>(v), get_tensor_category<T>()))
//{
//    return do_get_pointer_or_reference( std::forward<T>(v), get_tensor_category<T>());
//}

}//namespace dg

#endif //_DG_VECTOR_TRAITS_
