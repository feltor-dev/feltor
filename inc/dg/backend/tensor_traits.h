#ifndef _DG_VECTOR_TRAITS_
#define _DG_VECTOR_TRAITS_

#include <type_traits>
#include <thrust/iterator/constant_iterator.h>
#include "scalar_categories.h"
#include "vector_categories.h"
#include "matrix_categories.h"
#include "execution_policy.h"

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
\see vec_list
@ingroup vec_list
*/
template< class Vector, class Enable=void>
struct TensorTraits;
//{
//    using value_type        = double; //!< The underlying data type
//    using tensor_category   = ThrustVectorTag; //!< Policy how data has to be accessed (has to derive from \c AnyVectorTag)
//    using execution_policy  = OmpTag;  //!< The execution policy (has to derive from \c AnyPolicyTag)
//};
template<class Vector>
using get_value_type = typename TensorTraits<typename std::decay<Vector>::type>::value_type;
template<class Vector>
using get_tensor_category = typename TensorTraits< typename std::decay<Vector>::type >::tensor_category;
template<class Vector>
using get_execution_policy = typename TensorTraits<typename std::decay<Vector>::type>::execution_policy;
template<class Vector>
using get_pointer_type = typename std::conditional< std::is_const<Vector>::value, const get_value_type<Vector>*, get_value_type<Vector>* >::type;
////////////get element, data and iterator
template<class T>
inline auto do_get_element( T&& v, unsigned i, AnyVectorTag)-> decltype(v[i]){
    return v[i];
}
template<class T>
inline T do_get_element( T&& v, unsigned i, AnyScalarTag){
    return v;
}
template<class T>
inline auto get_element( T&& v, unsigned i ) -> decltype( do_get_element( std::forward<T>(v), i, get_tensor_category<T>()) ) {
    return do_get_element( std::forward<T>(v), i, get_tensor_category<T>());
}

template<class T>
inline auto do_get_data( T&& v, AnyVectorTag)-> decltype(v.data()){
    return v.data();
}
template<class T>
inline T do_get_data( T&& v, AnyScalarTag){
    return v;
}
template<class T>
inline auto get_data( T&& v)-> decltype(do_get_data( std::forward<T>(v), get_tensor_category<T>() )){
    return do_get_data( std::forward<T>(v), get_tensor_category<T>());
}

template<class T>
auto get_iterator( T&& v, AnyVectorTag) -> decltype(v.begin()){
    return v.begin();
}
template<class T>
thrust::constant_iterator<T> get_iterator( T&& v, AnyScalarTag){
    return thrust::constant_iterator<T>(v);
}

template<class T>
auto get_iterator( T&& v ) -> decltype( get_iterator( std::forward<T>(v), get_tensor_category<T>())) {
    return get_iterator( std::forward<T>(v), get_tensor_category<T>());
}
///@}

}//namespace dg

#endif //_DG_VECTOR_TRAITS_
