#ifndef _DG_VECTOR_TRAITS_
#define _DG_VECTOR_TRAITS_

#include <vector>
#include <type_traits>
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
struct TypeTraits;
//{
//    using value_type        = double; //!< The underlying data type
//    using tensor_category   = ThrustVectorTag; //!< Policy how data has to be accessed (has to derive from \c AnyVectorTag)
//    using execution_policy  = OmpTag;  //!< The execution policy (has to derive from \c AnyPolicyTag)
//};
template<class Vector>
using get_value_type = typename TypeTraits<typename std::decay<Vector>::type>::value_type;
template<class Vector>
using get_tensor_category = typename TypeTraits< typename std::decay<Vector>::type >::tensor_category;
template<class Vector>
using get_execution_policy = typename TypeTraits<typename std::decay<Vector>::type>::execution_policy;
//using is the new typedef in C++11
template<class Vector>
using get_pointer_type = typename std::conditional< std::is_const<Vector>::value, const get_value_type<Vector>*, get_value_type<Vector>* >::type;
///@}

///@cond

//from stackoverflow implement Columbo's bool pack trick to check parameter packs
template < bool...> struct bool_pack;

template<bool... v>
using all_true = std::is_same<bool_pack<true,v...>, bool_pack<v..., true>>;

///@endcond

}//namespace dg

#endif //_DG_VECTOR_TRAITS_
