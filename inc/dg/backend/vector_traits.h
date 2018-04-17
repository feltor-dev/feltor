#ifndef _DG_VECTOR_TRAITS_
#define _DG_VECTOR_TRAITS_

#include <vector>
#include <type_traits>
#include "vector_categories.h"
#include "execution_policy.h"

namespace dg{
///@addtogroup dispatch
///@{
/*! @brief The vector traits

Specialize this struct if you want to enable your own vector/container class for the use in blas1 functions
@ingroup vec_list
*/
template< class Vector, class Enable=void>
template<class Vector>
using get_value_type = typename VectorTraits<Vector>::value_type;
template<class Vector>
using get_vector_category = typename VectorTraits<Vector>::vector_category;
template<class Vector>
using get_execution_policy = typename VectorTraits<Vector>::execution_policy;
//using is the new typedef in C++11
template<class Vector>
using get_pointer_type = typename std::conditional< std::is_const<Vector>::value, const get_value_type<Vector>*, get_value_type<Vector>* >::type;
///@}

///@addtogroup vec_list
///@{
template<class T>
struct VectorTraits<std::vector<T>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
template<class T>
struct VectorTraits<std::vector<T>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
///@}

///@cond
template<class Tag>
struct ThrustTag { };
template <>
struct ThrustTag<SerialTag>
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template <>
struct ThrustTag<CudaTag>
#else
template <>
struct ThrustTag<OmpTag>
#endif
template<class Vector>
using get_thrust_tag = typename ThrustTag<get_execution_policy<Vector>>::thrust_tag;
///@endcond

}//namespace dg

#endif //_DG_VECTOR_TRAITS_
