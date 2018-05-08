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
\see vec_list
@ingroup vec_list
*/
template< class Vector, class Enable=void>
struct VectorTraits {
    using value_type        = double; //!< The underlying data type
    using vector_category   = ThrustVectorTag; //!< Policy how data has to be accessed (has to derive from \c AnyVectorTag)
    using execution_policy  = OmpTag;  //!< The execution policy (has to derive from \c AnyPolicyTag)
};
template<class Vector>
using get_value_type = typename VectorTraits<typename std::decay<Vector>::type>::value_type;
template<class Vector>
using get_vector_category = typename VectorTraits< typename std::decay<Vector>::type >::vector_category;
template<class Vector>
using get_execution_policy = typename VectorTraits<typename std::decay<Vector>::type>::execution_policy;
//using is the new typedef in C++11
template<class Vector>
using get_pointer_type = typename std::conditional< std::is_const<Vector>::value, const get_value_type<Vector>*, get_value_type<Vector>* >::type;
///@}

///@addtogroup vec_list
///@{
template<class T>
struct VectorTraits<std::vector<T>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = ThrustVectorTag;
    using execution_policy  = OmpTag;
};
template<class T>
struct VectorTraits<std::vector<T>,
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using vector_category   = VectorVectorTag;
    using execution_policy  = get_execution_policy<T>;
};
///@}

///@cond
template<class Tag>
struct ThrustTag { };
template <>
struct ThrustTag<SerialTag>
{
    using thrust_tag = thrust::cpp::tag;
};
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template <>
struct ThrustTag<CudaTag>
{
    using thrust_tag = thrust::cuda::tag;
};
#else
template <>
struct ThrustTag<OmpTag>
{
    using thrust_tag = thrust::omp::tag;
};
#endif
template<class Vector>
using get_thrust_tag = typename ThrustTag<get_execution_policy<Vector>>::thrust_tag;

//from stackoverflow implement Columbo's bool pack trick to check parameter packs
template < bool...> struct bool_pack;

template<bool... v>
using all_true = std::is_same<bool_pack<true,v...>, bool_pack<v..., true>>;

///@endcond

}//namespace dg

#endif //_DG_VECTOR_TRAITS_
