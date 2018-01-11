#ifndef _DG_MATRIX_TRAITS_THRUST
#define _DG_MATRIX_TRAITS_THRUST


#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "matrix_traits.h"
#include "matrix_categories.h"

namespace dg{

template<class T>
struct MatrixTraits<std::vector<T> >{
    using value_type = T;
    using matrix_category = ThrustMatrixTag;
};
template<class T>
struct MatrixTraits<const std::vector<T> >{
    using value_type = T;
    using matrix_category = ThrustMatrixTag;
};

template< class T>
struct MatrixTraits<thrust::host_vector<T> > {
    using value_type = T;
    using matrix_category = ThrustMatrixTag;
};
template< class T>
struct MatrixTraits<thrust::device_vector<T> > {
    using value_type = T;
    using matrix_category = ThrustMatrixTag;
};
template< class T>
struct MatrixTraits<const thrust::host_vector<T> > {
    using value_type = T;
    using matrix_category = ThrustMatrixTag;
};
template< class T>
struct MatrixTraits<const thrust::device_vector<T> > {
    using value_type = T;
    using matrix_category = ThrustMatrixTag;
};


}//namespace dg

#endif//_DG_MATRIX_TRAITS_THRUST
