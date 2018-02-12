#ifndef _DG_MATRIX_TRAITS_THRUST
#define _DG_MATRIX_TRAITS_THRUST


#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "matrix_traits.h"
#include "matrix_categories.h"

#include "weights.cuh"

namespace dg{

///@cond
template< class T>
struct MatrixTraits<thrust::host_vector<T> > {
    typedef T value_type;
    typedef ThrustMatrixTag matrix_category;
};
template< class T>
struct MatrixTraits<thrust::device_vector<T> > {
    typedef T value_type;
    typedef ThrustMatrixTag matrix_category;
};
template< class T>
struct MatrixTraits<const thrust::host_vector<T> > {
    typedef T value_type;
    typedef ThrustMatrixTag matrix_category;
};
template< class T>
struct MatrixTraits<const thrust::device_vector<T> > {
    typedef T value_type;
    typedef ThrustMatrixTag matrix_category;
};
///@endcond


}//namespace dg

#endif//_DG_MATRIX_TRAITS_THRUST
