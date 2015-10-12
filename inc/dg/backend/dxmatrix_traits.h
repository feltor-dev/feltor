#ifndef _DG_DXMATRIX_TRAITS
#define _DG_DXMATRIX_TRAITS


#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "matrix_traits.h"
#include "matrix_categories.h"


//exclude from doxygen Docu
///@cond

namespace dg{

    template <typename T>
    struct MatrixTraits<thrust::host_vector<T> > {
        typedef T value_type;
        typedef dx_matrixTag matrix_category;
    };

    template <typename T>
    struct MatrixTraits<thrust::device_vector<T> > {
        typedef T value_type;
        typedef dx_matrixTag matrix_category;
    };

    template <typename T>
    struct MatrixTraits<const thrust::host_vector<T> > {
        typedef T value_type;
        typedef dx_matrixTag matrix_category;
    };

    template <typename T>
    struct MatrixTraits<const thrust::device_vector<T> > {
        typedef T value_type;
        typedef dx_matrixTag matrix_category;
    };

} //namespace dg
///@endcond

#endif //_DG_DXMATRIX_TRAITS

