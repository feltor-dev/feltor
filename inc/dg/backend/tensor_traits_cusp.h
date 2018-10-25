#pragma once

#include <cassert>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include "vector_categories.h"
#include "matrix_categories.h"
#include "tensor_traits.h"

namespace dg
{
//makes such a long name in class list
///@cond
///@addtogroup dispatch
///@{
template<class T>
struct TensorTraits<cusp::array1d<T,cusp::host_memory>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using tensor_category   = CuspVectorTag;
    using execution_policy  = SerialTag;
};
template<class T>
struct TensorTraits<cusp::array1d<T,cusp::device_memory>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using tensor_category   = CuspVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ; //!< if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#else
    using execution_policy  = OmpTag ; //!< if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA

#endif
};
///@}
///@endcond
///@addtogroup dispatch
///@{

template< class I, class V, class M>
struct TensorTraits< cusp::coo_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::csr_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::dia_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::ell_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::hyb_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};

///@}

} //namespace dg
