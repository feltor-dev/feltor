#pragma once
#include <cusp/transpose.h>
#include "blas2_cusp.h"
#include "tensor_traits.h"
#ifdef MPI_VERSION
#include "blas2_dispatch_mpi.h"
#endif //MPI_VERSION

namespace dg
{

///@cond
namespace detail
{
template <class Matrix>
Matrix doTranspose( const Matrix& src, CuspMatrixTag)
{
    Matrix out;
    cusp::transpose( src, out);
    return out;
}
}//namespace detail
///@endcond

/**
 * @brief Generic matrix transpose method
 *
 * @tparam Matrix one of
 *  - any cusp matrix
 * @param src the marix to transpose
 *
 * @return the matrix that acts as the transpose of src
 * @ingroup lowlevel
 */
template<class Matrix>
Matrix transpose( const Matrix& src)
{
    //%Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix!!!
    return detail::doTranspose( src, get_tensor_category<Matrix>());
}

} //namespace dg
