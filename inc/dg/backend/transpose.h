#pragma once
#include <cusp/transpose.h>
#ifdef MPI_VERSION
#include "mpi_matrix.h"
#endif //MPI_VERSION

namespace dg
{

///@cond
namespace detail
{
template <class Matrix>
Matrix doTranspose( const Matrix& src, CuspMatrixTag)
{
    Matrix out(src);
    cusp::transpose( src, out);
    return out;
}
#ifdef MPI_VERSION
template <class Matrix, class Collective>
MPIDistMat<Matrix, Collective> doTranspose( const MPIDistMat<Matrix, Collective>& src, MPIMatrixTag)
{
    Matrix tr = doTranspose( src.matrix(), typename MatrixTraits<Matrix>::matrix_category());
    MPIDistMat<Matrix, Collective> out( tr, src.collective());
    return out;
}
#endif// MPI_VERSION
}//namespace detail
///@endcond

/**
 * @brief Generic matrix transpose method
 *
 * @tparam Matrix one of 
 *  - any cusp matrix
 *  - any MPIDistMatrix with a cusp matrix as template parameter
 * @param src the marix to transpose
 *
 * @return the matrix that acts as the transpose of src
 * @ingroup lowlevel 
 */
template<class Matrix>
Matrix transpose( const Matrix& src)
{
    return detail::doTranspose( src, typename MatrixTraits<Matrix>::matrix_category());
}

} //namespace dg
