#pragma once

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
