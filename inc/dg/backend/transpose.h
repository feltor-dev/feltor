#pragma once

//some thoughts on how to transpose a MPI matrix 

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

template<class Matrix>
Matrix transpose( const Matrix& src)
{
    return detail::doTranspose( src, typename MatrixTraits<Matrix>::matrix_category());
}

} //namespace dg
