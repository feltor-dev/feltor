#pragma once

//some thoughts on how to transpose a MPI matrix 

///@cond
namespace dg
{

template <class Matrix>
Matrix doTranspose( const Matrix& src, CuspMatrixTag)
{
    Matrix out(src);
    cusp::transpose( src, out);
    return out;
}
template <class Matrix, class Collective>
ColDistMat doTranspose( const RowDistMat<Matrix, Collective>& src, MPIMatrixTag)
{
    Matrix tr( src.matrix());
    cusp::transpose( src.matrix(), tr);
    ColDistMat out( tr, src.collective());
    return out;
}


template<class Matrix>
Matrix transpose( const Matrix& src)
{
    out = doTranspose( src, typename MatrixTraits<Matrix>::matrix_category());
}

} //namespace dg
///@endcond
