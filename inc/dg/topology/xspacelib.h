#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

#include <thrust/host_vector.h>

#include <cassert>

#include "dg/backend/typedefs.h"
#include "grid.h"
#include "dlt.h"
#include "operator.h"
#include "operator_tensor.h"


/*! @file

  * @brief utility functions
  */

namespace dg{

/**
* @brief \f$ L\otimes R\f$ Form the tensor (Kronecker) product between two matrices
*
* The Kronecker product is formed by the triplets
* \f$I = k n_r+l\f$, \f$ J = i N_r +j \f$,  \f$ M_{IJ} = L_{ki}R_{lj}\f$
* @ingroup lowlevel
* @note This function is "order preserving" in the sense that the order of row
* and column entries of lhs and rhs are preserved in the output. This is
* important for stencil computations.
* @tparam T value type
* @param lhs The left hand side matrix (duplicate entries lead to duplicate entries in result)
* @param rhs The right hand side matrix (duplicate entries lead to duplicate entries in result)
*
* @return newly allocated matrix containing the tensor product
*/
template< class T>
dg::SparseMatrix< int, T, thrust::host_vector> tensorproduct(
        const dg::SparseMatrix< int, T, thrust::host_vector>& lhs,
        const dg::SparseMatrix< int, T, thrust::host_vector>& rhs)
{
    //dimensions of the matrix
    size_t num_rows     = lhs.num_rows()*rhs.num_rows();
    size_t num_cols     = lhs.num_cols()*rhs.num_cols();
    size_t num_entries  = lhs.num_entries()* rhs.num_entries();
    // allocate output matrix
    thrust::host_vector<int> A_row_offsets(num_rows+1), A_column_indices( num_entries);
    thrust::host_vector<T> A_values( num_entries);
    //LHS x RHS
    A_row_offsets[0] = 0;
    int counter = 0;
    for( unsigned i=0; i<lhs.num_rows(); i++)
    for( unsigned j=0; j<rhs.num_rows(); j++)
    {
        int num_entries_in_row =
            (lhs.row_offsets()[i+1] - lhs.row_offsets()[i])*
            (rhs.row_offsets()[j+1] - rhs.row_offsets()[j]);
        A_row_offsets[i*rhs.num_rows()+j+1] =
            A_row_offsets[i*rhs.num_rows()+j] + num_entries_in_row;
        for( int k=lhs.row_offsets()[i]; k<lhs.row_offsets()[i+1]; k++)
        for( int l=rhs.row_offsets()[j]; l<rhs.row_offsets()[j+1]; l++)
        {
            A_column_indices[counter] =
                lhs.column_indices()[k]*rhs.num_cols() +  rhs.column_indices()[l];
            A_values[counter]  = lhs.values()[k]*rhs.values()[l];
            counter++;
        }
    }
    // allocate output matrix
    return { num_rows, num_cols, A_row_offsets, A_column_indices, A_values};
}

/**
* @brief \f$ L\otimes R\f$ Form the tensor (Kronecker) product between two matrices in the column index
*
* The Kronecker product in the columns is formed by the triplets
* \f$ J = i N_r +j \f$,  \f$ M_{kJ} = L_{ki}R_{kj}\f$
* @ingroup lowlevel
* @note This function is "order preserving" in the sense that the order of row
* and column entries of lhs and rhs are preserved in the output. This is
* important for stencil computations.
* @tparam T value type
* @param lhs The left hand side matrix (duplicate entries lead to duplicate entries in result)
* @param rhs The right hand side matrix (duplicate entries lead to duplicate entries in result)
*
* @return newly allocated sparse matrix containing the tensor product
*/
template< class T>
dg::SparseMatrix< int, T, thrust::host_vector> tensorproduct_cols(
        const dg::SparseMatrix< int, T, thrust::host_vector>& lhs,
        const dg::SparseMatrix< int, T, thrust::host_vector>& rhs)
{
    if( lhs.num_rows() != rhs.num_rows())
        throw Error( Message(_ping_)<<"lhs and rhs must have same number of rows: "<<lhs.num_rows()<<" rhs "<<rhs.num_rows());

    //dimensions of the matrix
    size_t num_rows     = lhs.num_rows();
    size_t num_cols     = lhs.num_cols()*rhs.num_cols();
    size_t num_entries = 0;
    for( unsigned i=0; i<lhs.num_rows(); i++)
    {
        int num_entries_in_row =
            (lhs.row_offsets()[i+1] - lhs.row_offsets()[i])*
            (rhs.row_offsets()[i+1] - rhs.row_offsets()[i]);
        num_entries += num_entries_in_row;
    }
    // allocate output matrix
    thrust::host_vector<int> A_row_offsets(num_rows+1), A_column_indices( num_entries);
    thrust::host_vector<T> A_values( num_entries);
    //LHS x RHS
    A_row_offsets[0] = 0;
    int counter = 0;
    for( unsigned i=0; i<lhs.num_rows(); i++)
    {
        int num_entries_in_row =
            (lhs.row_offsets()[i+1] - lhs.row_offsets()[i])*
            (rhs.row_offsets()[i+1] - rhs.row_offsets()[i]);
        A_row_offsets[i+1] = A_row_offsets[i] + num_entries_in_row;
        for( int k=lhs.row_offsets()[i]; k<lhs.row_offsets()[i+1]; k++)
        for( int l=rhs.row_offsets()[i]; l<rhs.row_offsets()[i+1]; l++)
        {
            A_column_indices[counter] =
                lhs.column_indices()[k]*rhs.num_cols() +  rhs.column_indices()[l];
            A_values[counter]  = lhs.values()[k]*rhs.values()[l];
            counter++;
        }
    }
    return { num_rows, num_cols, A_row_offsets, A_column_indices, A_values};
}


namespace create{
///@addtogroup scatter
///@{

/**
 * @brief Create a matrix \f$ B_{eq} F\f$ that interpolates values to an equidistant grid ready for visualisation
 *
 * Useful if you want to visualize a dg-formatted vector.
 * @param g The grid on which to operate
 *
 * @return transformation matrix (block diagonal)
 * @sa dg::create::backproject
 */
template<class real_type>
dg::IHMatrix_t<real_type> backscatter( const RealGrid1d<real_type>& g)
{
    //create equidistant backward transformation
    dg::SquareMatrix<real_type> backwardeq( dg::DLT<real_type>::backwardEQ(g.n()));
    dg::SquareMatrix<real_type> forward( dg::DLT<real_type>::forward(g.n()));
    dg::SquareMatrix<real_type> backward1d = backwardeq*forward;

    return (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(), backward1d);
}

///@copydoc backscatter(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> backscatter( const aRealTopology2d<real_type>& g)
{
    //create equidistant backward transformation
    auto transformX = backscatter( g.gx());
    auto transformY = backscatter( g.gy());
    return dg::tensorproduct( transformY, transformX);
}

///@copydoc backscatter(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> backscatter( const aRealTopology3d<real_type>& g)
{
    auto transformX = backscatter( g.gx());
    auto transformY = backscatter( g.gy());
    auto transformZ = backscatter( g.gz());
    return dg::tensorproduct( transformZ, dg::tensorproduct(transformY, transformX));
}

/**
 * @brief Create a matrix \f$ (B_{eq} F)^{-1}\f$ that transforms values from an equidistant grid back to a dg grid
 *
 * The inverse of \c dg::create::backscatter
 * @note The inverse of the backscatter matrix is **not** its adjoint! The adjoint \f$ (B_{eq}F)^\dagger\f$
 * is the matrix that computes the (inexact) projection integral on an equidistant grid.
 * @param g The grid on which to operate
 *
 * @return transformation matrix (block diagonal)
 * @sa dg::create::inv_backproject dg::create::backscatter
 */
template<class real_type>
dg::IHMatrix_t<real_type> inv_backscatter( const RealGrid1d<real_type>& g)
{
    //create equidistant backward transformation
    dg::SquareMatrix<real_type> backwardeq( dg::DLT<real_type>::backwardEQ(g.n()));
    dg::SquareMatrix<real_type> backward( dg::DLT<real_type>::backward(g.n()));
    dg::SquareMatrix<real_type> forward1d = backward*dg::invert(backwardeq);

    return (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(), forward1d);
}
///@copydoc inv_backscatter(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> inv_backscatter( const aRealTopology2d<real_type>& g)
{
    //create equidistant backward transformation
    auto transformX = inv_backscatter( g.gx());
    auto transformY = inv_backscatter( g.gy());
    return dg::tensorproduct( transformY, transformX);
}

///@copydoc inv_backscatter(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> inv_backscatter( const aRealTopology3d<real_type>& g)
{
    auto transformX = inv_backscatter( g.gx());
    auto transformY = inv_backscatter( g.gy());
    auto transformZ = inv_backscatter( g.gz());
    return dg::tensorproduct( transformZ, dg::tensorproduct(transformY, transformX));
}

///@}

} //namespace create
} //namespace dg
#endif // _DG_XSPACELIB_CUH_
