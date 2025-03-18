#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

#include <thrust/host_vector.h>

#include <cusp/coo_matrix.h>
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
* @return newly allocated cusp matrix containing the tensor product
* @note use \c cusp::add and \c cusp::multiply to add and multiply matrices.
*/
template< class T>
cusp::csr_matrix< int, T, cusp::host_memory> tensorproduct(
        const cusp::csr_matrix< int, T, cusp::host_memory>& lhs,
        const cusp::csr_matrix< int, T, cusp::host_memory>& rhs)
{
    //dimensions of the matrix
    int num_rows     = lhs.num_rows*rhs.num_rows;
    int num_cols     = lhs.num_cols*rhs.num_cols;
    int num_entries  = lhs.num_entries* rhs.num_entries;
    // allocate output matrix
    cusp::csr_matrix<int, T, cusp::host_memory> A(num_rows, num_cols, num_entries);
    //LHS x RHS
    A.row_offsets[0] = 0;
    int counter = 0;
    for( unsigned i=0; i<lhs.num_rows; i++)
    for( unsigned j=0; j<rhs.num_rows; j++)
    {
        int num_entries_in_row =
            (lhs.row_offsets[i+1] - lhs.row_offsets[i])*
            (rhs.row_offsets[j+1] - rhs.row_offsets[j]);
        A.row_offsets[i*rhs.num_rows+j+1] =
            A.row_offsets[i*rhs.num_rows+j] + num_entries_in_row;
        for( int k=lhs.row_offsets[i]; k<lhs.row_offsets[i+1]; k++)
        for( int l=rhs.row_offsets[j]; l<rhs.row_offsets[j+1]; l++)
        {
            A.column_indices[counter] =
                lhs.column_indices[k]*rhs.num_cols +  rhs.column_indices[l];
            A.values[counter]  = lhs.values[k]*rhs.values[l];
            counter++;
        }
    }
    return A;
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
* @return newly allocated cusp matrix containing the tensor product
* @note use \c cusp::add and \c cusp::multiply to add and multiply matrices.
*/
template< class T>
cusp::csr_matrix< int, T, cusp::host_memory> tensorproduct_cols(
        const cusp::csr_matrix< int, T, cusp::host_memory>& lhs,
        const cusp::csr_matrix< int, T, cusp::host_memory>& rhs)
{
    if( lhs.num_rows != rhs.num_rows)
        throw Error( Message(_ping_)<<"lhs and rhs must have same number of rows: "<<lhs.num_rows<<" rhs "<<rhs.num_rows);

    //dimensions of the matrix
    int num_rows     = lhs.num_rows;
    int num_cols     = lhs.num_cols*rhs.num_cols;
    int num_entries = 0;
    for( unsigned i=0; i<lhs.num_rows; i++)
    {
        int num_entries_in_row =
            (lhs.row_offsets[i+1] - lhs.row_offsets[i])*
            (rhs.row_offsets[i+1] - rhs.row_offsets[i]);
        num_entries += num_entries_in_row;
    }
    // allocate output matrix
    cusp::csr_matrix<int, T, cusp::host_memory> A(num_rows, num_cols, num_entries);
    //LHS x RHS
    A.row_offsets[0] = 0;
    int counter = 0;
    for( unsigned i=0; i<lhs.num_rows; i++)
    {
        int num_entries_in_row =
            (lhs.row_offsets[i+1] - lhs.row_offsets[i])*
            (rhs.row_offsets[i+1] - rhs.row_offsets[i]);
        A.row_offsets[i+1] = A.row_offsets[i] + num_entries_in_row;
        for( int k=lhs.row_offsets[i]; k<lhs.row_offsets[i+1]; k++)
        for( int l=rhs.row_offsets[i]; l<rhs.row_offsets[i+1]; l++)
        {
            A.column_indices[counter] =
                lhs.column_indices[k]*rhs.num_cols +  rhs.column_indices[l];
            A.values[counter]  = lhs.values[k]*rhs.values[l];
            counter++;
        }
    }
    return A;
}
///@cond
template< class T>
cusp::coo_matrix< int, T, cusp::host_memory> tensorproduct(
        const cusp::coo_matrix< int, T, cusp::host_memory>& lhs,
        const cusp::coo_matrix< int, T, cusp::host_memory>& rhs)
{
    //dimensions of the matrix
    int num_rows     = lhs.num_rows*rhs.num_rows;
    int num_cols     = lhs.num_cols*rhs.num_cols;
    int num_entries  = lhs.num_entries* rhs.num_entries;
    // allocate output matrix
    cusp::coo_matrix<int, T, cusp::host_memory> A(num_rows, num_cols, num_entries);
    //LHS x RHS
    int counter = 0;
    for( int k=0; k<lhs.num_entries; k++)
    for( int l=0; l<rhs.num_entries; l++)
    {
        A.row_indices[counter] =
            lhs.row_indices[k]*rhs.num_rows + rhs.row_indices[l];
        A.column_indices[counter] =
            lhs.column_indices[k]*rhs.num_cols +  rhs.column_indices[l];
        A.values[counter]  = lhs.values[k]*rhs.values[l];
        counter++;
    }
    return A;
}
// tensorproduct_cols does not work for coo_matrix without converting to csr_matrix ...
///@endcond


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
