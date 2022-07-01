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

namespace create{
///@addtogroup scatter
///@{

/**
 * @brief Create a matrix that transforms values to an equidistant grid ready for visualisation
 *
 * Useful if you want to visualize a dg-formatted vector.
 * @param g The grid on which to operate
 *
 * @return transformation matrix (block diagonal)
 * @sa dg::blas2::symv
 */
template<class real_type>
dg::IHMatrix_t<real_type> backscatter( const RealGrid1d<real_type>& g)
{
    //create equidistant backward transformation
    dg::Operator<real_type> backwardeq( g.dlt().backwardEQ());
    dg::Operator<real_type> forward( g.dlt().forward());
    dg::Operator<real_type> backward1d = backwardeq*forward;

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
 * @brief Create a matrix that transforms values from an equidistant grid back to a dg grid
 *
 * The inverse of \c dg::create::backscatter
 * @param g The grid on which to operate
 *
 * @return transformation matrix (block diagonal)
 * @sa dg::blas2::symv dg::create::backscatter
 */
template<class real_type>
dg::IHMatrix_t<real_type> inv_backscatter( const RealGrid1d<real_type>& g)
{
    //create equidistant backward transformation
    dg::Operator<real_type> backwardeq( g.dlt().backwardEQ());
    dg::Operator<real_type> backward( g.dlt().backward());
    dg::Operator<real_type> forward1d = backward*dg::invert(backwardeq);

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
