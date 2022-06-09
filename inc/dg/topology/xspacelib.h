#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

//#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

#include <cusp/coo_matrix.h>
#include <cassert>

//
////functions for evaluation
#include "grid.h"
#include "dlt.h"
#include "operator.h"
#include "operator_tensor.h"
#include "interpolation.h" //makes typedefs available


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
* Takes care of correct permutation of indices.
* @tparam T value type
* @param lhs The left hand side matrix (duplicate entries lead to duplicate entries in result)
* @param rhs The right hand side matrix (duplicate entries lead to duplicate entries in result)
*
* @return newly allocated cusp matrix containing the tensor product
* @note use \c cusp::add and \c cusp::multiply to add and multiply matrices
*/
template< class T>
cusp::coo_matrix< int, T, cusp::host_memory> tensorproduct(
        const cusp::coo_matrix< int, T, cusp::host_memory>& lhs,
        const cusp::coo_matrix< int, T, cusp::host_memory>& rhs)
{
    //dimensions of the matrix
    int num_rows     = lhs.num_rows*rhs.num_rows;
    int num_cols     = lhs.num_cols*rhs.num_cols;
    // allocate storage for unordered triplets
    cusp::array1d< int,     cusp::host_memory> I; // row indices
    cusp::array1d< int,     cusp::host_memory> J; // column indices
    cusp::array1d< T,  cusp::host_memory> V; // values
    //LHS x RHS
    for( unsigned i=0; i<lhs.num_entries; i++)
        for( unsigned j=0; j<rhs.num_entries; j++)
        {
            I.push_back( lhs.row_indices[i]*rhs.num_rows + rhs.row_indices[j]);
            J.push_back( lhs.column_indices[i]*rhs.num_cols +  rhs.column_indices[j]);
            V.push_back( lhs.values[i]*rhs.values[j]);
        }
    // sort triplets by (I,J) index using two stable sorts (first by J, then by I)
    thrust::stable_sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
    thrust::stable_sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));

    // allocate output matrix
    int num_entries = lhs.num_entries* rhs.num_entries;
    cusp::coo_matrix<int, T, cusp::host_memory> A(num_rows, num_cols, num_entries);
    A.row_indices = I;
    A.column_indices = J;
    A.values = V;
    return A;
}

namespace create{
///@addtogroup scatter
///@{

/**
 * @brief make a matrix that transforms values to an equidistant grid ready for visualisation
 *
 * Useful if you want to visualize a dg-formatted vector.
 * @param g The grid on which to operate
 *
 * @return transformation matrix
 * @note this matrix has ~n^4 N^2 entries
 */
template<class real_type>
dg::IHMatrix backscatter( const aRealTopology2d<real_type>& g)
{
    typedef cusp::coo_matrix<int, real_type, cusp::host_memory> Matrix;
    //create equidistant backward transformation
    dg::Operator<real_type> backwardeqX( g.dltx().backwardEQ());
    dg::Operator<real_type> backwardeqY( g.dlty().backwardEQ());
    dg::Operator<real_type> forwardX( g.dltx().forward());
    dg::Operator<real_type> forwardY( g.dlty().forward());
    dg::Operator<real_type> backward1dX = backwardeqX*forwardX;
    dg::Operator<real_type> backward1dY = backwardeqY*forwardY;

    Matrix transformX = dg::tensorproduct( g.Nx(), backward1dX);
    Matrix transformY = dg::tensorproduct( g.Ny(), backward1dY);
    Matrix backward = dg::tensorproduct( transformY, transformX);

    return (dg::IHMatrix)backward;

}
///@copydoc backscatter(const aRealTopology2d&)
template<class real_type>
dg::IHMatrix backscatter( const RealGrid1d<real_type>& g)
{
    typedef cusp::coo_matrix<int, real_type, cusp::host_memory> Matrix;
    //create equidistant backward transformation
    dg::Operator<real_type> backwardeq( g.dlt().backwardEQ());
    dg::Operator<real_type> forward( g.dlt().forward());
    dg::Operator<real_type> backward1d = backwardeq*forward;

    Matrix backward = dg::tensorproduct( g.N(), backward1d);
    return (dg::IHMatrix)backward;

}

///@copydoc backscatter(const aRealTopology2d&)
template<class real_type>
dg::IHMatrix backscatter( const aRealTopology3d<real_type>& g)
{
    Grid2d g2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy());
    cusp::coo_matrix<int,real_type, cusp::host_memory> back2d = backscatter( g2d);
    return (dg::IHMatrix)tensorproduct<real_type>( tensorproduct<real_type>( g.Nz(), delta<real_type>(1)), back2d);
}
///@}

///@cond
/**
 * @brief Index map for scatter operation on dg - formatted vectors
 *
 * Use in thrust::scatter function on a dg-formatted vector. We obtain a vector
 where the y direction is contiguous in memory.
 * @param n # of polynomial coefficients
 * @param Nx # of points in x
 * @param Ny # of points in y
 *
 * @return map of indices
 */
static inline thrust::host_vector<int> scatterMapInvertxy( unsigned n, unsigned Nx, unsigned Ny)
{
    unsigned Nx_ = n*Nx, Ny_ = n*Ny;
    //thrust::host_vector<int> reorder = scatterMap( n, Nx, Ny);
    thrust::host_vector<int> map( n*n*Nx*Ny);
    thrust::host_vector<int> map2( map);
    for( unsigned i=0; i<map.size(); i++)
    {
        int row = i/Nx_;
        int col = i%Nx_;

        map[i] =  col*Ny_+row;
    }
    //for( unsigned i=0; i<map.size(); i++)
        //map2[i] = map[reorder[i]];
    //return map2;
    return map;
}

/**
 * @brief write a matrix containing it's line number as elements
 *
 * Useful in a reduce_by_key computation
 * @param rows # of rows of the matrix
 * @param cols # of cols of the matrix
 *
 * @return a vector of size rows*cols containing line numbers
 */
static inline thrust::host_vector<int> contiguousLineNumbers( unsigned rows, unsigned cols)
{
    thrust::host_vector<int> map( rows*cols);
    for( unsigned i=0; i<map.size(); i++)
    {
        map[i] = i/cols;
    }
    return map;
}
///@endcond


} //namespace create
}//namespace dg
#endif // _DG_XSPACELIB_CUH_
