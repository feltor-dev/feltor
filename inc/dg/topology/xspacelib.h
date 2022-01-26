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
///@cond
namespace detail{

struct AddIndex2d{
    AddIndex2d( size_t M ):M(M), number(0) {}

    void operator() ( cusp::array1d< int, cusp::host_memory>& Idx,
                      unsigned i, unsigned k)
    {
        Idx[ number] = M*i + k ;
        number ++;
    }
    template<class T>
    void operator() ( cusp::array1d< T, cusp::host_memory>& Val, T value)
    {
        Val[ number] = value;
        number ++;
    }
  private:
    size_t M;
    unsigned number;
};

} //namespace detail
///@endcond
/**
* @brief \f$ L\otimes R\f$ Form the tensor (Kronecker) product between two matrices
*
* @ingroup lowlevel
* Takes care of correct permutation of indices.
* @tparam T value type
* @param lhs The left hand side (1D )
* @param rhs The right hand side (1D )
*
* @return A newly allocated cusp matrix containing the tensor product
* @note use \c cusp::add and \c cusp::multiply to add and multiply matrices
*/
template< class T>
cusp::coo_matrix< int, T, cusp::host_memory> tensorproduct(
        const cusp::coo_matrix< int, T, cusp::host_memory>& lhs,
        const cusp::coo_matrix< int, T, cusp::host_memory>& rhs)
{
    //assert quadratic matrices
    assert( lhs.num_rows == lhs.num_cols);
    assert( rhs.num_rows == rhs.num_cols);
    //assert dg matrices
    unsigned Nx = rhs.num_rows;
    //taken from the cusp examples:
    //dimensions of the matrix
    int num_rows     = lhs.num_rows*rhs.num_rows;
    int num_cols     = num_rows;
    int num_triplets = lhs.num_entries*rhs.num_entries;
    // allocate storage for unordered triplets
    cusp::array1d< int,     cusp::host_memory> I( num_triplets); // row indices
    cusp::array1d< int,     cusp::host_memory> J( num_triplets); // column indices
    cusp::array1d< T,  cusp::host_memory> V( num_triplets); // values
    //fill triplet arrays
    detail::AddIndex2d addIndexRow( Nx);
    detail::AddIndex2d addIndexCol( Nx);
    detail::AddIndex2d addIndexVal( Nx);
    //LHS x RHS
    for( unsigned i=0; i<lhs.num_entries; i++)
        for( unsigned j=0; j<rhs.num_entries; j++)
        {
            addIndexRow( I, lhs.row_indices[i], rhs.row_indices[j]);
            addIndexCol( J, lhs.column_indices[i], rhs.column_indices[j]);
            addIndexVal( V, lhs.values[i]*rhs.values[j]);
        }
    cusp::array1d< int, cusp::host_memory> dI( I); // row indices
    cusp::array1d< int, cusp::host_memory> dJ( J); // column indices
    cusp::array1d< T,   cusp::host_memory> dV( V); // values
    // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
    thrust::stable_sort_by_key(dJ.begin(), dJ.end(), thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dV.begin())));
    thrust::stable_sort_by_key(dI.begin(), dI.end(), thrust::make_zip_iterator(thrust::make_tuple(dJ.begin(), dV.begin())));
    // compute unique number of ( values with different (i,j) index)  in the output
    int num_entries = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(dI.end (),  dJ.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())) + 1,
                                            int(0),
                                            thrust::plus<int>(),
                                            thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

    // allocate output matrix
    cusp::coo_matrix<int, T, cusp::host_memory> A(num_rows, num_cols, num_entries);
    // sum values with the same (i,j) index
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(dI.end(),   dJ.end())),
                          dV.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                          A.values.begin(),
                          thrust::equal_to< thrust::tuple<int,int> >(),
                          thrust::plus<T>());
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
