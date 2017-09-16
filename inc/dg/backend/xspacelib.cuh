#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

//#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cusp/coo_matrix.h>
//
////functions for evaluation
#include "grid.h"
#include "dlt.h"
#include "operator.h"
#include "operator_tensor.cuh"
#include "dgtensor.cuh"
#include "interpolation.cuh" //makes typedefs available


/*! @file

  * @brief provides some utility functions
  */

namespace dg{

namespace create{
///@addtogroup scatter
///@{

/**
 * @brief Create a permutation matrix from a permutation map
 *
 * A permutation can be done in two ways. Either you name to every index in a vector
 * an index where this place should go to ( scatter) or you name to every index the 
 * index of the position that comes to this place (gather). A Scatter is the
 * inverse of a Gather operation with the same index-map. 
 * When transformed to a
 * permutation matrix scatter is the inverse ( = transpose) of gather. (Permutation
 * matrices are orthogonal and sparse)
 * @param map index map
 *
 * @return Permutation matrix
 */
cusp::coo_matrix<int, double, cusp::host_memory> gather( const thrust::host_vector<int>& map)
{
    typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
    Matrix p( map.size(), map.size(), map.size());
    cusp::array1d<int, cusp::host_memory> rows( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(map.size()));
    cusp::array1d<int, cusp::host_memory> cols( map.begin(), map.end());
    cusp::array1d<double, cusp::host_memory> values(map.size(), 1.);
    p.row_indices = rows;
    p.column_indices = cols;
    p.values = values;
    p.sort_by_row_and_column(); //important!!
    return p;
}

/**
 * @brief Create a permutation matrix from a permutation map
 *
 * A permutation can be done in two ways. Either you name to every index in a vector
 * an index where this place should go to ( scatter) or you name to every index the 
 * index of the position that comes to this place (gather). A Scatter is the
 * inverse of a Gather operation with the same index-map. 
 * When transformed to a
 * permutation matrix scatter is the inverse ( = transpose) of gather. (Permutation
 * matrices are orthogonal and sparse)
 * @param map index map
 *
 * @return Permutation matrix
 */
cusp::coo_matrix<int, double, cusp::host_memory> scatter( const thrust::host_vector<int>& map)
{
    typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
    Matrix p = gather( map);
    p.row_indices.swap( p.column_indices);
    p.sort_by_row_and_column(); //important!!
    return p;
}

/**
 * @brief make a matrix that transforms values to an equidistant grid ready for visualisation
 *
 * Useful if you want to visualize a dg-formatted vector.
 * @param g The grid on which to operate 
 *
 * @return transformation matrix
 * @note this matrix has ~n^4 N^2 entries and is not sorted
 */
cusp::coo_matrix<int, double, cusp::host_memory> backscatter( const aTopology2d& g)
{
    typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
    //create equidistant backward transformation
    dg::Operator<double> backwardeq( g.dlt().backwardEQ());
    dg::Operator<double> forward( g.dlt().forward());
    dg::Operator<double> backward1d = backwardeq*forward;

    Matrix transformX = dg::tensorproduct( g.Nx(), backward1d);
    Matrix transformY = dg::tensorproduct( g.Ny(), backward1d);
    Matrix backward = dg::dgtensor( g.n(), transformY, transformX);

    //thrust::host_vector<int> map = dg::create::gatherMap( g.n(), g.Nx(), g.Ny());
    //Matrix p = gather( map);
    //Matrix scatter( p);
    //cusp::multiply( p, backward, scatter);
    //choose vector layout
    //return scatter;
    return backward; 

}
/**
 * @brief make a matrix that transforms values to an equidistant grid ready for visualisation
 *
 * Useful if you want to visualize a dg-formatted vector.
 * @param g The 3d grid on which to operate 
 *
 * @return transformation matrix
 * @note this matrix has ~n^4 N^2 entries and is not sorted
 */
cusp::coo_matrix<int, double, cusp::host_memory> backscatter( const aTopology3d& g)
{
    Grid2d g2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy());
    cusp::coo_matrix<int,double, cusp::host_memory> back2d = backscatter( g2d);
    return dgtensor<double>( 1, tensorproduct<double>( g.Nz(), delta(1)), back2d);
}


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
thrust::host_vector<int> scatterMapInvertxy( unsigned n, unsigned Nx, unsigned Ny)
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
thrust::host_vector<int> contiguousLineNumbers( unsigned rows, unsigned cols)
{
    thrust::host_vector<int> map( rows*cols);
    for( unsigned i=0; i<map.size(); i++)
    {
        map[i] = i/cols;
    }
    return map;
}

///@}

} //namespace create
}//namespace dg
#endif // _DG_XSPACELIB_CUH_
