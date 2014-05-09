#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>

//functions for evaluation
#include "grid.cuh"
#include "functors.cuh"
#include "dlt.cuh"
#include "evaluation.cuh"


//creational functions
#include "derivatives.cuh"
#include "arakawa.cuh"
#include "polarisation.cuh"
#include "weights.cuh"

//integral functions
#include "typedefs.cuh"

/*! @file

  * includes all relevant dg lib files for matrix creation and function evaluation
  * and provides some utility functions
  */

namespace dg{

namespace create{
///@addtogroup scatter
///@{
//TODO make one scatterMap for n, m and then apply to projection
//to be used in thrust::scatter and thrust::gather (Attention: don't scatter inplace -> Pb with n>1)
//(the inverse is its transpose) 
/**
 * @brief Index Map for scatter operation on dg formatted vectors

 In 2D the vector elements of an x-space dg vector in one cell  lie
 contiguously in memory. Sometimes you want elements in the x-direction 
 to lie contiguously instead. This map can be used in a scatter operation 
 to permute elements in exactly that way.
 The elements of the map contain the indices where this place goes to
 i.e. w[m[i]] = v[i]
 Scatter from not-contiguous to contiguous or Gather from contiguous to non-contiguous
 
 * @param nx # of polynomial coefficients in x
 * @param ny # of polynomial coefficients in y
 * @param Nx # of points in x
 * @param Ny # of points in y
 *
 * @return map of indices
 */
thrust::host_vector<int> scatterMap(unsigned nx, unsigned ny, unsigned Nx, unsigned Ny )
{
    thrust::host_vector<int> map( nx*ny*Nx*Ny);
    for( unsigned i=0; i<Ny; i++)
        for( unsigned j=0; j<Nx; j++)
            for( unsigned k=0; k<ny; k++)
                for( unsigned l=0; l<nx; l++)
                    map[ i*Nx*nx*ny + j*nx*ny + k*nx + l] =(int)( i*Nx*nx*ny + k*Nx*nx + j*nx + l);
    return map;
}

thrust::host_vector<int> scatterMap( unsigned n, unsigned Nx, unsigned Ny)
{
    return scatterMap( n, n, Nx, Ny);
}

/**
 * @brief Index map for gather operations on dg formatted vectors

 In 2D the vector elements of an x-space dg vector in one cell  lie
 contiguously in memory. Sometimes you want elements in the x-direction 
 to lie contiguously instead. This map can be used in a gather operation 
 to permute elements in exactly that way.
 The elements of the map contain the indices that come at that place
 i.e. w[i] = v[m[i]]
 Gather from not-contiguous to contiguous or Scatter from contiguous to non-contiguous
 *
 * @param n # of polynomial coefficients
 * @param Nx # of points in x
 * @param Ny # of points in y
 *
 * @return map of indices
 */
thrust::host_vector<int> gatherMap( unsigned n, unsigned Nx, unsigned Ny )
{
    thrust::host_vector<int> map( n*n*Nx*Ny);
    for( unsigned i=0; i<Ny; i++)
        for( unsigned j=0; j<Nx; j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    map[ i*Nx*n*n + k*Nx*n + j*n + l] =(int)( i*Nx*n*n + j*n*n + k*n + l);
    return map;
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
 * @tparam T value type
 * @param g The grid on which to operate 
 * @param s your vectors are given in XSPACE or in LSPACE
 *
 * @return transformation matrix
 * @note this matrix has ~n^4 N^2 entries and is not sorted
 */
template < class T>
cusp::coo_matrix<int, T, cusp::host_memory> backscatter( const Grid2d<T>& g, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    //create equidistant backward transformation
    dg::Operator<double> backwardeq( g.dlt().backwardEQ());
    dg::Operator<double> backward2d = dg::tensor( backwardeq, backwardeq);

    if( s == XSPACE){
        dg::Operator<double> forward( g.dlt().forward());
        dg::Operator<double> forward2d = dg::tensor( forward, forward);
        backward2d = backward2d*forward2d;
    }

    Matrix backward = dg::tensor( g.Nx()*g.Ny(), backward2d);

    //you get a permutation matrix by setting the column indices to the permutation values and the values to 1
    thrust::host_vector<int> map = dg::create::gatherMap( g.n(), g.Nx(), g.Ny());
    Matrix p = gather( map);
    /*
    Matrix permutation( map.size(), map.size(), map.size());
    cusp::array1d<int, cusp::host_memory> rows( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(map.size()));
    cusp::array1d<int, cusp::host_memory> cols( map.begin(), map.end());
    cusp::array1d<T, cusp::host_memory> values(map.size(), 1.);
    permutation.row_indices = rows;
    permutation.column_indices = cols;
    permutation.values = values;
    */
    Matrix scatter( p);
    cusp::multiply( p, backward, scatter);
    return scatter;

}
/**
 * @brief make a matrix that transforms values to an equidistant grid ready for visualisation
 *
 * Useful if you want to visualize a dg-formatted vector.
 * @tparam T value type
 * @param g The 3d grid on which to operate 
 * @param s your vectors are given in XSPACE or in LSPACE
 *
 * @return transformation matrix
 * @note this matrix has ~n^4 N^2 entries and is not sorted
 */
template < class T>
cusp::coo_matrix<int, T, cusp::host_memory> backscatter( const Grid3d<T>& g, space s = XSPACE)
{
    Grid2d<T> g2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy());
    cusp::coo_matrix<int,T, cusp::host_memory> back2d = backscatter( g2d, s);
    return dgtensor<T>( 1, tensor<T>( g.Nz(), delta(1)), back2d);
}

 /*
 * @brief Evaluate the jumps on grid boundaries
 *
 * @tparam n number of legendre nodes per cell
 * @param v A DG Host Vector 
 *
 * @return Vector with the jump values
thrust::host_vector< double> evaluate_jump( const ArrVec1d& v)
{
    //compute the interior jumps of a DG approximation
    unsigned N = v.size();
    thrust::host_vector<double> jump(N-1, 0.);
    for( unsigned i=0; i<N-1; i++)
        for( unsigned j=0; j<v.n(); j++)
            jump[i] += v(i,j) - v(i+1,j)*( (j%2==0)?(1):(-1));
    return jump;
}
 */

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
    thrust::host_vector<int> reorder = scatterMap( n, Nx, Ny);
    thrust::host_vector<int> map( n*n*Nx*Ny);
    thrust::host_vector<int> map2( map);
    for( unsigned i=0; i<map.size(); i++)
    {
        int row = i/Nx_;
        int col = i%Nx_;

        map[i] =  col*Ny_+row;
    }
    for( unsigned i=0; i<map.size(); i++)
        map2[i] = map[reorder[i]];
    return map2;
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
