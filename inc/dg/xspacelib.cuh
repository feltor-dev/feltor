#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>

//functions for evaluation
#include "grid.cuh"
#include "arrvec2d.cuh"
#include "functors.cuh"
#include "dlt.cuh"
#include "evaluation.cuh"


//creational functions
#include "derivatives.cuh"
#include "arakawa.cuh"
#include "polarisation.cuh"

//integral functions
#include "preconditioner.cuh"

#include "typedefs.cuh"

/*! @file

  * includes all relevant dg lib files for matrix creation and function evaluation
  * and provides some utility functions
  */

namespace dg{

namespace create{
///@addtogroup utilities
///@{
//to be used in thrust::scatter and thrust::gather (Attention: don't scatter inplace -> Pb with n>1)
//(the inverse is its transpose) 
/**
 * @brief Map for scatter operations on dg formatted vectors

 In 2D the vector elements of an x-space dg vector in one cell  lie
 contiguously in memory. Sometimes you want elements in the x-direction 
 to lie contiguously instead. This map can be used in a scatter operation 
 to permute elements in exactly that way.
 The elements of the map contain the indices where this place goes to
 i.e. w[m[i]] = v[i]
 
 * @param n # of polynomial coefficients
 * @param Nx # of points in x
 * @param Ny # of points in y
 *
 * @return map of indices
 */
thrust::host_vector<int> scatterMap(unsigned n, unsigned Nx, unsigned Ny )
{
    thrust::host_vector<int> map( n*n*Nx*Ny);
    for( unsigned i=0; i<Ny; i++)
        for( unsigned j=0; j<Nx; j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    map[ i*Nx*n*n + j*n*n + k*n + l] =(int)( i*Nx*n*n + k*Nx*n + j*n + l);
    return map;
}

/**
 * @brief Map for gather operations on dg formatted vectors

 In 2D the vector elements of an x-space dg vector in one cell  lie
 contiguously in memory. Sometimes you want elements in the x-direction 
 to lie contiguously instead. This map can be used in a gather operation 
 to permute elements in exactly that way.
 The elements of the map contain the indices that come at that place
 i.e. w[i] = v[m[i]]
 *
 * @param n # of polynomial coefficients
 * @param Nx # of points in x
 * @param Ny # of points in y
 *
 * @return map of indices
 */
thrust::host_vector<int> permutationMap( unsigned n, unsigned Nx, unsigned Ny )
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
 * @brief make a matrix that transforms values to an equidistant grid ready for visualisation
 *
 * Useful if you want to visualize a dg-formatted vector.
 * @tparam T value type
 * @param g The grid on which to operate 
 * @param s your vectors are given in XSPACE or in LSPACE
 *
 * @return transformation matrix
 */
template < class T>
cusp::coo_matrix<int, T, cusp::host_memory> backscatter( const Grid<T>& g, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    unsigned n = g.n();
    //create equidistant backward transformation
    dg::Operator<double> backwardeq( g.dlt().backwardEQ());
    dg::Operator<double> backward2d = dg::tensor( backwardeq, backwardeq);

    if( s == XSPACE){
        dg::Operator<double> forward( g.dlt().forward());
        dg::Operator<double> forward2d = dg::tensor( forward, forward);
        backward2d = backward2d*forward2d;
    }

    Matrix backward = dg::tensor( g.Nx()*g.Ny(), backward2d);

    thrust::host_vector<int> map = dg::create::permutationMap( g.n(), g.Nx(), g.Ny());
    Matrix permutation( map.size(), map.size(), map.size());
    cusp::array1d<int, cusp::host_memory> rows( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(map.size()));
    cusp::array1d<int, cusp::host_memory> cols( map.begin(), map.end());
    cusp::array1d<T, cusp::host_memory> values(map.size(), 1.);
    permutation.row_indices = rows;
    permutation.column_indices = cols;
    permutation.values = values;
    Matrix scatter( permutation);

    cusp::multiply( permutation, backward, scatter);
    return scatter;

}

///**
// * @brief Evaluate the jumps on grid boundaries
// *
// * @tparam n number of legendre nodes per cell
// * @param v A DG Host Vector 
// *
// * @return Vector with the jump values
// */
//thrust::host_vector< double> evaluate_jump( const ArrVec1d& v)
//{
//    //compute the interior jumps of a DG approximation
//    unsigned N = v.size();
//    thrust::host_vector<double> jump(N-1, 0.);
//    for( unsigned i=0; i<N-1; i++)
//        for( unsigned j=0; j<v.n(); j++)
//            jump[i] += v(i,j) - v(i+1,j)*( (j%2==0)?(1):(-1));
//    return jump;
//}

///@}

} //namespace create
}//namespace dg
#endif // _DG_XSPACELIB_CUH_
