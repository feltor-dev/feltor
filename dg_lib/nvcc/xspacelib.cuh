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
#include "dlt.h"
#include "evaluation.cuh"


//creational functions
#include "derivatives.cuh"
#include "arakawa.cuh"
#include "polarisation.cuh"

//integral functions
#include "preconditioner.cuh"

#include "typedefs.cuh"

namespace dg{
///@addtogroup utilities
///@{

namespace create{
//to be used in thrust::scatter and thrust::gather (Attention: don't scatter inplace -> Pb with n>1)
//(the inverse is its transpose) 
/**
 * @brief Map for scatter operations on dg formatted vectors

 The elements of the map contain the indices where this place goes to
 i.e. w[m[i]] = v[i]
 
 * @tparam n # of polynomial coefficients
 * @param Nx # of points in x
 * @param Ny # of points in y
 *
 * @return map of indices
 */
template< size_t n>
thrust::host_vector<int> scatterMap( unsigned Nx, unsigned Ny )
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

 The elements of the map contain the indices that come at that place
 i.e. w[i] = v[m[i]]
 
 *
 * @tparam n # of polynomial coefficients
 * @param Nx # of points in x
 * @param Ny # of points in y
 *
 * @return map of indices
 */
template< size_t n>
thrust::host_vector<int> permutationMap( unsigned Nx, unsigned Ny )
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
 * @tparam T value type
 * @tparam n # of polynomial coefficients
 * @param g The grid on which to operate 
 * @param s your vectors are given in XSPACE or in LSPACE
 *
 * @return transformation matrix
 */
template < class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> backscatter( const Grid<T,n>& g, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    //create equidistant backward transformation
    dg::Operator<double, n> backwardeq( dg::DLT<n>::backwardEQ);
    dg::Operator<double, n*n> backward2d = dg::tensor( backwardeq, backwardeq);

    if( s == XSPACE){
        dg::Operator<double, n> forward( dg::DLT<n>::forward);
        dg::Operator<double, n*n> forward2d = dg::tensor( forward, forward);
        backward2d = backward2d*forward2d;
    }

    Matrix backward = dg::tensor( g.Nx()*g.Ny(), backward2d);

    thrust::host_vector<int> map = dg::create::permutationMap<n>( g.Nx(), g.Ny());
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
} //namespace create
///@}
}//namespace dg
#endif // _DG_XSPACELIB_CUH_
