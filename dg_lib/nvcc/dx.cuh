#ifndef _DG_DX_CUH
#define _DG_DX_CUH

#include <cusp/coo_matrix.h>

#include "functions.h"
#include "operator.cuh"
#include "creation.cuh"

namespace dg
{
namespace create
{
/**
* @brief Create and assemble a cusp Matrix for the periodic 1d single derivative
*
* @ingroup utilities
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* The matrix is skew-symmetric
* @tparam n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size
*
* @return Host Matrix in coordinate form 
*/
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx_per( unsigned N, T h)
{
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, 3*n*n*N);
    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
    Operator<T, n> l( dg::lilj);
    Operator<T, n> r( dg::rirj);
    Operator<T, n> lr(dg::lirj);
    Operator<T, n> rl(dg::rilj);
    Operator<T, n> d( dg::pidxpj);
    Operator<T, n> t( dg::pipj_inv);
    t *= 2./h;
    Operator< T, n> a = 1./2.*t*(d-d.transpose());
    Operator< T, n> b = t*(1./2.*rl);
    //std::cout << a << "\n"<<b <<std::endl;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,0,k,l, a(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,1,k,l, b(k,l)); //1+ x B
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,N-1,k,l, -b(l,k)); //- 1- x B^T
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i-1, k, l, -b(l,k));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++) 
            detail::add_index<T, n>( A, number, N-1,0,  k,l, b(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, N-1,N-2,k,l, -b(l,k));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
};

} //namespace create
} //namespace dg

#endif //_DG_DX_CUH
