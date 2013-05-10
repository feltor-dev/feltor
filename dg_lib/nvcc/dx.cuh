#ifndef _DG_DX_CUH
#define _DG_DX_CUH

#include <cassert>
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
* @param bc boundary condition: <0 is periodic, else homogeneous dirichlet 
*
* @return Host Matrix in coordinate form 
*/
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx_symm( unsigned N, T h, int bc = -1)
{
    unsigned size;
    if( bc < 0) //periodic
        size = 3*n*n*N;
    else
        size = 3*n*n*N-2*n*n;
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

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
    Operator< T, n> a_bound_right = t*(-1./2.*l-d.transpose());
    Operator< T, n> a_bound_left = t*(1./2.*r-d.transpose());
    if( bc < 0 ) //periodic bc
        a_bound_left = a_bound_right = a;
    Operator< T, n> b = t*(1./2.*rl);
    Operator< T, n> bp = t*(-1./2.*lr); //pitfall: T*-m^T is NOT -(T*m)^T
    //std::cout << a << "\n"<<b <<std::endl;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,0,k,l, a_bound_left(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,1,k,l, b(k,l)); //1+ x B
        if( bc <0 )
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>( A, number, 0,N-1,k,l, bp(k,l)); //- 1- x B^T
        }
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i-1, k, l, bp(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        if( bc < 0)
        {
            for( unsigned l=0; l<n; l++) 
                detail::add_index<T, n>( A, number, N-1,0,  k,l, b(k,l));
        }
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, N-1,N-2,k,l, bp(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, N-1,N-1,k,l, a_bound_right(k,l));
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the periodic 1d single derivative
*
* @ingroup utilities
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* The matrix is skew-symmetric
* @tparam n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size
* @param bc boundary condition: <0 is periodic, else homogeneous dirichlet 
*
* @return Host Matrix in coordinate form 
*/
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx_asymm_mt( unsigned N, T h, int bc = -1)
{
    unsigned size;
    if( bc < 0) //periodic
        size = 2*n*n*N;
    else
        size = 2*n*n*N-n*n;
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
    Operator<T, n> l( dg::lilj);
    Operator<T, n> r( dg::rirj);
    Operator<T, n> lr(dg::lirj);
    Operator<T, n> rl(dg::rilj);
    Operator<T, n> d( dg::pidxpj);
    Operator<T, n> t( dg::pipj_inv);
    t *= 2./h;
    Operator<T, n>  a = t*(-l-d.transpose());
    Operator< T, n> a_bound_left = t*(-d.transpose());
    if( bc < 0) //periodic bc
        a_bound_left = a;
    Operator< T, n> b = t*(rl);
    Operator< T, n> bp = t*(-lr); //pitfall: T*-m^T is NOT -(T*m)^T
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,0,k,l, a_bound_left(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,1,k,l, b(k,l)); //1+ x B
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        if( bc < 0)
        {
            for( unsigned l=0; l<n; l++) 
                detail::add_index<T, n>( A, number, N-1,0,  k,l, b(k,l));
        }
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
};

template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> jump_ot( unsigned N, int bc = -1)
{
    unsigned size;
    if( bc < 0) //periodic
        size = 3*n*n*N;
    else
        size = 3*n*n*N-2*n*n;
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
    Operator<T, n> l( dg::lilj);
    Operator<T, n> r( dg::rirj);
    Operator<T, n> lr(dg::lirj);
    Operator<T, n> rl(dg::rilj);
    Operator< T, n> a = l+r;
    Operator< T, n> b = -rl;
    Operator< T, n> bp = b.transpose(); 
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,0,k,l, a(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, 0,1,k,l, b(k,l)); //1+ x B
        if( bc <0 )
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>( A, number, 0,N-1,k,l, bp(k,l)); //- 1- x B^T
        }
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i-1, k, l, bp(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T, n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        if( bc < 0)
        {
            for( unsigned l=0; l<n; l++) 
                detail::add_index<T, n>( A, number, N-1,0,  k,l, b(k,l));
        }
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, N-1,N-2,k,l, bp(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T, n>( A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
};
} //namespace create
} //namespace dg

#endif //_DG_DX_CUH
