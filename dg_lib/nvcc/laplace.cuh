#ifndef _DG_LAPLACE_CUH
#define _DG_LAPLACE_CUH

#include <cusp/coo_matrix.h>

#include "functions.h"
#include "operators.cuh"

namespace dg
{

namespace create{

namespace detail{

template< size_t n>
void add_index( cusp::coo_matrix<int, double, cusp::host_memory>& hm, 
                int& number, 
                unsigned i, unsigned j, unsigned k, unsigned l, 
                double value )
{
    hm.row_indices[number] = n*i+k;
    hm.column_indices[number] = n*j+l;
    hm.values[number] = value;
    number++;
}
} //namespace detail

/**
* @brief Create and assemble a cusp Matrix for the periodic 1d laplacian
*
* @ingroup utilities
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* @tparam n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size
* @param alpha Optional parameter for penalization term
*
* @return Host Matrix in coordinate form 
*/
template< size_t n>
cusp::coo_matrix<int, double, cusp::host_memory> laplace1d_per( unsigned N, double h, double alpha = 1.)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A( n*N, n*N, 3*n*n*N);
    Operator<double, n> l( detail::lilj);
    Operator<double, n> r( detail::rirj);
    Operator<double, n> lr( detail::lirj);
    Operator<double, n> rl( detail::rilj);
    Operator<double, n> d( detail::pidxpj);
    Operator<double, n> t( detail::pipj_inv);
    t *= 2./h;
    Operator< double, n> a = lr*t*rl+(d+l)*t*(d+l).transpose() + alpha*(l+r);
    Operator< double, n> b = -((d+l)*t*rl+alpha*rl);
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, 0,0,k,l, a(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, 0,1,k,l, b(k,l)); //1+ x B
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, 0,N-1,k,l, b(l,k)); //1- x B^T
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<n>(A, number, i, i-1, k, l, b(l,k));
            for( unsigned l=0; l<n; l++)
                detail::add_index<n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++) 
            detail::add_index<n>( A, number, N-1,0,  k,l, b(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, N-1,N-2,k,l, b(l,k));
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the Dirichlet 1d laplacian
*
* @ingroup utilities
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* @tparam n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size
* @param alpha Optional parameter for penalization term
*
* @return Host Matrix in coordinate form 
*/
template< size_t n>
cusp::coo_matrix<int, double, cusp::host_memory> laplace1d_dir( unsigned N, double h, double alpha = 1.)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A( n*N, n*N, 3*n*n*N);
    Operator<double, n> l( detail::lilj);
    Operator<double, n> r( detail::rirj);
    Operator<double, n> lr( detail::lirj);
    Operator<double, n> rl( detail::rilj);
    Operator<double, n> d( detail::pidxpj);
    Operator<double, n> s( detail::pipj);
    Operator<double, n> t( detail::pipj_inv);
    t *= 2./h;

    Operator<double, n> a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    Operator<double, n> b = -((d+l)*t*rl+rl);
    Operator<double, n> ap = d*t*d.transpose() + l + r;
    Operator<double, n> bp = -(d*t*rl + rl);
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, 0,0,k,l, ap(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, 0,1,k,l, bp(k,l));
    }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>(A, number, 1, 1-1, k, l, bp(l,k));
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>(A, number, 1, 1, k, l, a(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>(A, number, 1, 1+1, k, l, b(k,l));
    }
    for( unsigned i=2; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<n>(A, number, i, i-1, k, l, b(l,k));
            for( unsigned l=0; l<n; l++)
                detail::add_index<n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, N-1,N-2,k,l, b(l,k));
        for( unsigned l=0; l<n; l++)
            detail::add_index<n>( A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
}


} //namespace create

} //namespace dg

#include "blas/thrust_vector.cuh"
#include "blas/laplace.cuh"

#endif // _DG_LAPLACE_CUH
