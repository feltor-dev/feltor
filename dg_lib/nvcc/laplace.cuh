#ifndef _DG_LAPLACE_CUH
#define _DG_LAPLACE_CUH

#include <cusp/coo_matrix.h>

#include "functions.h"
#include "operator.cuh"
#include "creation.cuh"

namespace dg
{

namespace create{

/**
* @brief Create and assemble a cusp Matrix for the periodic negative 1d laplacian
*
* @ingroup utilities
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* @tparam n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size
* @param alpha Optional parameter for penalization term
*
* @return Host Matrix in coordinate form 
* @note The normalisation factor T is missing from this matrix
*/
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d_per( unsigned N, T h, T alpha = 1.)
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
    Operator< T, n> a = lr*t*rl+(d+l)*t*(d+l).transpose() + alpha*(l+r);
    Operator< T, n> b = -((d+l)*t*rl+alpha*rl);
    //std::cout << a << "\n"<<b <<std::endl;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, 0,0,k,l, a(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, 0,1,k,l, b(k,l)); //1+ x B
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, 0,N-1,k,l, b(l,k)); //1- x B^T
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T,n>(A, number, i, i-1, k, l, b(l,k));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T,n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T,n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++) 
            detail::add_index<T,n>( A, number, N-1,0,  k,l, b(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, N-1,N-2,k,l, b(l,k));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the Dirichlet negative 1d laplacian
*
* @ingroup utilities
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* @tparam n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size
* @param alpha Optional parameter for penalization term
*
* @return Host Matrix in coordinate form 
* @note The normalisation factor T is missing from this matrix
*/
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d_dir( unsigned N, T h, T alpha = 1.)
{
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, 3*n*n*N - 2*n*n);
    Operator<T, n> l( dg::lilj);
    Operator<T, n> r( dg::rirj);
    Operator<T, n> lr(dg::lirj);
    Operator<T, n> rl(dg::rilj);
    Operator<T, n> d( dg::pidxpj);
    Operator<T, n> s( dg::pipj);
    Operator<T, n> t( dg::pipj_inv);
    t *= 2./h;

    Operator<T, n> a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    Operator<T, n> b = -((d+l)*t*rl+rl);
    Operator<T, n> ap = d*t*d.transpose() + l + r;
    Operator<T, n> bp = -(d*t*rl + rl);
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, 0,0,k,l, ap(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, 0,1,k,l, bp(k,l));
    }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>(A, number, 1, 1-1, k, l, bp(l,k));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>(A, number, 1, 1, k, l, a(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>(A, number, 1, 1+1, k, l, b(k,l));
    }
    for( unsigned i=2; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T,n>(A, number, i, i-1, k, l, b(l,k));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T,n>(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T,n>(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, N-1,N-2,k,l, b(l,k));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T,n>( A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
}


} //namespace create

//as far as i see in the source code cusp only supports coo - coo matrix
//-matrix multiplication
template< class T, size_t n>
struct NonlinearLaplace
{
    typedef cusp::coo_matrix<int, T, cusp::device_memory> DMartrix;
    NonlinearLaplace( unsigned N, T h, int bc);
    DMatrix operator()( const thrust::device_vector<T>& );
  private:
    DMatrix left, right;

};
template <class T, size_t n>
NonlinearLaplace<T,n>::NonlinearLaplace( unsigned N, T h, int bc):
    //allocate left and right
{
    //assemble left and right
}

template< class T, size_t n>
DMatrix NonlinearLaplace::operator()( const thrust::device_vector<T>& n)

} //namespace dg

#endif // _DG_LAPLACE_CUH
