#ifndef _DG_LAPLACE_CUH
#define _DG_LAPLACE_CUH

#include <cusp/coo_matrix.h>

#include "grid.cuh"
#include "functions.h"
#include "operator_dynamic.h"
#include "creation.cuh"

/*! @file 1d laplacians
  */

namespace dg
{
/**
 * @brief Switch between normalisations
 *
 * @ingroup creation
 */
enum norm{
    normed,   //!< indicates that output is properly normalized
    not_normed //!< indicates that normalisation weights (either T or V) are missing from output
};

namespace create{

/**
 * @brief Create and assemble a cusp Matrix for the negative periodic 1d laplacian in LSPACE
 *
 * @ingroup highlevel
 * Use cusp internal conversion to create e.g. the fast ell_matrix format.
 * @tparam T value-type
 * @param n Number of Legendre nodes per cell
 * @param N Vector size ( number of cells)
 * @param h cell size
 * @param no use normed if you want to compute e.g. diffusive terms
             use not_normed if you want to solve symmetric matrix equations (T is missing)
 * @param alpha Optional parameter for penalization term
 *
 * @return Host Matrix in coordinate form 
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d_per( unsigned n, unsigned N, T h, norm no = not_normed, T alpha = 1.)
{
    if( n ==1 ) alpha = 0; //makes laplacian of order 2
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, 3*n*n*N);
    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator<T> d = create::pidxpj(n);
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;
    Operator< T> a = lr*t*rl+(d+l)*t*(d+l).transpose() + alpha*(l+r);
    Operator< T> b = -((d+l)*t*rl+alpha*rl);
    Operator< T> bT = b.transpose();
    if( no == normed) { a = t*a; b = t*b; bT = t*bT; }
    //std::cout << a << "\n"<<b <<std::endl;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,0,k,l, a(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,1,k,l, b(k,l)); //1+ x B
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,N-1,k,l, bT(k,l)); //1- x B^T
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i-1, k, l, bT(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++) 
            detail::add_index<T>(n, A, number, N-1,0,  k,l, b(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-2,k,l, bT(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
};

/**
 * @brief Create and assemble a cusp Matrix for the Dirichlet negative 1d laplacian in LSPACE
 *
 * @ingroup highlevel
 * Use cusp internal conversion to create e.g. the fast ell_matrix format.
 * @tparam T value-type
 * @param n Number of Legendre nodes per cell
 * @param N Vector size ( number of cells)
 * @param h cell size
 * @param no use normed if you want to compute e.g. diffusive terms
             use not_normed if you want to solve symmetric matrix equations (T is missing)
 *
 * @return Host Matrix in coordinate form 
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d_dir( unsigned n, unsigned N, T h, norm no = not_normed)
{
    //if( n == 1) alpha = 0; //not that easily because dirichlet 
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, 3*n*n*N - 2*n*n);
    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator<T> d = create::pidxpj(n);
    Operator<T> s = create::pipj(n);
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;

    Operator<T> a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    Operator<T> b = -((d+l)*t*rl+rl);
    Operator<T> bT= b.transpose();
    Operator<T> ap = d*t*d.transpose() + (l + r);
    Operator<T> bp = -(d*t*rl + rl);
    Operator<T> bpT= bp.transpose();
    if( no == normed) { 
        a=t*a; b=t*b; bT=t*bT; 
        ap=t*ap; bp=t*bp; bpT=t*bpT;
    }
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,0,k,l, ap(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,1,k,l, bp(k,l));
    }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 1, 1-1, k, l, bpT(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 1, 1, k, l, a(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 1, 1+1, k, l, b(k,l));
    }
    for( unsigned i=2; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i-1, k, l, bT(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-2,k,l, bT(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
}

/**
 * @brief Convenience function for the creation of a 1d laplacian in LSPACE
 *
 * @ingroup highlevel
 * @tparam T value_type
 * @param g The grid on which to create the laplacian (including boundary condition)
 * @param no use normed if you want to compute e.g. diffusive terms
            use not_normed if you want to solve symmetric matrix equations (T is missing)
 *
 * @return Host Matrix in coordinate form
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d( const Grid1d<T>& g, norm no = not_normed)
{

    if( g.bcx() == DIR)
        return laplace1d_dir<T>( g.n(), g.N(), g.h(), no);
    else 
        return laplace1d_per<T>( g.n(), g.N(), g.h(), no);
}



} //namespace create

} //namespace dg

#endif // _DG_LAPLACE_CUH
