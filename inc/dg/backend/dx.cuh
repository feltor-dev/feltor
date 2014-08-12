#pragma once

#include <cassert>
#include <cusp/coo_matrix.h>

#include "grid.h"
#include "functions.h"
#include "operator.h"
#include "creation.cuh"

/*! @file 
  
  Simple 1d derivatives
  */
namespace dg
{
namespace create
{
///@addtogroup lowlevel
///@{
/**
* @brief Create and assemble a cusp Matrix for the symmetric 1d single derivative in XSPACE
*
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* The matrix isn't symmetric due to the normalisation T.
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size (used to compute normalisation)
* @param bcx boundary condition 
*
* @return Host Matrix in coordinate form 
*/
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dx_symm_normed(unsigned n, unsigned N, T h, bc bcx)
{
    unsigned size;
    if( bcx == PER) //periodic
        size = 3*n*n*N;
    else
        size = 3*n*n*N-2*n*n;
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator<T> d = create::pidxpj(n);
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;

    Operator< T> a = 1./2.*t*(d-d.transpose());
    //bcx = PER
    Operator<T> a_bound_right(a), a_bound_left(a);
    //left boundary
    if( bcx == DIR || bcx == DIR_NEU )
        a_bound_left += 0.5*t*l;
    else if (bcx == NEU || bcx == NEU_DIR)
        a_bound_left -= 0.5*t*l;
    //right boundary
    if( bcx == DIR || bcx == NEU_DIR)
        a_bound_right -= 0.5*t*r;
    else if( bcx == NEU || bcx == DIR_NEU)
        a_bound_right += 0.5*t*r;
    if( bcx == PER ) //periodic bc
        a_bound_left = a_bound_right = a;
    Operator< T> b = t*(1./2.*rl);
    Operator< T> bp = t*(-1./2.*lr); //pitfall: T*-m^T is NOT -(T*m)^T
    //transform to XSPACE
    Grid1d<T> g( 0,1, n, N);
    Operator<T> backward=g.dlt().backward();
    Operator<T> forward=g.dlt().forward();
    a = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
    b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
    bp = backward*bp*forward;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, 0,0,k,l, a_bound_left(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, 0,1,k,l, b(k,l)); //1+ x B
        if( bcx == PER )
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, 0,N-1,k,l, bp(k,l)); //- 1- x B^T
        }
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i-1, k, l, bp(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        if( bcx == PER)
        {
            for( unsigned l=0; l<n; l++) 
                detail::add_index<T>(n, A, number, N-1,0,  k,l, b(k,l));
        }
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-2,k,l, bp(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-1,k,l, a_bound_right(k,l));
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the 1d single forward derivative in XSPACE
*
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* Neumann BC means inner value for flux
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix in coordinate form 
*/
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dx_plus_normed( unsigned n, unsigned N, T h, bc bcx )
{
    unsigned size;
    if( bcx == PER) //periodic
        size = 2*n*n*N;
    else
        size = 2*n*n*N-n*n;
    //assert( (bcx == DIR || bcx == PER) && "only Dirichlet BC allowed"); 
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator<T> d = create::pidxpj(n);
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;
    Operator<T>  a = t*(-l-d.transpose());
    //if( dir == backward) a = -a.transpose();
    Operator< T> a_bound_left = a; //PER, NEU and NEU_DIR
    Operator< T> a_bound_right = a; //PER, DIR and NEU_DIR
    if( bcx == dg::DIR || bcx == dg::DIR_NEU) 
        a_bound_left = t*(-d.transpose());
    if( bcx == dg::NEU || bcx == dg::DIR_NEU)
        a_bound_right = t*(d);
    Operator< T> b = t*rl;
    //transform to XSPACE
    Grid1d<T> g( 0,1, n, N);
    Operator<T> backward=g.dlt().backward();
    Operator<T> forward=g.dlt().forward();
    a = backward*a*forward, a_bound_left = backward*a_bound_left*forward;
    b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, 0,0,k,l, a_bound_left(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, 0,1,k,l, b(k,l)); //1+ x B
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        if( bcx == PER )
        {
            for( unsigned l=0; l<n; l++) 
                detail::add_index<T>(n, A, number, N-1,0,  k,l, b(k,l));
        }
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-1,k,l, a_bound_right(k,l));
    }
    return A;
};
/**
* @brief Create and assemble a cusp Matrix for the 1d single backward derivative in XSPACE
*
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* Neumann BC means inner value for flux
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix in coordinate form 
*/
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dx_minus_normed( unsigned n, unsigned N, T h, bc bcx )
{
    unsigned size;
    if( bcx == PER) //periodic
        size = 2*n*n*N;
    else
        size = 2*n*n*N-n*n;
    //assert( (bcx == DIR || bcx == PER) && "only Dirichlet BC allowed"); 
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator<T> d = create::pidxpj(n);
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;
    Operator<T>  a = t*(l+d);
    //if( dir == backward) a = -a.transpose();
    Operator< T> a_bound_right = a; //PER, NEU and DIR_NEU
    Operator< T> a_bound_left = a; //PER, DIR and DIR_NEU
    if( bcx == dg::DIR || bcx == dg::NEU_DIR) 
        a_bound_right = t*(-d.transpose());
    if( bcx == dg::NEU || bcx == dg::NEU_DIR)
        a_bound_left = t*d;
    Operator< T> bp = -t*lr;
    //transform to XSPACE
    Grid1d<T> g( 0,1, n, N);
    Operator<T> backward=g.dlt().backward();
    Operator<T> forward=g.dlt().forward();
    a  = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
    bp = backward*bp*forward, a_bound_right = backward*a_bound_right*forward;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, 0,0,k,l, a_bound_left(k,l)); //1 x A
        if( bcx == PER )
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, 0,N-1,k,l, bp(k,l)); //- 1- x B^T
        }
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i-1, k, l, bp(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, N-1,N-2,k,l, bp(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-1,k,l, a_bound_right(k,l));
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the normalised jump in 1d in XSPACE.
*
* @ingroup create
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* The matrix is symmetric. Normalisation is missing
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param bcx boundary condition
*
* @return Host Matrix in coordinate form 
*/
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> jump_normed( unsigned n, unsigned N, T h, bc bcx)
{
    unsigned size;
    if( bcx == PER) //periodic
        size = 3*n*n*N;
    else
        size = 3*n*n*N-2*n*n;
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator< T> a = l+r;
    Operator< T> a_bound_left = a;//DIR and PER
    if( bcx == NEU || bcx == NEU_DIR)
        a_bound_left = r;
    Operator< T> a_bound_right = a; //DIR and PER
    if( bcx == NEU || bcx == DIR_NEU)
        a_bound_right = l;
    Operator< T> b = -rl;
    Operator< T> bp = -lr; 
    //transform to XSPACE
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;
    Grid1d<T> g( 0,1, n, N);
    Operator<T> backward=g.dlt().backward();
    Operator<T> forward=g.dlt().forward();
    a = backward*t*a*forward, a_bound_left  = backward*t*a_bound_left*forward;
    b = backward*t*b*forward, a_bound_right = backward*t*a_bound_right*forward;
    bp = backward*t*bp*forward;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,0,k,l, a_bound_left(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,1,k,l, b(k,l)); //1+ x B
        if( bcx == PER )
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, 0,N-1,k,l, bp(k,l)); //- 1- x B^T
        }
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i-1, k, l, bp(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        if( bcx == PER)
        {
            for( unsigned l=0; l<n; l++) 
                detail::add_index<T>(n, A, number, N-1,0,  k,l, b(k,l));
        }
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, N-1,N-2,k,l, bp(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>( n, A, number, N-1,N-1,k,l, a_bound_right(k,l));
    }
    return A;
};
///@}
} //namespace create
} //namespace dg

