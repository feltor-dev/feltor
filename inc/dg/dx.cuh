#ifndef _DG_DX_CUH
#define _DG_DX_CUH

#include <cassert>
#include <cusp/coo_matrix.h>

#include "grid.cuh"
#include "functions.h"
#include "operator_dynamic.h"
#include "creation.cuh"

/*!@file simple 1d derivatives
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
enum direction{
    forward, 
    backward
};
namespace create
{
///@addtogroup lowlevel
///@{
/**
* @brief Create and assemble a cusp Matrix for the symmetric 1d single derivative
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
cusp::coo_matrix<int, T, cusp::host_memory> dx_symm(unsigned n, unsigned N, T h, bc bcx)
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
    //Operator< T> a_bound_right = t*(-1./2.*l-d.transpose()); //=T[ 0.5(D-D^T)-0.5 R]
    //Operator< T> a_bound_left = t*(1./2.*r-d.transpose());// = T[ 0.5(D-D^T) +0.5 L]
    Operator< T> b = t*(1./2.*rl);
    Operator< T> bp = t*(-1./2.*lr); //pitfall: T*-m^T is NOT -(T*m)^T
    //std::cout << a << "\n"<<b <<std::endl;
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
* @brief Create and assemble a cusp Matrix for the 1d single forward derivative
*
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* The matrix isn't skew-symmetric due to normalisation T.
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
cusp::coo_matrix<int, T, cusp::host_memory> dx_asymm_mt( unsigned n, unsigned N, T h, bc bcx )
{
    unsigned size;
    if( bcx == PER) //periodic
        size = 2*n*n*N;
    else
        size = 2*n*n*N-n*n;
    //assert( (bcx == DIR || bcx == PER) && "only Dirichlet BC allowed"); 
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, size);

    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
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
* @brief Create and assemble a cusp Matrix for the unnormalised jump in 1d.
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
cusp::coo_matrix<int, T, cusp::host_memory> jump_ot( unsigned n, unsigned N, bc bcx)
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

#endif //_DG_DX_CUH
