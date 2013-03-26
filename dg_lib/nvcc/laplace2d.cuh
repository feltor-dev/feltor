#ifndef _DG_LAPLACE2D_CUH
#define _DG_LAPLACE2D_CUH

#include <cassert>
#include <cusp/coo_matrix.h>

#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

#include "functions.h"
#include "operators.cuh"

namespace dg
{

namespace create{

namespace detail{

struct AddIndex2d{
    AddIndex2d( size_t M, size_t n, size_t m):M(M), n(n), m(m), number(0) {}
    void operator()(    cusp::coo_matrix<int, double, cusp::host_memory>& hm, 
                        unsigned ip, unsigned i, unsigned jp, unsigned j, 
                        unsigned kp, unsigned k, unsigned lp, unsigned l, 
                        double value )
    {
        if( value == 0.) return;
        //std::cout << "number "<<number<<" value"<< value<<"\n";
        hm.row_indices[number]    = M*n*m*ip + n*m*jp + m*kp + lp;
        hm.column_indices[number] = M*n*m*i  + n*m*j  + m*k  + l;
        hm.values[number]         = value;
        number++;
    }
    void operator() ( cusp::array1d< int, cusp::host_memory>& Idx, 
                      unsigned i, unsigned j, unsigned k, unsigned l)
    {
        Idx[ number] = M*n*m*i + n*m*j + m*k + l;
        number ++;
    }
    void operator() ( cusp::array1d< double, cusp::host_memory>& Val, double value)
    {
        Val[ number] = value;
        number ++;
    }
  private:
    size_t M, n, m;
    unsigned number;
};

} //namespace detail

template< size_t n >
cusp::coo_matrix< int, double, cusp::host_memory> tensor( const cusp::coo_matrix< int, double, cusp::host_memory>& lhs,
        const cusp::coo_matrix< int, double, cusp::host_memory>& rhs)
{
    //assert quadratic matrices
    assert( lhs.num_rows == lhs.num_cols);
    assert( rhs.num_rows == rhs.num_cols);
    //assert dg matrices
    assert( lhs.num_rows%n == 0);
    assert( rhs.num_rows%n == 0);
    unsigned Nx = rhs.num_rows/n; 
    unsigned Ny = lhs.num_rows/n; 
    //std::cout << "Nx "<< Nx << " Ny "<<Ny<<" n "<<n<<"\n";
    //taken from the cusp examples
    //dimensions of the matrix
    int num_cols = lhs.num_rows*rhs.num_rows, num_rows( num_cols);
    //std::cout << "num_cols "<<num_cols<<std::endl;
    //std::cout << "num_values_lhs "<<lhs.values.size()<<std::endl;
    //std::cout << "num_values_rhs "<<rhs.values.size()<<std::endl;
    // number of (i,j,v) triplets
    int num_triplets    = lhs.values.size()*rhs.num_rows + lhs.num_rows*rhs.values.size();
    //std::cout << "num_triplets "<<num_triplets<<std::endl;
    // allocate storage for unordered triplets
    cusp::array1d< int,     cusp::host_memory> I( num_triplets); // row indices
    cusp::array1d< int,     cusp::host_memory> J( num_triplets); // column indices
    cusp::array1d< double,  cusp::host_memory> V( num_triplets); // values
    //fill triplet arrays

    detail::AddIndex2d addIndexRow( Nx, n,n );
    detail::AddIndex2d addIndexCol( Nx, n,n );
    detail::AddIndex2d addIndexVal( Nx, n,n );
    //First 1 x RHS
    for( unsigned i=0; i<Ny; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<rhs.num_entries; j++)
            {
                addIndexRow( I, i, rhs.row_indices[j]/n, k, rhs.row_indices[j]%n);
                addIndexCol( J, i, rhs.column_indices[j]/n, k, rhs.column_indices[j]%n);
                addIndexVal( V, rhs.values[j]);
            }

    for( unsigned i=0; i<Nx; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<lhs.num_entries; j++)
            {
                addIndexRow( I, lhs.row_indices[j]/n,i, lhs.row_indices[j]%n, k);
                addIndexCol( J, lhs.column_indices[j]/n,i, lhs.column_indices[j]%n, k);
                addIndexVal( V, lhs.values[j]);
            }
    //std::cout << "Last "<< lhs.column_indices[lhs.num_entries-1] << " "<<lhs.column_indices[lhs.num_entries-1]/n<<std::endl;
    //std::cout << "last values: " <<I[num_triplets-1]<< " "<<J[num_triplets-1] << " ";
    //std::cout << V[num_triplets-1]<<"\n";
    // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
    thrust::stable_sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
    thrust::stable_sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));

    // compute unique number of nonzeros in the output
    int num_entries = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                            int(0),
                                            thrust::plus<int>(),
                                            thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

    //std::cout << "Num_entries "<<num_entries<<"\n";
    //std::cout << "should be   "<<n*n*(6*n-1)*Nx*Ny <<std::endl;
    // allocate output matrix
    cusp::coo_matrix<int, double, cusp::host_memory> A(num_rows, num_cols, num_entries);
    
    // sum values with the same (i,j) index
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
                          V.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                          A.values.begin(),
                          thrust::equal_to< thrust::tuple<int,int> >(),
                          thrust::plus<double>());
    //std::cout << "last ping\n";
    return A;
}

/**
* @brief Create and assemble a cusp Matrix for the periodic 2d laplacian
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
cusp::coo_matrix<int, double, cusp::host_memory> laplace2d_per( unsigned Nx, unsigned Ny, double hx, double hy, double alpha = 1.)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A( n*n*Nx*Ny, n*n*Nx*Ny, n*n*(6*n-1)*Nx*Ny);
    //std::cout << "# of entries "<<n*n*(6*n-1)*Nx*Ny <<std::endl;
    Operator<double, n> l(  detail::lilj);
    Operator<double, n> r(  detail::rirj);
    Operator<double, n> lr( detail::lirj);
    Operator<double, n> rl( detail::rilj);
    Operator<double, n> d(  detail::pidxpj);
    Operator<double, n> tx( detail::pipj_inv), ty( tx);
    tx *= 2./hx;
    ty *= 2./hy;
    Operator< double, n> ax = lr*tx*rl+(d+l)*tx*(d+l).transpose() + alpha*(l+r);
    Operator< double, n> bx = -((d+l)*tx*rl+alpha*rl);
    Operator< double, n> ay = lr*ty*rl+(d+l)*ty*(d+l).transpose() + alpha*(l+r);
    Operator< double, n> by = -((d+l)*ty*rl+alpha*rl);
    std::cout << "ax/ay\n";
    std::cout << ax << ay <<std::endl;
    std::cout << "bx/by\n";
    std::cout << bx <<"\n" << by <<std::endl;
    //assemble the matrix
    detail::AddIndex2d add_index2d( Nx, n, n);
    for( unsigned i = 0; i < Ny; i++)
        for( unsigned j = 0; j < Nx; j++)
        {
            unsigned ip = ((i+1) > Ny-1) ? 0 : i+1;
            unsigned im =  (i==0) ? Ny-1 : i-1;
            unsigned jp = ((j+1) > Nx-1) ? 0 : j+1;
            unsigned jm =  (j==0) ? Nx-1 : j-1;
            for( unsigned kp=0; kp<n; kp++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned lp=0; lp<n; lp++)
                        for( unsigned l=0; l<n; l++)
                        {
                            add_index2d( A, i,i,j,j,kp,k, lp, l, 
                                       detail::delta( kp, k)*ax(lp,l) + 
                                       ay( kp, k)*detail::delta(lp,l) 
                                     ); // 2*n*( n*n) - n*n entries
                            add_index2d( A, i,i,j,jp,kp,k, lp, l, 
                                       detail::delta( kp, k)*bx(lp,l)
                                     );//n* (n*n) entries
                            add_index2d( A, i,i,j,jm,kp,k, lp, l, 
                                       detail::delta( kp, k)*bx(l,lp) 
                                     );//n* (n*n) entries
                            add_index2d( A, i,ip,j,j,kp,k, lp, l, 
                                       by( kp, k)*detail::delta( lp, l)
                                     );//n* (n*n) entries
                            add_index2d( A, i,im,j,j,kp,k, lp, l, 
                                       by( k, kp)*detail::delta( lp, l)
                                     );//n* (n*n) entries
                        }
        }
    A.sort_by_row_and_column( );

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
/*
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
*/


} //namespace create

} //namespace dg

#include "blas/thrust_vector.cuh"
#include "blas/laplace.cuh"

#endif // _DG_LAPLACE2D_CUH

