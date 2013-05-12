#ifndef _DG_POLARISATION_CUH
#define _DG_POLARISATION_CUH

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>

#include "functions.h"
#include "operator.cuh"
#include "operator_matrix.cuh"
#include "creation.cuh"
#include "dx.cuh"
#include "dlt.h"

namespace dg
{

//as far as i see in the source code cusp only supports coo - coo matrix
//-matrix multiplication
template< class T, size_t n, class Memory>
struct Polarisation
{
    typedef cusp::coo_matrix<int, T, Memory> Matrix;
    typedef cusp::array1d<T, Memory> Vector;
    Polarisation( unsigned N, T h, int bc);
    Matrix create( const Vector& );
  private:
    typedef cusp::array1d<int, Memory> Array;
    Matrix left, middle, right, jump;
    cusp::array1d<int, Memory> I, J;
    Vector xspace;
    unsigned N;

};
template <class T, size_t n, class Memory>
Polarisation<T,n, Memory>::Polarisation( unsigned N, T h, int bc): I(n*N), J(I), xspace( n*N), N(N)
{
    for( unsigned i=0; i<n*N; i++)
        I[i] = J[i] = i;
    right = create::dx_asymm_mt<T,n>( N, h, bc); //create and transfer to device
    Operator<T, n> backward( DLT<n>::backward);
    middle = tensor<T,n>( N, backward);
    cusp::multiply( middle, right, right);
    cusp::transpose( right, left); 
    Operator<T,n> weights(0.);
    for( unsigned i=0; i<n; i++)
        weights(i,i) = DLT<n>::weight[i]*h/2.; // normalisation because F is invariant
    middle = tensor<T,n>( N, weights*backward);
    jump = create::jump_ot<T,n>( N, bc); //without jump cg is unstable

}

template< class T, size_t n, class Memory>
cusp::coo_matrix<int, T, Memory> Polarisation<T,n, Memory>::create( const Vector& chi)
{
    Matrix laplace;
    cusp::multiply( middle, chi, xspace);
    cusp::coo_matrix_view<Array, Array, Vector,  int, T, Memory> chi_view( n*N, n*N, n*N, I, J, xspace);
    cusp::multiply( chi_view, right, laplace);
    cusp::multiply( left, laplace, laplace);
    cusp::add( laplace, jump, laplace);
    return laplace;
}
template< class T, size_t n, class Memory>
struct Polarisation2d
{
    typedef cusp::coo_matrix<int, T, Memory> Matrix;
    typedef cusp::array1d<T, Memory> Vector;
    Polarisation2d( unsigned Nx, unsigned Ny, T hx, T hy int bcx, int bcy);
    Matrix create( const Vector& );
  private:
    typedef cusp::array1d<int, Memory> Array;
    Matrix leftx, lefty, middle, rightx, righty, jumpx, jumpy;
    cusp::array1d<int, Memory> I, J;
    Vector xspace;
    unsigned Nx, Ny;
};

template <class T, size_t n, class Memory>
Polarisation2d<T,n, Memory>::Polarisation2d( unsigned Nx, unsigned Ny, T hx, T hy, int bcx, int bcy): I(n*n*Nx*Ny), J(I), xspace( n*n*Nx*Ny), Nx(Nx), Ny(Ny)
{
    for( unsigned i=0; i<n*n*Nx*Ny; i++)
        I[i] = J[i] = i;
    rightx = create::dx_asymm_mt<T,n>( Nx, hx, bcx); //create and transfer to device
    rightx = dg::dgtensor<T,n>( tensor<T,n>(Ny, delta), rightx);
    righty = create::dx_asymm_mt<T,n>( Ny, hy, bcy); //create and transfer to device
    righty = dg::dgtensor<T,n>( righty, tensor<T,n>( Nx, delta) );
    Operator<T, n> backward1d( DLT<n>::backward);
    Operator<T, n*n> backward2d = tensor( backward1d, backward1d);
    middle = tensor( Nx*Ny, backward2d);

    cusp::multiply( middle, rightx, rightx);
    cusp::multiply( middle, righty, righty);
    cusp::transpose( rightx, leftx); 
    cusp::transpose( righty, lefty); 
    Operator<T,n> weights(0.);
    for( unsigned i=0; i<n; i++)
        weights(i,i) = DLT<n>::weight[i]*h/2.; // normalisation because F is invariant
    Operator<T, n*n> weights2d = tensor( weights, weights);
    middle = tensor<T,n>( Nx*Ny, weights2d*backward2d);
    jumpx = create::jump_ot<T,n>( Nx, bcx); //without jump cg is unstable
    jumpx = dg::dgtensor( tensor<T,n>(Ny, delta), jumpx);
    jumpy = create::jump_ot<T,n>( Ny, bcy); //without jump cg is unstable
    jumpx = dg::dgtensor( jumpy, tensor<T,n>(Nx, delta));

}

template< class T, size_t n, class Memory>
cusp::coo_matrix<int, T, Memory> Polarisation2d<T,n, Memory>::create( const Vector& chi)
{
    Matrix laplacex;
    Matrix laplacey;
    unsigned size = chi.size();
    assert( chi.size() == I.size());
    cusp::multiply( middle, chi, xspace);
    cusp::coo_matrix_view<Array, Array, Vector,  int, T, Memory> chi_view( size, size, size, I, J, xspace);
    cusp::multiply( chi_view, rightx, laplacex);
    cusp::multiply( chi_view, righty, laplacey);
    cusp::multiply( leftx, laplacex, laplacex);
    cusp::multiply( lefty, laplacey, laplacey);
    cusp::add( laplacex, jumpx, laplacex);
    cusp::add( laplacey, jumpy, laplacey);
    cusp::add( laplacex, laplacey, laplacey);
    return laplacey;
}

} //namespace dg

#endif // _DG_POLARISATION_CUH
