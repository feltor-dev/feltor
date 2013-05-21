#ifndef _DG_POLARISATION_CUH
#define _DG_POLARISATION_CUH

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>


#include "grid.cuh"
#include "functions.h"
#include "tensor.cuh"
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
    Polarisation2d( unsigned Nx, unsigned Ny, T hx, T hy, int bcx, int bcy);
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
    //create diagonal matrix entries
    for( unsigned i=0; i<n*n*Nx*Ny; i++)
        I[i] = J[i] = i;

    //create derivatives
    rightx = create::dx_asymm_mt<T,n>( Nx, hx, bcx); //create and transfer to device
    rightx = dg::dgtensor<T,n>( tensor<T,n>(Ny, delta), rightx);
    righty = create::dx_asymm_mt<T,n>( Ny, hy, bcy); //create and transfer to device
    righty = dg::dgtensor<T,n>( righty, tensor<T,n>( Nx, delta) );

    //create backward2d
    Operator<T, n> backward1d( DLT<n>::backward);
    Operator<T, n*n> backward2d = tensor( backward1d, backward1d);
    middle = tensor( Nx*Ny, backward2d);

    cusp::multiply( middle, rightx, rightx);
    cusp::multiply( middle, righty, righty);
    cusp::transpose( rightx, leftx); 
    cusp::transpose( righty, lefty); 
    //create weights
    Operator<T,n> weightsx(0.), weightsy(0.);
    for( unsigned i=0; i<n; i++)
        weightsx(i,i) = DLT<n>::weight[i]*hx/2.; // normalisation because F is invariant
    for( unsigned i=0; i<n; i++)
        weightsy(i,i) = DLT<n>::weight[i]*hy/2.; // normalisation because F is invariant
    Operator<T, n*n> weights2d = tensor( weightsy, weightsx);
    middle = tensor( Nx*Ny, weights2d*backward2d);

    //create jump 
    jumpx = create::jump_ot<T,n>( Nx, bcx); //without jump cg is unstable
    jumpx = dg::dgtensor<T,n>( tensor<T,n>(Ny, pipj), jumpx);
    jumpy = create::jump_ot<T,n>( Ny, bcy); //without jump cg is unstable
    jumpy = dg::dgtensor<T,n>( jumpy, tensor<T,n>(Nx, pipj));

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
//////////////////////////////////////////////////////////////////////////////////
template< class T, size_t n, class container>
struct Polarisation2dX
{
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::coo_matrix<int, T, MemorySpace> Matrix;
    Polarisation2dX( const Grid<T,n>& grid);
    Polarisation2dX( const Grid<T,n>& grid, bc bcx, bc bcy);
    Polarisation2dX( unsigned Nx, unsigned Ny, T hx, T hy, int bcx, int bcy);
    Matrix create( const container& );
  private:
    typedef cusp::array1d<int, MemorySpace> Array;
    void construct( unsigned Nx, unsigned Ny, T hx, T hy, int bcx, int bcy);
    Matrix leftx, lefty, rightx, righty, jumpx, jumpy;
    cusp::array1d<int, MemorySpace> I, J;
    container middle; //contain coeffs for chi multiplication
    container xchi;
    //cusp::array1d_view< typename container::iterator> xchi_view;
    cusp::coo_matrix_view<Array, Array, container,  int, T, MemorySpace> xchi_view; //make view of thrust vector
};

template <class T, size_t n, class container>
Polarisation2dX<T,n, container>::Polarisation2dX( const Grid<T,n>& g):
    I(n*n*g.Nx()*g.Ny()), J(I), middle( n*n*g.Nx()*g.Ny()), 
    xchi(middle), xchi_view( xchi.size(), xchi.size(), xchi.size(), I, J, xchi)
{
    int bx = (g.bcx() == PER)?-1:0;
    int by = (g.bcy() == PER)?-1:0;
    construct( g.Nx(), g.Ny(), g.hx(), g.hy(), bx, by);
}
template <class T, size_t n, class container>
Polarisation2dX<T,n, container>::Polarisation2dX( const Grid<T,n>& g, bc bcx, bc bcy):
    I(n*n*g.Nx()*g.Ny()), J(I), middle( n*n*g.Nx()*g.Ny()), 
    xchi(middle), xchi_view( xchi.size(), xchi.size(), xchi.size(), I, J, xchi)
{
    int bx = (bcx == PER)?-1:0;
    int by = (bcy == PER)?-1:0;
    construct( g.Nx(), g.Ny(), g.hx(), g.hy(), bx, by);
}
template <class T, size_t n, class container>
Polarisation2dX<T,n, container>::Polarisation2dX( unsigned Nx, unsigned Ny, T hx, T hy, int bcx, int bcy):
    I(n*n*Nx*Ny), J(I), middle( n*n*Nx*Ny), 
    xchi(middle), xchi_view( n*n*Nx*Ny, n*n*Nx*Ny, n*n*Nx*Ny, I, J, xchi)
{
    construct( Nx, Ny, hx, hy, bcx, bcy);
}
template <class T, size_t n, class container>
void Polarisation2dX<T,n, container>::construct( unsigned Nx, unsigned Ny, T hx, T hy, int bcx, int bcy)
{
    //create diagonal matrix entries
    for( unsigned i=0; i<n*n*Nx*Ny; i++)
        I[i] = J[i] = i;
    Operator<T, n> backward1d( DLT<n>::backward);
    Operator<T, n> forward1d( DLT<n>::forward);

    //create x and y derivative in xspace
    rightx = create::dx_asymm_mt<T,n>( Nx, hx, bcx); 
    rightx = sandwich<T,n>( backward1d, rightx, forward1d);
    rightx = dg::dgtensor<T,n>( tensor<T,n>(Ny, delta), rightx);
    righty = create::dx_asymm_mt<T,n>( Ny, hy, bcy); //create and transfer to device
    righty = sandwich<T,n>( backward1d, righty, forward1d);
    righty = dg::dgtensor<T,n>( righty, tensor<T,n>( Nx, delta) );

    cusp::transpose( rightx, leftx); 
    cusp::transpose( righty, lefty); 
    //create middle weight vector
    for( unsigned i=0; i<Ny*Nx; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<n; k++)
            {
                middle[i*n*n+j*n+k] = DLT<n>::weight[k]*hx/2.*DLT<n>::weight[j]*hy/2.; 
            }

    //create norm for jump matrices 
    Operator<T,n> weightsx(0.), weightsy(0.), winvx(0.), winvy(0.);
    for( unsigned i=0; i<n; i++)
    {
        weightsx(i,i) = DLT<n>::weight[i]*hx/2.; // normalisation because F is invariant
        weightsy(i,i) = DLT<n>::weight[i]*hy/2.; // normalisation because F is invariant
        winvx(i,i) = 1./weightsx(i,i); 
        winvy(i,i) = 1./weightsy(i,i); 
    }
    //create jump
    jumpx = create::jump_ot<T,n>( Nx, bcx); //jump without t!
    jumpx = sandwich<T,n>( winvx*forward1d.transpose(), jumpx, forward1d);
    jumpx = dg::dgtensor<T,n>( tensor<T,n>( Ny, weightsy), jumpx); //proper normalisation

    jumpy = create::jump_ot<T,n>( Ny, bcy); //without jump cg is unstable
    jumpy = sandwich<T,n>( winvy*forward1d.transpose(), jumpy, forward1d);
    jumpy = dg::dgtensor<T,n>( jumpy, tensor<T,n>( Nx, weightsx));

}

template< class T, size_t n, class container>
cusp::coo_matrix<int, T, typename thrust::iterator_space<typename container::iterator>::type> Polarisation2dX<T,n, container>::create( const container& chi)
{
    Matrix laplacex;
    Matrix laplacey;
    blas1::pointwiseDot( middle, chi, xchi);
    cusp::multiply( xchi_view, rightx, laplacex);
    cusp::multiply( xchi_view, righty, laplacey);
    cusp::multiply( leftx, laplacex, laplacex);
    cusp::multiply( lefty, laplacey, laplacey);
    cusp::add( laplacex, jumpx, laplacex);
    cusp::add( laplacey, jumpy, laplacey);
    cusp::add( laplacex, laplacey, laplacey);
    return laplacey;
}

} //namespace dg

#endif // _DG_POLARISATION_CUH
