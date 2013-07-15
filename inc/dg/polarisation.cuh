#ifndef _DG_POLARISATION_CUH
#define _DG_POLARISATION_CUH

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>


#include "blas.h"
#include "grid.cuh"
#include "functions.h"
#include "preconditioner.cuh"
#include "tensor.cuh"
#include "operator.cuh"
#include "operator_matrix.cuh"
#include "creation.cuh"
#include "dx.cuh"
#include "dlt.h"

//#include "cusp_eigen.h"

/*! @file 

  Contains object for the polarisation matrix creation
  */
namespace dg
{

//as far as i see in the source code cusp only supports coo - coo matrix
//-matrix multiplication
///@cond DEV
template< class T, size_t n, class Memory>
struct Polarisation
{
    typedef cusp::coo_matrix<int, T, Memory> Matrix;
    typedef cusp::array1d<T, Memory> Vector;
    Polarisation( unsigned N, T h, bc bcx);
    Matrix create( const Vector& );
  private:
    typedef cusp::array1d<int, Memory> Array;
    Matrix left, middle, right, jump;
    cusp::array1d<int, Memory> I, J;
    Vector xspace;
    unsigned N;

};
template <class T, size_t n, class Memory>
Polarisation<T,n, Memory>::Polarisation( unsigned N, T h, bc bcx): I(n*N), J(I), xspace( n*N), N(N)
{
    for( unsigned i=0; i<n*N; i++)
        I[i] = J[i] = i;
    right = create::dx_asymm_mt<T,n>( N, h, bcx); //create and transfer to device
    Operator<T, n> backward( DLT<n>::backward);
    middle = tensor<T,n>( N, backward);
    cusp::multiply( middle, right, right);
    cusp::transpose( right, left); 
    Operator<T,n> weights(0.);
    for( unsigned i=0; i<n; i++)
        weights(i,i) = DLT<n>::weight[i]*h/2.; // normalisation because F is invariant
    middle = tensor<T,n>( N, weights*backward);
    jump = create::jump_ot<T,n>( N, bcx); //without jump cg is unstable

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
    cusp::coo_matrix_view<Array, Array, Vector,  int, T, Memory> chi_view( size, size, size, I, J, xspace);//this actually copies the vectors ...
    cusp::multiply( chi_view, rightx, laplacex);
    cusp::multiply( chi_view, righty, laplacey);
    cusp::multiply( leftx, laplacex, laplacex);
    cusp::multiply( lefty, laplacey, laplacey);
    cusp::add( laplacex, jumpx, laplacex);
    cusp::add( laplacey, jumpy, laplacey);
    cusp::add( laplacex, laplacey, laplacey);
    return laplacey;
}
///@endcond
//////////////////////////////////////////////////////////////////////////////////
/**
 * @brief X-space version of polarisation term
 *
 * @ingroup creation
 * The term discretized is \f[ \nabla ( \chi \nabla ) \f]
 * @tparam T value-type
 * @tparam n # of polynomial coefficients
 * @tparam container The vector class on which to operate on
 */
template< class T, size_t n, class container>
struct Polarisation2dX
{
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::csr_matrix<int, T, MemorySpace> Matrix;
    /**
     * @brief Create Polarisation on a grid 
     *
     * @param g The 2D grid
     */
    Polarisation2dX( const Grid<T,n>& grid);
    /**
     * @brief Create polarisation term on a grid using different boundary conditions
     *
     * @param g The 2D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    Polarisation2dX( const Grid<T,n>& grid, bc bcx, bc bcy);
    //deprecated
    //Polarisation2dX( unsigned Nx, unsigned Ny, T hx, T hy, int bcx, int bcy);

    /**
     * @brief Create a unnormalized matrix for the 2d polarisation term in XSPACE
     *
     * The term discretized is \f[ \nabla ( \chi \nabla ) \f]
     * The returned matrix is symmetric with W2D missing from it and ready to use in CG
     * @param chi The polarisation vector on the grid
     *
     * @return matrix containing discretisation of polarisation term using chi 
     */
    Matrix create( const container& chi );
  private:
    typedef cusp::array1d<int, MemorySpace> Array;
    typedef cusp::array1d<T, MemorySpace> VArray;
    void construct( unsigned Nx, unsigned Ny, T hx, T hy, bc bcx, bc bcy);
    Matrix leftx, lefty, rightx, righty, jump;
    Array I, J;
    typename Array::view I_view, J_view;
    container middle; //contain coeffs for chi multiplication
    container xchi;
    typename VArray::view xchi_view;
    //cusp::array1d_view< typename container::iterator> xchi_view;
    cusp::coo_matrix_view<typename Array::view, typename Array::view, typename VArray::view,  int, T, MemorySpace> xchi_matrix_view; //make view of thrust vector
};

template <class T, size_t n, class container>
Polarisation2dX<T,n, container>::Polarisation2dX( const Grid<T,n>& g):
    I(n*n*g.Nx()*g.Ny()), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    middle( I.size()), xchi(middle), xchi_view( xchi.begin(), xchi.end()),
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    construct( g.Nx(), g.Ny(), g.hx(), g.hy(), g.bcx(), g.bcy());
}
template <class T, size_t n, class container>
Polarisation2dX<T,n, container>::Polarisation2dX( const Grid<T,n>& g, bc bcx, bc bcy):
    I(n*n*g.Nx()*g.Ny()), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    middle( I.size()), xchi(middle), xchi_view( xchi.begin(), xchi.end()),
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    construct( g.Nx(), g.Ny(), g.hx(), g.hy(), bcx, bcy);
}

//template <class T, size_t n, class container>
//Polarisation2dX<T,n, container>::Polarisation2dX( unsigned Nx, unsigned Ny, T hx, T hy, bc bcx, bc bcy):
//    I(n*n*Nx*Ny), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
//    middle( I.size()), xchi(middle), xchi_view( xchi.begin(), xchi.end()),
//    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
//{
//    construct( Nx, Ny, hx, hy, bcx, bcy);
//}
template <class T, size_t n, class container>
void Polarisation2dX<T,n, container>::construct( unsigned Nx, unsigned Ny, T hx, T hy, bc bcx, bc bcy)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> HMatrix;

    Operator<T, n> backward1d( DLT<n>::backward);
    Operator<T, n> forward1d( DLT<n>::forward);

    //create x and y derivative in xspace
    HMatrix rightx_ = create::dx_asymm_mt<T,n>( Nx, hx, bcx); 
    rightx_ = sandwich<T,n>( backward1d, rightx_, forward1d);
    rightx = dg::dgtensor<T,n>( tensor<T,n>(Ny, delta), rightx_);

    HMatrix righty_ = create::dx_asymm_mt<T,n>( Ny, hy, bcy); //create and transfer to device
    righty_ = sandwich<T,n>( backward1d, righty_, forward1d);
    righty = dg::dgtensor<T,n>( righty_, tensor<T,n>( Nx, delta) );

    cusp::transpose( rightx, leftx); 
    cusp::transpose( righty, lefty); 

    //create diagonal matrix entries
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());
    //create middle weight vector
    dg::W2D<T,n> w2d( hx, hy);
    thrust::transform( I.begin(), I.end(), middle.begin(), w2d);

    //create norm for jump matrices 
    Operator<T,n> normx(0.), normy(0.), winvx(0.), winvy(0.);
    for( unsigned i=0; i<n; i++)
    {
        normx(i,i) = DLT<n>::weight[i]*hx/2.; // normalisation because F is invariant
        normy(i,i) = DLT<n>::weight[i]*hy/2.; // normalisation because F is invariant
        //winvx(i,i) = 1./normx(i,i); 
        //winvy(i,i) = 1./normy(i,i); 
    }
    //create jump
    HMatrix jumpx = create::jump_ot<T,n>( Nx, bcx); //jump without t!
    jumpx = sandwich<T,n>( forward1d.transpose(), jumpx, forward1d);
    jumpx = dg::dgtensor<T,n>( tensor<T,n>( Ny, normy), jumpx); //proper normalisation

    HMatrix jumpy = create::jump_ot<T,n>( Ny, bcy); //without jump cg is unstable
    jumpy = sandwich<T,n>( forward1d.transpose(), jumpy, forward1d);
    jumpy = dg::dgtensor<T,n>( jumpy, tensor<T,n>( Nx, normx));
    HMatrix jump_;
    cusp::add( jumpx, jumpy, jump_); //does not respect sorting!!!
    jump_.sort_by_row_and_column();
    jump = jump_;
}

template< class T, size_t n, class container>
cusp::csr_matrix<int, T, typename thrust::iterator_space<typename container::iterator>::type> Polarisation2dX<T,n, container>::create( const container& chi)
{
    Matrix temp1, temp2, temp3;
    blas1::pointwiseDot( middle, chi, xchi);
    //multiply also does not necessarily keep the sorting
    cusp::multiply( xchi_matrix_view, rightx, temp1); //D_x*R_x
    cusp::multiply( xchi_matrix_view, righty, temp2); //D_y*R_y
    cusp::multiply( leftx, temp1, temp3); //L_x*D_x*R_x
    cusp::multiply( lefty, temp2, temp1); //L_y*D_y*R_y
    cusp::add( temp1, temp3, temp2);  // D_yy + D_xx
    cusp::add( temp2, jump, temp1); // Lap + Jump
    //temp1.sort_by_row_and_column(); //add does not sort
    /*
    cusp::coo_matrix<int, double, cusp::host_memory> temp_ = temp2;
    t.tic();
    eigenpol.compute( xchi_matrix_view, temp_);
    t.toc();
    std::cout << "point8 "<<t.diff()<<"s\n";
    */
   
    return temp1;
}

} //namespace dg

#endif // _DG_POLARISATION_CUH
