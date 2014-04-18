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
#include "operator_dynamic.h"
#include "operator_matrix.cuh"
#include "matrix_traits_thrust.h"
#include "creation.cuh"
#include "dx.cuh"
#include "dlt.cuh"

//#include "cusp_eigen.h"
//CAN'T BE TRANSVERSE SYMMETRIC?

/*! @file 

  Contains object for the polarisation matrix creation
  */
namespace dg
{

//as far as i see in the source code cusp only supports coo - coo matrix
//-matrix multiplication
///@cond DEV
template< class T, class Memory>
struct Polarisation
{
    typedef cusp::coo_matrix<int, T, Memory> Matrix;
    typedef cusp::array1d<T, Memory> Vector;
    Polarisation( const Grid1d<T>& g);
    Matrix create( const Vector& );
  private:
    typedef cusp::array1d<int, Memory> Array;
    Matrix left, middle, right, jump;
    cusp::array1d<int, Memory> I, J;
    Vector xspace;
    unsigned n, N;

};
template <class T, class Memory>
Polarisation<T, Memory>::Polarisation( const Grid1d<T>& g): I(g.size()), J(I), xspace( g.size()), n(g.n()), N(g.N())
{
    T h = g.h();
    bc bcx = g.bcx();
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());

    right = create::dx_plus_mt<T>( n, N, h, bcx); //create and transfer to device
    Operator<T> backward( g.dlt().backward());
    middle = tensor<T>( N, backward);
    cusp::multiply( middle, right, right);
    cusp::transpose( right, left); 
    Operator<T> weights(n,0);
    for( unsigned i=0; i<g.n(); i++)
        weights( i,i) = g.dlt().weights()[i];
    weights *= h/2.;
    middle = tensor( N, weights*backward);
    jump = create::jump_ot<T>( n, N, bcx); //without jump cg is unstable

}

template< class T, class Memory>
cusp::coo_matrix<int, T, Memory> Polarisation<T, Memory>::create( const Vector& chi)
{
    Matrix laplace, temp;
    cusp::multiply( middle, chi, xspace);
    cusp::coo_matrix_view<Array, Array, Vector,  int, T, Memory> chi_view( n*N, n*N, n*N, I, J, xspace);
    cusp::multiply( chi_view, right, laplace);
    cusp::multiply( left, laplace, temp);
    cusp::add( temp, jump, laplace);
    return laplace;
}

///@endcond
//////////////////////////////////////////////////////////////////////////////////
/**
 * @brief X-space version of polarisation term
 *
 * @ingroup creation
 * The term discretized is \f[ \nabla ( \chi \nabla ) \f]
 * @tparam container The vector class on which to operate on
 */
template< class container>
struct Polarisation2dX
{
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    typedef cusp::csr_matrix<int, value_type, MemorySpace> Matrix;
    //typedef cusp::coo_matrix<int, value_type, MemorySpace> Matrix;
    /**
     * @brief Create Polarisation on a grid 
     *
     * @param g The 2D grid
     */
    Polarisation2dX( const Grid2d<value_type>& grid, direction dir = forward);
    /**
     * @brief Create polarisation term on a grid using different boundary conditions
     *
     * @param g The 2D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    Polarisation2dX( const Grid2d<value_type>& grid, bc bcx, bc bcy, direction dir = forward);

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
    typedef cusp::array1d<value_type, MemorySpace> VArray;
    void construct( unsigned n, unsigned Nx, unsigned Ny, value_type hx, value_type hy, bc bcx, bc bcy, const DLT<value_type>&, direction dir  );
    Matrix leftx, lefty, rightx, righty, jump;
    Array I, J;
    typename Array::view I_view, J_view;
    container middle; //contain coeffs for chi multiplication
    container xchi;
    typename VArray::view xchi_view;
    //cusp::array1d_view< typename container::iterator> xchi_view;
    cusp::csr_matrix_view<typename Array::view, typename Array::view, typename VArray::view,  int, value_type, MemorySpace> xchi_matrix_view; //make view of thrust vector
    //cusp::coo_matrix_view<typename Array::view, typename Array::view, typename VArray::view,  int, value_type, MemorySpace> xchi_matrix_view; //make view of thrust vector
};

template <class container>
Polarisation2dX< container>::Polarisation2dX( const Grid2d<value_type>& g, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    middle( create::w2d(g) ), xchi(middle), xchi_view( xchi.begin(), xchi.end()),
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    construct( g.n(), g.Nx(), g.Ny(), g.hx(), g.hy(), g.bcx(), g.bcy(), g.dlt(), dir);
}
template <class container>
Polarisation2dX<container>::Polarisation2dX( const Grid2d<value_type>& g, bc bcx, bc bcy, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    middle( create::w2d(g) ), xchi(middle), xchi_view( xchi.begin(), xchi.end()),
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    construct( g.n(), g.Nx(), g.Ny(), g.hx(), g.hy(), bcx, bcy, g.dlt(), dir);
}

template <class container>
void Polarisation2dX<container>::construct( unsigned n, unsigned Nx, unsigned Ny, value_type hx, value_type hy, bc bcx, bc bcy, const DLT<value_type>& dlt, direction dir)
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> HMatrix;

    Operator<value_type> backward1d( dlt.backward());
    Operator<value_type> forward1d( dlt.forward());

    //create x and y derivative in xspace
    HMatrix rightx_, righty_;
    if( dir == forward)
        rightx_ = create::dx_plus_mt<value_type>( n, Nx, hx, bcx); 
    else
        rightx_ = create::dx_minus_mt<value_type>( n, Nx, hx, bcx); 

    rightx_ = sandwich( backward1d, rightx_, forward1d);
    rightx = dg::dgtensor( n, tensor(Ny, create::delta(n)), rightx_);

    if( dir == forward) 
        righty_ = create::dx_plus_mt( n, Ny, hy, bcy); //create and transfer to device
    else 
        righty_ = create::dx_minus_mt( n, Ny, hy, bcy); //create and transfer to device

    righty_ = sandwich( backward1d, righty_, forward1d);
    righty = dg::dgtensor( n, righty_, tensor( Nx, create::delta(n)) );

    cusp::transpose( rightx, leftx); 
    cusp::transpose( righty, lefty); 

    //create diagonal matrix entries for x_chi_view
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());

    //create norm for jump matrices 
    Operator<value_type> normx(n, 0.), normy(n, 0.);
    for( unsigned i=0; i<n; i++)
        normx( i,i) = normy( i,i) = dlt.weights()[i];
    normx *= hx/2.;
    normy *= hy/2.; // normalisation because F is invariant
    //create jump
    HMatrix jumpx = create::jump_ot<value_type>( n, Nx, bcx); //jump without t!
    jumpx = sandwich( forward1d.transpose(), jumpx, forward1d);
    jumpx = dg::dgtensor( n, tensor( Ny, normy), jumpx); //proper normalisation

    HMatrix jumpy = create::jump_ot<value_type>( n, Ny, bcy); //without jump cg is unstable
    jumpy = sandwich( forward1d.transpose(), jumpy, forward1d);
    jumpy = dg::dgtensor(n, jumpy, tensor( Nx, normx));
    HMatrix jump_;
    cusp::add( jumpx, jumpy, jump_); //does not respect sorting!!!
    jump_.sort_by_row_and_column();
    jump = jump_;
}

template< class container>
typename Polarisation2dX<container>::Matrix Polarisation2dX<container>::create( const container& chi)
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
