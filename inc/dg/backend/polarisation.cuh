#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>


#include "../blas.h"

#include "functions.h"
#include "grid.h"
#include "tensor.cuh"
#include "operator.h"
#include "operator_tensor.cuh"
#include "matrix_traits_thrust.h"
#include "creation.cuh"
#include "derivatives.cuh"
#include "dx.cuh"

#include "dlt.h"

/*! @file 

  Contains object for the polarisation matrix creation
  */
namespace dg
{


//////////////////////////////////////////////////////////////////////////////////
/**
 * @brief X-space version of polarisation term
 *
 * @ingroup highlevel
 * The term discretized is \f[ \nabla ( \chi \nabla ) \f]
 * @tparam container The vector class on which to operate on
 */
template< class container = thrust::host_vector<double> , class Matrix = cusp::csr_matrix<int, double, cusp::host_memory> >
struct Polarisation2dX
{
    typedef typename container::value_type value_type; //!< value type to be used
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace; //!< Memory Space
    //typedef cusp::csr_matrix<int, value_type, MemorySpace> Matrix;//!< CSR Matrix is the best for host computations
    //typedef cusp::coo_matrix<int, value_type, MemorySpace> Matrix;
    /**
     * @brief Create Polarisation on a grid 
     *
     * @param grid The 2D grid
     * @param dir The direction of the first derivative
     */
    Polarisation2dX( const Grid2d<value_type>& grid, direction dir = forward);
    /**
     * @brief Create polarisation term on a grid using different boundary conditions
     *
     * @param grid The 2D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     * @param dir The direction of the first derivative
     */
    Polarisation2dX( const Grid2d<value_type>& grid, bc bcx, bc bcy, direction dir = forward);
    /**
     * @brief Create Polarisation on a grid 
     *
     * @param grid The 3D grid
     * @param dir The direction of the first derivative
     */
    Polarisation2dX( const Grid3d<value_type>& grid, direction dir = forward);
    /**
     * @brief Create polarisation term on a grid using different boundary conditions
     *
     * @param g The 3D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    /**
     * @brief Create Polarisation on a grid 
     *
     * @param grid The 3D grid
     * @param bcx X-boundary condition
     * @param bcy Y-boundary condition
     * @param dir The direction of the first derivative
     */
    Polarisation2dX( const Grid3d<value_type>& grid, bc bcx, bc bcy, direction dir = forward);

    /**
     * @brief Create a unnormalized matrix for the polarisation term
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
    void construct( unsigned Nz, value_type hz);
    Matrix leftx, lefty, rightx, righty, jump;
    Array I, J;
    typename Array::view I_view, J_view;
    const container weights_, veights_; //contain coeffs for chi multiplication
    container xchi;
    typename VArray::view xchi_view;
    //cusp::array1d_view< typename container::iterator> xchi_view;
    cusp::csr_matrix_view<typename Array::view, typename Array::view, typename VArray::view,  int, value_type, MemorySpace> xchi_matrix_view; //make view of thrust vector
    //cusp::coo_matrix_view<typename Array::view, typename Array::view, typename VArray::view,  int, value_type, MemorySpace> xchi_matrix_view; //make view of thrust vector
};

template <class container, class Matrix>
Polarisation2dX< container, Matrix>::Polarisation2dX( const Grid2d<value_type>& g, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    weights_( create::w2d(g) ), xchi(weights_), xchi_view( xchi.begin(), xchi.end()),
    veights_( create::v2d(g) ), 
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    //construct( g.n(), g.Nx(), g.Ny(), g.hx(), g.hy(), g.bcx(), g.bcy(), g.dlt(), dir);
    rightx=dg::create::dx( g, normed, backward);
    righty=dg::create::dy( g, normed, backward);
    //leftx =dg::create::dx( g, backward, not_normed);
    //lefty =dg::create::dy( g, backward, not_normed);
    cusp::transpose( rightx, leftx); 
    cusp::transpose( righty, lefty); 
    //create diagonal matrix entries for x_chi_view
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());
    jump  =dg::create::jump2d( g);
}
template <class container, class Matrix>
Polarisation2dX<container, Matrix>::Polarisation2dX( const Grid2d<value_type>& g, bc bcx, bc bcy, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    weights_( create::w2d(g) ), xchi(weights_), xchi_view( xchi.begin(), xchi.end()),
    veights_( create::v2d(g) ),
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    //construct( g.n(), g.Nx(), g.Ny(), g.hx(), g.hy(), bcx, bcy, g.dlt(), dir);
    rightx=dg::create::dx( g, bcx, normed, backward);
    righty=dg::create::dy( g, bcy, normed, backward);
    //leftx =dg::create::dx( g,bcx,bcy backward, not_normed);
    //lefty =dg::create::dy( g,bcx,bcy backward, not_normed);
    cusp::transpose( rightx, leftx); 
    cusp::transpose( righty, lefty); 
    //create diagonal matrix entries for x_chi_view
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());
    jump  =dg::create::jump2d( g, bcx, bcy);
}
template <class container, class Matrix>
Polarisation2dX< container, Matrix>::Polarisation2dX( const Grid3d<value_type>& g, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    weights_( create::w3d(g) ), xchi(weights_), xchi_view( xchi.begin(), xchi.end()),
    veights_( create::v3d(g) ),
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    construct( g.n(), g.Nx(), g.Ny(), g.hx(), g.hy(), g.bcx(), g.bcy(), g.dlt(), dir);
    construct( g.Nz(), g.hz());
}
template <class container, class Matrix>
Polarisation2dX<container, Matrix>::Polarisation2dX( const Grid3d<value_type>& g, bc bcx, bc bcy, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()), 
    weights_( create::w3d(g) ), xchi(weights_), xchi_view( xchi.begin(), xchi.end()),
    veights_( create::v3d(g) ),
    xchi_matrix_view( xchi.size(), xchi.size(), xchi.size(), I_view, J_view, xchi_view)
{
    construct( g.n(), g.Nx(), g.Ny(), g.hx(), g.hy(), bcx, bcy, g.dlt(), dir);
    construct( g.Nz(), g.hz());
}

template <class container, class Matrix>
void Polarisation2dX<container, Matrix>::construct( unsigned n, unsigned Nx, unsigned Ny, value_type hx, value_type hy, bc bcx, bc bcy, const DLT<value_type>& dlt, direction dir)
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

    //create jump matrices
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

template <class container, class Matrix>
void Polarisation2dX<container, Matrix >::construct( unsigned Nz, value_type hz)
{
    Matrix temp; 
    temp = dgtensor<value_type>( 1, tensor<value_type>( Nz, create::delta(1)), rightx);
    rightx = temp;
    temp = dgtensor<value_type>( 1, tensor<value_type>( Nz, create::delta(1)), righty);
    righty = temp;
    temp = dgtensor<value_type>( 1, tensor<value_type>( Nz, create::delta(1)), leftx);
    leftx = temp;
    temp = dgtensor<value_type>( 1, tensor<value_type>( Nz, create::delta(1)), lefty);
    lefty = temp;
    temp = dgtensor<value_type>( 1, tensor<value_type>( Nz, hz*create::delta(1)), jump);
    jump = temp;
}

template< class container, class Matrix>
Matrix Polarisation2dX<container, Matrix>::create( const container& chi)
{

    Matrix temp1, temp2, temp3;
    blas1::pointwiseDot( weights_, chi, xchi);
    //multiply also does not necessarily keep the sorting
    cusp::multiply( xchi_matrix_view, rightx, temp1); //D_x*R_x
    cusp::multiply( xchi_matrix_view, righty, temp2); //D_y*R_y
    cusp::multiply( leftx, temp1, temp3); //L_x*D_x*R_x
    cusp::multiply( lefty, temp2, temp1); //L_y*D_y*R_y
    cusp::add( temp1, temp3, temp2);  // D_yy + D_xx
    cusp::add( temp2, jump, temp1); // Lap + Jump
   
    return temp1;
}

} //namespace dg

