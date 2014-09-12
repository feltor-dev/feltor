#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>


#include "../blas.h"
#include "../enums.h"

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

/**
 * @brief X-space version of polarisation term
 *
 * @ingroup highlevel
 * The term discretized is \f[ -\nabla ( \chi \nabla ) \f]
 * @tparam container The vector class on which to operate on
 */
template< class container = thrust::host_vector<double> , class Matrix = cusp::csr_matrix<int, double, cusp::host_memory> >
struct Polarisation2dX
{
    typedef typename container::value_type value_type; //!< value type to be used
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
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace; 
    typedef cusp::array1d<int, MemorySpace> Array;
    Matrix leftx, lefty, rightx, righty, jump;
    Array I, J;
    typename Array::view I_view, J_view;
};

template< class container, class Matrix>
Matrix Polarisation2dX<container, Matrix>::create( const container& chi)
{
    typedef cusp::array1d<value_type, MemorySpace> VArray;
    typename VArray::const_view chi_view( chi.begin(), chi.end());

    cusp::csr_matrix_view<typename Array::view, typename Array::view, 
            typename VArray::const_view, int, value_type, MemorySpace> 
        chi_matrix_view( chi.size(), chi.size(), chi.size(), 
                         I_view, J_view, chi_view);
    Matrix temp1, temp2, temp3;
    //blas1::pointwiseDot( weights_, chi, xchi);
    //multiply also does not necessarily keep the sorting
    cusp::multiply( chi_matrix_view, rightx, temp1); //D_x*R_x
    cusp::multiply( chi_matrix_view, righty, temp2); //D_y*R_y
    cusp::multiply( leftx, temp1, temp3); //L_x*D_x*R_x
    cusp::multiply( lefty, temp2, temp1); //L_y*D_y*R_y
    cusp::add( temp1, temp3, temp2);  // D_yy + D_xx
    cusp::add( temp2, jump, temp1); // Lap + Jump
   
    return temp1;
}

//////////////////////////////////////////constructors///////////////////////
template <class container, class Matrix>
Polarisation2dX< container, Matrix>::Polarisation2dX( const Grid2d<value_type>& g, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end())
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> Mat;
    rightx=dg::create::dx( g, g.bcx(), normed, forward);
    righty=dg::create::dy( g, g.bcy(), normed, forward);
    Mat x =dg::create::dx( g, create::detail::inverse(g.bcx()), not_normed, backward);
    Mat y =dg::create::dy( g, create::detail::inverse(g.bcy()), not_normed, backward);
    cusp::blas::scal( x.values, -1.);
    cusp::blas::scal( y.values, -1.);
    leftx = x, lefty = y;
    //create diagonal matrix entries for x_chi_view
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());
    jump  =dg::create::jump2d( g, g.bcx(), g.bcy(), not_normed);
}
template <class container, class Matrix>
Polarisation2dX<container, Matrix>::Polarisation2dX( const Grid2d<value_type>& g, bc bcx, bc bcy, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()) 
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> Mat;
    rightx=dg::create::dx( g, bcx, normed, forward);
    righty=dg::create::dy( g, bcy, normed, forward);
    Mat x =dg::create::dx( g, create::detail::inverse(bcx), not_normed, forward);
    Mat y =dg::create::dy( g, create::detail::inverse(bcy), not_normed, forward);
    cusp::blas::scal( x.values, -1.);
    cusp::blas::scal( y.values, -1.);
    leftx = x, lefty = y;
    //create diagonal matrix entries for x_chi_view
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());
    jump  =dg::create::jump2d( g, bcx, bcy, not_normed);
}
template <class container, class Matrix>
Polarisation2dX< container, Matrix>::Polarisation2dX( const Grid3d<value_type>& g, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()) 
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> Mat;
    rightx=dg::create::dx( g, g.bcx(), normed, forward);
    righty=dg::create::dy( g, g.bcy(), normed, forward);
    Mat x =dg::create::dx( g, create::detail::inverse(g.bcx()), not_normed, backward);
    Mat y =dg::create::dy( g, create::detail::inverse(g.bcy()), not_normed, backward);
    cusp::blas::scal( x.values, -1.);
    cusp::blas::scal( y.values, -1.);
    leftx = x, lefty = y;
    //create diagonal matrix entries for x_chi_view
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());
    jump  =dg::create::jump2d( g, g.bcx(), g.bcy(), not_normed);
}
template <class container, class Matrix>
Polarisation2dX<container, Matrix>::Polarisation2dX( const Grid3d<value_type>& g, bc bcx, bc bcy, direction dir):
    I(g.size()+1), J(I), I_view( I.begin(), I.end()), J_view( J.begin(), J.end()) 
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> Mat;
    rightx=dg::create::dx( g, bcx, normed, forward);
    righty=dg::create::dy( g, bcy, normed, forward);
    Mat x =dg::create::dx( g, create::detail::inverse(bcx), not_normed, backward);
    Mat y =dg::create::dy( g, create::detail::inverse(bcy), not_normed, backward);
    cusp::blas::scal( x.values, -1.);
    cusp::blas::scal( y.values, -1.);
    leftx = x, lefty = y;
    //create diagonal matrix entries for x_chi_view
    thrust::sequence( I.begin(), I.end());
    thrust::sequence( J.begin(), J.end());
    jump  =dg::create::jump2d( g, bcx, bcy, not_normed);
}



} //namespace dg

