#ifndef _DG_DERIVATIVES_CUH_
#define _DG_DERIVATIVES_CUH_

#include <cusp/elementwise.h>

#include "grid.cuh"
#include "dlt.h"
#include "creation.cuh"
#include "dx.cuh"
#include "functions.h"
#include "laplace.cuh"
#include "operator.cuh"
#include "operator_matrix.cuh"
#include "tensor.cuh"

/*! @file 
  
  Convenience functions to create 2D derivatives
  */
namespace dg{

///@addtogroup creation
///@{
/**
 * @brief Switch between x-space and l-space
 */
enum space {
    XSPACE, //!< indicates, that the given matrix operates on x-space values
    LSPACE  //!< indicates, that the given matrix operates on l-space values
};
///@}

/**
 * @brief Contains functions used for matrix creation
 */
namespace create{

///@addtogroup highlevel
///@{


/**
 * @brief Create 2d derivative in x-direction
 *
 * @tparam T value-type
 * @tparam n # of Legendre coefficients 
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T, n>& g, bc bcx, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Matrix dx = create::dx_symm<T,n>( g.Nx(), g.hx(), bcx);
    Matrix bdxf( dx);
    if( s == XSPACE)
        bdxf = sandwich<T,n>( dx);

    return dgtensor<T,n>( tensor<T,n>( g.Ny(), delta), bdxf );
}
/**
 * @brief Create 2d derivative in x-direction
 *
 * @tparam T value-type
 * @tparam n # of Legendre coefficients 
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T, n>& g, space s = XSPACE) { return dx( g, g.bcx(), s);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @tparam n # of Legendre coefficients 
 * @param g The grid on which to create dy
 * @param bcx The boundary condition
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g, bc bcy, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Matrix dy = create::dx_symm<T,n>( g.Ny(), g.hy(), bcy);
    Matrix bdyf_(dy);
    if( s == XSPACE)
        bdyf_ = sandwich<T,n>( dy);

    return dgtensor<T,n>( bdyf_, tensor<T,n>( g.Nx(), delta));
}
/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @tparam n # of Legendre coefficients 
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g, space s = XSPACE){ return dy( g, g.bcy(), s);}

//the behaviour of CG is completely the same in xspace as in lspace
/**
 * @brief Create 2d negative laplacian
 *
 * \f[ -\Delta = -(\partial_x^2 + \partial_y^2) \f]
 * @tparam T value-type
 * @tparam n # of Legendre coefficients 
 * @param g The grid on which to operate
 * @param bcx Boundary condition in x
 * @param bcy Boundary condition in y
 * @param no use normed if you want to compute e.g. diffusive terms,
             use not_normed if you want to solve symmetric matrix equations (T resp. V is missing)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplacian( const Grid<T, n>& g, bc bcx, bc bcy, norm no = normed, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;

    Matrix ly;
    if( bcy == PER) {
        ly = create::laplace1d_per<double,  n>( g.Ny(), g.hy(), no);
    } else if( bcy == DIR) {
        ly = create::laplace1d_dir<double,  n>( g.Ny(), g.hy(), no);
    }
    Matrix lx;
    if( bcx == PER) {
        lx = create::laplace1d_per<double,  n>( g.Nx(), g.hx(), no);
    }else if( bcx == DIR) {
        lx = create::laplace1d_dir<double,  n>( g.Nx(), g.hx(), no);
    }

    Matrix flxf(lx), flyf(ly);
    //sandwich with correctly normalized matrices
    if( s == XSPACE)
    {
        Operator<T, n> forward1d( DLT<n>::forward);
        Operator<T, n> backward1d( DLT<n>::backward);
        Operator<T,n> leftx( backward1d ), lefty( backward1d);
        if( no == not_normed)
            leftx = lefty = forward1d.transpose();

        flxf = sandwich<T,n>( leftx, lx, forward1d);
        flyf = sandwich<T,n>( lefty, ly, forward1d);
    }
    Operator<T,n> normx(0.), normy(0.);
    dg::W1D<T,n> w1dx( g.hx()), w1dy( g.hy());
    dg::S1D<T,n> s1dx( g.hx()), s1dy( g.hy());
    //generate norm
    for( unsigned i=0; i<n; i++)
    {
        if( no == not_normed) 
        {
            if( s==XSPACE)
            {
                normx(i,i) = w1dx(i);
                normy(i,i) = w1dy(i);
            } else {
                normx(i,i) = s1dx(i);
                normy(i,i) = s1dy(i);
            }
        }
        else
            normx(i,i) = normy(i,i) = 1.;
    }

    Matrix ddyy = dgtensor<double, n>( flyf, tensor( g.Nx(), normx));
    Matrix ddxx = dgtensor<double, n>( tensor(g.Ny(), normy), flxf);
    Matrix laplace;
    cusp::add( ddxx, ddyy, laplace); //cusp add does not sort output!!!!
    laplace.sort_by_row_and_column();
    //std::cout << "Is sorted? "<<laplace.is_sorted_by_row_and_column()<<"\n";
    return laplace;
}

/**
 * @brief Create 2d negative laplacian
 *
 * \f[ -\Delta = -(\partial_x^2 + \partial_y^2) \f]
 * @tparam T value-type
 * @tparam n # of Legendre coefficients 
 * @param g The grid on which to operate (boundary conditions are taken from here)
 * @param no use normed if you want to compute e.g. diffusive terms, 
             use not_normed if you want to solve symmetric matrix equations (T resp. V is missing)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplacian( const Grid<T, n>& g, norm no = normed, space s = XSPACE)
{
    return laplacian( g, g.bcx(), g.bcy(), no, s);
}
///@}

} //namespace create

} //namespace dg
#endif//_DG_DERIVATIVES_CUH_
