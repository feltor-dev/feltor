#ifndef _DG_DERIVATIVES_CUH_
#define _DG_DERIVATIVES_CUH_

#include <cusp/elementwise.h>

#include "grid.cuh"
#include "creation.cuh"
#include "dx.cuh"
#include "functions.h"
#include "laplace.cuh"
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
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T>& g, bc bcx, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Matrix dx = create::dx_symm<T>(g.n(), g.Nx(), g.hx(), bcx);
    Matrix bdxf( dx);
    if( s == XSPACE)
        bdxf = sandwich<T>( g.dlt().backward(), dx, g.dlt().forward());

    return dgtensor<T>( g.n(), tensor<T>( g.Ny(), delta(g.n()) ), bdxf );
}
/**
 * @brief Create 2d derivative in x-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T>& g, space s = XSPACE) { return dx( g, g.bcx(), s);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy
 * @param bcx The boundary condition
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T>& g, bc bcy, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Matrix dy = create::dx_symm<T>( g.n(), g.Ny(), g.hy(), bcy);
    Matrix bdyf_(dy);
    if( s == XSPACE)
        bdyf_ = sandwich<T>(g.dlt().backward(), dy, g.dlt().forward());

    return dgtensor<T>( g.n(), bdyf_, tensor<T>( g.Nx(), delta(g.n())));
}
/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T>& g, space s = XSPACE){ return dy( g, g.bcy(), s);}

//the behaviour of CG is completely the same in xspace as in lspace
/**
 * @brief Create 2d negative laplacian
 *
 * \f[ -\Delta = -(\partial_x^2 + \partial_y^2) \f]
 * @tparam T value-type
 * @param g The grid on which to operate
 * @param bcx Boundary condition in x
 * @param bcy Boundary condition in y
 * @param no use normed if you want to compute e.g. diffusive terms,
             use not_normed if you want to solve symmetric matrix equations (T resp. V is missing)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplacianM( const Grid<T>& g, bc bcx, bc bcy, norm no = normed, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;

    Grid1d<T> gy( g.y0(), g.y1(), g.n(), g.Ny(), g.bcy());
    Matrix ly = create::laplace1d( gy, no);
    
    //if( bcy == PER) {
    //    ly = create::laplace1d_per<double>( g.n(), g.Ny(), g.hy(), no);
    //} else if( bcy == DIR) {
    //    ly = create::laplace1d_dir<double>( g.n(), g.Ny(), g.hy(), no);
    //}
    Grid1d<T> gx( g.x0(), g.x1(), g.n(), g.Nx(), g.bcx());
    Matrix lx = create:: laplace1d( gx, no);
    //if( bcx == PER) {
    //    lx = create::laplace1d_per<double>( g.n(), g.Nx(), g.hx(), no);
    //}else if( bcx == DIR) {
    //    lx = create::laplace1d_dir<double>( g.n(), g.Nx(), g.hx(), no);
    //}

    Matrix flxf(lx), flyf(ly);
    //sandwich with correctly normalized matrices
    if( s == XSPACE)
    {
        Operator<T> forward1d( g.dlt().forward( ));
        Operator<T> backward1d( g.dlt().backward( ));
        Operator<T> leftx( backward1d ), lefty( backward1d);
        if( no == not_normed)
            leftx = lefty = forward1d.transpose();

        flxf = sandwich<T>( leftx, lx, forward1d);
        flyf = sandwich<T>( lefty, ly, forward1d);
    }
    Operator<T> normx(g.n(), 0.), normy( g.n(), 0.);

    //generate norm
    if( no == not_normed) 
    {
        if( s==XSPACE)
        {
            for( unsigned i=0; i<g.n(); i++)
                normx( i,i) = normy( i,i) = g.dlt().weights()[i];
        } else {
            normx = normy = create::pipj( g.n());
        }
        normx *= g.hx()/2.;
        normy *= g.hy()/2.;
    }
    else
        normx = normy = create::delta(g.n());

    Matrix ddyy = dgtensor<double>( g.n(), flyf, tensor( g.Nx(), normx));
    Matrix ddxx = dgtensor<double>( g.n(), tensor(g.Ny(), normy), flxf);
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
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplacianM( const Grid<T>& g, norm no = normed, space s = XSPACE)
{
    return laplacianM( g, g.bcx(), g.bcy(), no, s);
}
///@}

} //namespace create

} //namespace dg
#endif//_DG_DERIVATIVES_CUH_
