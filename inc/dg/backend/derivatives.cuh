#ifndef _DG_DERIVATIVES_CUH_
#define _DG_DERIVATIVES_CUH_

#include <cusp/elementwise.h>

#include "grid.h"
#include "creation.cuh"
#include "dx.cuh"
#include "functions.h"
#include "laplace.cuh"
#include "operator_tensor.cuh"
#include "tensor.cuh"

/*! @file 
  
  Convenience functions to create 2D derivatives
  */
namespace dg{


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
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid2d<T>& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Matrix dx;
    if( dir == symmetric)
        dx = create::dx_symm<T>(g.n(), g.Nx(), g.hx(), bcx);
    else if (dir == forward)
        dx = create::dx_plus_mt<T>(g.n(), g.Nx(), g.hx(), bcx);
    else if (dir == backward)
        dx = create::dx_minus_mt<T>(g.n(), g.Nx(), g.hx(), bcx);
    Matrix bdxf( dx);

    //norm 1 x b*dx*f or w x w*b*dx*f
    Operator<T> backward=g.dlt().backward();
    Operator<T> normx(g.n(), 0.), normy(g.n(), 0.);
    if( no == not_normed)
    {
        for( unsigned i=0; i<g.n(); i++)
            normx( i,i) = normy( i,i) = g.dlt().weights()[i];
        normx *= g.hx()/2.;
        normy *= g.hy()/2.; // normalisation because F is invariant
    }
    else
    {
        normx = normy = create::delta(g.n());
    }
    bdxf = sandwich<T>( normx*backward, dx, g.dlt().forward());
    return dgtensor<T>( g.n(), tensor<T>( g.Ny(), normy ), bdxf );
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
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid2d<T>& g, norm no = normed, direction dir = symmetric) { return dx( g, g.bcx(), no, dir);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid2d<T>& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Matrix dy;
    if( dir == symmetric)
        dy = create::dx_symm<T>(g.n(), g.Ny(), g.hy(), bcy);
    else if (dir == forward)
        dy = create::dx_plus_mt<T>(g.n(), g.Ny(), g.hy(), bcy);
    else if (dir == backward)
        dy = create::dx_minus_mt<T>(g.n(), g.Ny(), g.hy(), bcy);

    Matrix bdyf(dy);
    //norm b*dy*f x 1 or w*b*dy*f x 1
    Operator<T> backward=g.dlt().backward();
    Operator<T> normx(g.n(), 0.), normy(g.n(), 0.);
    if( no == not_normed)
    {
        for( unsigned i=0; i<g.n(); i++)
            normx( i,i) = normy( i,i) = g.dlt().weights()[i];
        normx *= g.hx()/2.;
        normy *= g.hy()/2.; // normalisation because F is invariant
    }
    else
    {
        normx = normy = create::delta(g.n());
    }
    bdyf = sandwich<T>( normy*backward, dy, g.dlt().forward());
    return dgtensor<T>( g.n(), bdyf, tensor<T>( g.Nx(), normx ) );
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
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid2d<T>& g, norm no = normed, direction dir = symmetric){ return dy( g, g.bcy(), no, dir);}

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
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplacianM( const Grid2d<T>& g, bc bcx, bc bcy, norm no = normed, direction dir = forward)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;

    Grid1d<T> gy( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    Matrix ly = create::laplace1d( gy, no, dir);
    
    Grid1d<T> gx( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    Matrix lx = create:: laplace1d( gx, no, dir);

    Matrix flxf(lx), flyf(ly);
    //sandwich with correctly normalized matrices
    Operator<T> forward1d( g.dlt().forward( ));
    Operator<T> backward1d( g.dlt().backward( ));
    Operator<T> leftx( backward1d ), lefty( backward1d);
    if( no == not_normed)
        leftx = lefty = forward1d.transpose();

    flxf = sandwich<T>( leftx, lx, forward1d);
    flyf = sandwich<T>( lefty, ly, forward1d);
    Operator<T> normx(g.n(), 0.), normy( g.n(), 0.);

    //generate norm (w1d or s1d)
    if( no == not_normed) 
    {
        for( unsigned i=0; i<g.n(); i++)
            normx( i,i) = normy( i,i) = g.dlt().weights()[i];
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
 * @param g The grid on which to operate (boundary conditions are taken from here)
 * @param no use normed if you want to compute e.g. diffusive terms, 
             use not_normed if you want to solve symmetric matrix equations (T resp. V is missing)
 * @param s The space on which the matrix operates on
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplacianM( const Grid2d<T>& g, norm no = normed, direction dir = forward)
{
    return laplacianM( g, g.bcx(), g.bcy(), no, dir);
}

template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> jump2d( const Grid2d<T>& g, bc bcx, bc bcy)
{
    const unsigned& n = g.n();
    Operator<T> normx(n, 0.), normy(n, 0.);
    for( unsigned i=0; i<n; i++)
        normx( i,i) = normy( i,i) = g.dlt().weights()[i];
    normx *= g.hx()/2.;
    normy *= g.hy()/2.; // normalisation because F is invariant
    typedef cusp::coo_matrix<int, T, cusp::host_memory> HMatrix;
    HMatrix jumpx = create::jump_ot<T>( n, g.Nx(), bcx); //jump without t!
    Operator<T> forward1d( g.dlt().forward( ));
    jumpx = sandwich( forward1d.transpose(), jumpx, forward1d);
    jumpx = dg::dgtensor( n, tensor( g.Ny(), normy), jumpx); //proper normalisation

    HMatrix jumpy = create::jump_ot<T>( n, g.Ny(), bcy); //without jump cg is unstable
    jumpy = sandwich( forward1d.transpose(), jumpy, forward1d);
    jumpy = dg::dgtensor(n, jumpy, tensor( g.Nx(), normx));
    HMatrix jump_;
    cusp::add( jumpx, jumpy, jump_); //does not respect sorting!!!
    jump_.sort_by_row_and_column();
    return jump_;
}
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> jump2d( const Grid2d<T>& g)
{
    return jump( g, g.bcx(), g.bcy());
}

///////////////////////////////////////////3D VERSIONS//////////////////////
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> jump2d( const Grid3d<T>& g, bc bcx, bc bcy)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Grid2d<T> g2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny());
    Matrix jump_ = create::jump( g2d, bcx, bcy);
    return dgtensor<T>( 1, tensor<T>( g.Nz(), g.hz()*delta(1)), jump); //w*hz/2 = hz
}
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> jump2d( const Grid3d<T>& g)
{
    return jump( g, g.bcx(), g.bcy());
}
/**
 * @brief Create 3d derivative in x-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid3d<T>& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Grid2d<T> g2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny());
    Matrix dx = create::dx( g2d, bcx, no, dir);

    if( no == normed)
        return dgtensor<T>( 1, tensor<T>( g.Nz(), delta(1) ), dx );
    else 
        return dgtensor<T>( 1, tensor<T>( g.Nz(), g.hz()*delta(1)) , dx); //w*hz/2 = hz
}

/**
 * @brief Create 3d derivative in x-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid3d<T>& g, norm no = normed, direction dir = symmetric) { return dx( g, g.bcx(), no, dir);}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid3d<T>& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Grid2d<T> g2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny());
    Matrix dy = create::dy( g2d, bcy, no, dir);

    if( no == normed)
        return dgtensor<T>( 1, tensor<T>( g.Nz(), delta(1) ), dy );
    else 
        return dgtensor<T>( 1, tensor<T>( g.Nz(), g.hz()*delta(1)) , dy); //w*hz/2 = hz
}
/**
 * @brief Create 3d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param s The space on which the matrix operates on
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid3d<T>& g, norm no = normed, direction dir = symmetric){ return dy( g, g.bcy(),no, dir);}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dz
 * @param bcz The boundary condition
 * @param dir The direction of the stencil
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dz( const Grid3d<T>& g, bc bcz, direction dir)
{
    //dasselbe wie dy in 2D: 
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Matrix dz; 
    if( dir == forward) 
        dz = create::dx_plus_mt<T>(1, g.Nz(), g.hz(), bcz);
    else if( dir == backward) 
        dz = create::dx_minus_mt<T>(1, g.Nz(), g.hz(), bcz);
    else
        dz = create::dx_symm<T>(1, g.Nz(), g.hz(), bcz);
    return dgtensor<T>( 1, dz,  tensor<T>( g.Nx()*g.Ny(), delta(g.n()*g.n()) ));
}
/**
 * @brief Create 3d derivative in z-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy (boundary condition is taken from here)
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> dz( const Grid3d<T>& g){ return dz( g, g.bcz(), symmetric);}

/**
 * @brief Create 3d negative laplacian_perp
 *
 * \f[ -\Delta = -(\partial_x^2 + \partial_y^2) \f]
 * @tparam T value-type
 * @param g The grid on which to operate
 * @param bcx Boundary condition in x
 * @param bcy Boundary condition in y
 * @param no use normed if you want to compute e.g. diffusive terms,
             use not_normed if you want to solve symmetric matrix equations (T resp. V is missing)
 * @param s The space on which the matrix operates on
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplacianM_perp( const Grid3d<T>& g, bc bcx, bc bcy, norm no = normed, direction dir = forward)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Grid2d<T> g2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny());
    Matrix laplace2d = create::laplacianM( g2d, bcx, bcy, no, dir);
    if( no == normed)
        return dgtensor<T>( 1, tensor<T>( g.Nz(), delta(1) ), laplace2d );
    else 
        return dgtensor<T>( 1, tensor<T>( g.Nz(), g.hz()*delta(1)), laplace2d); //w*hz/2 = hz

}

/**
 * @brief Create 3d negative perpendicular laplacian
 *
 * \f[ -\Delta = -(\partial_x^2 + \partial_y^2) \f]
 * @tparam T value-type
 * @param g The grid on which to operate (boundary conditions are taken from here)
 * @param no use normed if you want to compute e.g. diffusive terms, 
             use not_normed if you want to solve symmetric matrix equations (T resp. V is missing)
 * @param s The space on which the matrix operates on
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplacianM_perp( const Grid3d<T>& g, norm no = normed, direction dir = forward)
{
    return laplacianM_perp( g, g.bcx(), g.bcy(), no, dir);
}

/**
 * @brief Create normed 3d negative parallel laplacian in XSPACE
 *
 * \f[ -\Delta = -\partial_z^2  \f]
 * @tparam T value-type
 * @param g The grid on which to operate 
 * @param bcz boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplacianM_parallel( const Grid3d<T>& g, bc bcz, direction dir = forward)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Grid1d<T> g1d( g.z0(), g.z1(), 1, g.Nz(), bcz);
    Matrix lz = create::laplace1d( g1d, normed, dir);

    return dgtensor<T>( 1, lz, tensor<T>( g.Nx()*g.Ny(), delta(g.n()*g.n())));

}
///@}

} //namespace create

} //namespace dg
#endif//_DG_DERIVATIVES_CUH_
