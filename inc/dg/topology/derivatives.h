#pragma once

#include "grid.h"
#include "dx.h"

/*! @file
  @brief Convenience functions to create 2D derivatives
  */

namespace dg{


/**
 * @brief Contains functions used for matrix creation
 */
namespace create{

///@addtogroup creation
///@{

//dx, dy, jumpX, jumpY

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology2d<real_type>& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<real_type> dx;
    dx = dx_normed( g.nx(), g.Nx(), g.hx(), bcx, dir);
    dx.set_left_size( g.ny()*g.Ny());
    return dx;
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 * @copydoc hide_code_blas2_symv
 */
template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology2d<real_type>& g, direction dir = centered) {
    return dx( g, g.bcx(), dir);
}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology2d<real_type>& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<real_type> dy;
    dy = dx_normed( g.ny(), g.Ny(), g.hy(), bcy, dir);
    dy.set_right_size( g.nx()*g.Nx());
    return dy;
}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology2d<real_type>& g, direction dir = centered){
    return dy( g, g.bcy(), dir);
}

/**
 * @brief Matrix that contains 2d jump terms in X direction
 *
 * @param g grid
 * @param bcx boundary condition in x
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology2d<real_type>& g, bc bcx)
{
    EllSparseBlockMat<real_type> jx;
    jx = jump( g.nx(), g.Nx(), g.hx(), bcx);
    jx.set_left_size( g.ny()*g.Ny());
    return jx;
}

/**
 * @brief Matrix that contains 2d jump terms in Y direction
 *
 * @param g grid
 * @param bcy boundary condition in y
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology2d<real_type>& g, bc bcy)
{
    EllSparseBlockMat<real_type> jy;
    jy = jump( g.ny(), g.Ny(), g.hy(), bcy);
    jy.set_right_size( g.nx()*g.Nx());
    return jy;
}

/**
 * @brief Matrix that contains 2d jump terms in X direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology2d<real_type>& g) {
    return jumpX( g, g.bcx());
}

/**
 * @brief Matrix that contains 2d jump terms in Y direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology2d<real_type>& g) {
    return jumpY( g, g.bcy());
}

///////////////////////////////////////////3D VERSIONS//////////////////////
//jumpX, jumpY, jumpZ, dx, dy, dz
/**
 * @brief Matrix that contains jump terms in X direction in 3D
 *
 * @param g The 3D grid
 * @param bcx boundary condition in x
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology3d<real_type>& g, bc bcx)
{
    EllSparseBlockMat<real_type> jx;
    jx = jump( g.nx(), g.Nx(), g.hx(), bcx);
    jx.set_left_size( g.ny()*g.Ny()*g.nz()*g.Nz());
    return jx;
}

/**
 * @brief Matrix that contains jump terms in Y direction in 3D
 *
 * @param g The 3D grid
 * @param bcy boundary condition in y
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology3d<real_type>& g, bc bcy)
{
    EllSparseBlockMat<real_type> jy;
    jy = jump( g.ny(), g.Ny(), g.hy(), bcy);
    jy.set_right_size( g.nx()*g.Nx());
    jy.set_left_size( g.nz()*g.Nz());
    return jy;
}

/**
 * @brief Matrix that contains jump terms in Z direction in 3D
 *
 * @param g The 3D grid
 * @param bcz boundary condition in z
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpZ( const aRealTopology3d<real_type>& g, bc bcz)
{
    EllSparseBlockMat<real_type> jz;
    jz = jump( g.nz(), g.Nz(), g.hz(), bcz);
    jz.set_right_size( g.nx()*g.Nx()*g.ny()*g.Ny());
    return jz;
}

/**
 * @brief Matrix that contains 3d jump terms in X direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology3d<real_type>& g) {
    return jumpX( g, g.bcx());
}

/**
 * @brief Matrix that contains 3d jump terms in Y direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology3d<real_type>& g) {
    return jumpY( g, g.bcy());
}

/**
 * @brief Matrix that contains 3d jump terms in Z direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> jumpZ( const aRealTopology3d<real_type>& g) {
    return jumpZ( g, g.bcz());
}


/**
 * @brief Create 3d derivative in x-direction
 *
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology3d<real_type>& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<real_type> dx;
    dx = dx_normed( g.nx(), g.Nx(), g.hx(), bcx, dir);
    dx.set_left_size( g.ny()*g.Ny()*g.nz()*g.Nz());
    return dx;
}

/**
 * @brief Create 3d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology3d<real_type>& g, direction dir = centered) {
    return dx( g, g.bcx(), dir);
}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology3d<real_type>& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<real_type> dy;
    dy = dx_normed( g.ny(), g.Ny(), g.hy(), bcy, dir);
    dy.set_right_size( g.nx()*g.Nx());
    dy.set_left_size( g.nz()*g.Nz());
    return dy;
}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology3d<real_type>& g, direction dir = centered){
    return dy( g, g.bcy(), dir);
}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @param g The grid on which to create dz
 * @param bcz The boundary condition
 * @param dir The direction of the stencil
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dz( const aRealTopology3d<real_type>& g, bc bcz, direction dir = centered)
{
    EllSparseBlockMat<real_type> dz;
    dz = dx_normed( g.nz(), g.Nz(), g.hz(), bcz, dir);
    dz.set_right_size( g.nx()*g.ny()*g.Nx()*g.Ny());
    return dz;

}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @param g The grid on which to create dz (boundary condition is taken from here)
 * @param dir The direction of the stencil
 *
 * @return A host matrix
 */
template<class real_type>
EllSparseBlockMat<real_type> dz( const aRealTopology3d<real_type>& g, direction dir = centered){
    return dz( g, g.bcz(), dir);
}



///@}

} //namespace create

} //namespace dg

