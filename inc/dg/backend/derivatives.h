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
EllSparseBlockMat<double> dx( const aTopology2d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<double> dx;
    dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
    dx.left_size = g.n()*g.Ny();
    dx.set_default_range();
    return dx;
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
EllSparseBlockMat<double> dx( const aTopology2d& g, direction dir = centered) { return dx( g, g.bcx(), dir);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
EllSparseBlockMat<double> dy( const aTopology2d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<double> dy;
    dy = dx_normed( g.n(), g.Ny(), g.hy(), bcy, dir);
    dy.right_size = g.n()*g.Nx();
    dy.set_default_range();
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
EllSparseBlockMat<double> dy( const aTopology2d& g, direction dir = centered){ return dy( g, g.bcy(), dir);}

/**
 * @brief Matrix that contains 2d jump terms in X direction
 *
 * @param g grid
 * @param bcx boundary condition in x
 *
 * @return A host matrix 
 */
EllSparseBlockMat<double> jumpX( const aTopology2d& g, bc bcx)
{
    EllSparseBlockMat<double> jx;
    jx = jump( g.n(), g.Nx(), g.hx(), bcx);
    jx.left_size = g.n()*g.Ny();
    jx.set_default_range();
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
EllSparseBlockMat<double> jumpY( const aTopology2d& g, bc bcy)
{
    EllSparseBlockMat<double> jy;
    jy = jump( g.n(), g.Ny(), g.hy(), bcy);
    jy.right_size = g.n()*g.Nx();
    jy.set_default_range();
    return jy;
}

/**
 * @brief Matrix that contains 2d jump terms in X direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix 
 */
EllSparseBlockMat<double> jumpX( const aTopology2d& g)
{
    return jumpX( g, g.bcx());
}

/**
 * @brief Matrix that contains 2d jump terms in Y direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix 
 */
EllSparseBlockMat<double> jumpY( const aTopology2d& g)
{
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
EllSparseBlockMat<double> jumpX( const aTopology3d& g, bc bcx)
{
    EllSparseBlockMat<double> jx;
    jx = jump( g.n(), g.Nx(), g.hx(), bcx);
    jx.left_size = g.n()*g.Ny()*g.Nz();
    jx.set_default_range();
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
EllSparseBlockMat<double> jumpY( const aTopology3d& g, bc bcy)
{
    EllSparseBlockMat<double> jy;
    jy = jump( g.n(), g.Ny(), g.hy(), bcy);
    jy.right_size = g.n()*g.Nx();
    jy.left_size = g.Nz();
    jy.set_default_range();
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
EllSparseBlockMat<double> jumpZ( const aTopology3d& g, bc bcz)
{
    EllSparseBlockMat<double> jz;
    jz = jump( 1, g.Nz(), g.hz(), bcz);
    jz.right_size = g.n()*g.Nx()*g.n()*g.Ny();
    jz.set_default_range();
    return jz;
}

/**
 * @brief Matrix that contains 3d jump terms in X direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
EllSparseBlockMat<double> jumpX( const aTopology3d& g)
{
    return jumpX( g, g.bcx());
}

/**
 * @brief Matrix that contains 3d jump terms in Y direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
EllSparseBlockMat<double> jumpY( const aTopology3d& g)
{
    return jumpY( g, g.bcy());
}

/**
 * @brief Matrix that contains 3d jump terms in Z direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
EllSparseBlockMat<double> jumpZ( const aTopology3d& g)
{
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
EllSparseBlockMat<double> dx( const aTopology3d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<double> dx;
    dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
    dx.left_size = g.n()*g.Ny()*g.Nz();
    dx.set_default_range();
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
EllSparseBlockMat<double> dx( const aTopology3d& g, direction dir = centered) { return dx( g, g.bcx(), dir);}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
EllSparseBlockMat<double> dy( const aTopology3d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<double> dy;
    dy = dx_normed( g.n(), g.Ny(), g.hy(), bcy, dir);
    dy.right_size = g.n()*g.Nx();
    dy.left_size = g.Nz();
    dy.set_default_range();
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
EllSparseBlockMat<double> dy( const aTopology3d& g, direction dir = centered){ return dy( g, g.bcy(), dir);}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @param g The grid on which to create dz
 * @param bcz The boundary condition
 * @param dir The direction of the stencil
 *
 * @return A host matrix 
 */
EllSparseBlockMat<double> dz( const aTopology3d& g, bc bcz, direction dir = centered)
{
    EllSparseBlockMat<double> dz;
    dz = dx_normed( 1, g.Nz(), g.hz(), bcz, dir);
    dz.right_size = g.n()*g.n()*g.Nx()*g.Ny();
    dz.set_default_range();
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
EllSparseBlockMat<double> dz( const aTopology3d& g, direction dir = centered){ return dz( g, g.bcz(), dir);}



///@}

} //namespace create

} //namespace dg

