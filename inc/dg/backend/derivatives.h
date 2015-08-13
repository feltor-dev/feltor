#ifndef _DG_DERIVATIVES_CUH_
#define _DG_DERIVATIVES_CUH_

#include "grid.h"
#include "dx.h"

//create mpi derivatives by creating the whole matrix on the global grid
//then cut out the corresponding rows (depending on process grid structure) and map global indices to local ones 
//deprecate not_normed

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
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */

SparseBlockMat dx( const Grid2d<double>& g, bc bcx, direction dir = centered)
{
    SparseBlockMat dx;
    dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
    dx.left = g.n()*g.Ny();
    return dx;
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dx( const Grid2d<double>& g, direction dir = centered) { return dx( g, g.bcx(), dir);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid2d<double>& g, bc bcy, direction dir = centered)
{
    SparseBlockMat dy;
    dy = dx_normed( g.n(), g.Ny(), g.hy(), bcy, dir);
    dy.right = g.n()*g.Nx();
    return dy;
}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid2d<double>& g, direction dir = centered){ return dy( g, g.bcy(), dir);}

/**
 * @brief Matrix that contains 2d jump terms
 *
 * @tparam T value type
 * @param g grid
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat jumpX( const Grid2d<double>& g, bc bcx)
{
    SparseBlockMat jx;
    jx = jump( g.n(), g.Nx(), g.hx(), bcx);
    jx.left = g.n()*g.Ny();
    return jx;
}
SparseBlockMat jumpY( const Grid2d<double>& g, bc bcy)
{
    SparseBlockMat jy;
    jy = jump( g.n(), g.Ny(), g.hy(), bcy);
    jy.right = g.n()*g.Nx();
    return jy;
}

/**
 * @brief Matrix that contains 2d jump terms taking boundary conditions from the grid
 *
 * @tparam T value type
 * @param g grid
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat jumpX( const Grid2d<double>& g)
{
    return jumpX( g, g.bcx());
}
SparseBlockMat jumpY( const Grid2d<double>& g)
{
    return jumpY( g, g.bcy());
}

///////////////////////////////////////////3D VERSIONS//////////////////////
/**
 * @brief Matrix that contains 2d jump terms
 *
 * @tparam T value type
 * @param g grid
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat jumpX( const Grid3d<double>& g, bc bcx)
{
    SparseBlockMat jx;
    jx = jump( g.n(), g.Nx(), g.hx(), bcx);
    jx.left = g.n()*g.Ny()*g.Nz();
    return jx;
}
SparseBlockMat jumpY( const Grid3d<double>& g, bc bcy)
{
    SparseBlockMat jy;
    jy = jump( g.n(), g.Ny(), g.hy(), bcy);
    jy.right = g.n()*g.Nx();
    jy.left = g.Nz();
    return jy;
}
SparseBlockMat jumpZ( const Grid3d<double>& g, bc bcz)
{
    SparseBlockMat jz;
    jz = jump( 1, g.Nz(), g.hz(), bcz);
    jz.right = g.n()*g.Nx()*g.n()*g.Ny();
    return jz;
}

/**
 * @brief Matrix that contains 2d jump terms taking boundary conditions from the grid
 *
 * @tparam T value type
 * @param g grid
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat jumpX( const Grid3d<double>& g)
{
    return jumpX( g, g.bcx());
}
SparseBlockMat jumpY( const Grid3d<double>& g)
{
    return jumpY( g, g.bcy());
}
SparseBlockMat jumpZ( const Grid3d<double>& g)
{
    return jumpZ( g, g.bcz());
}


/**
 * @brief Create 3d derivative in x-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dx( const Grid3d<double>& g, bc bcx, direction dir = centered)
{
    SparseBlockMat dx;
    dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
    dx.left = g.n()*g.Ny()*g.Nz();
    return dx;
}

/**
 * @brief Create 3d derivative in x-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dx( const Grid3d<double>& g, direction dir = centered) { return dx( g, g.bcx(), dir);}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid3d<double>& g, bc bcy, direction dir = centered)
{
    SparseBlockMat dy;
    dy = dx_normed( g.n(), g.Ny(), g.hy(), bcy, dir);
    dy.right = g.n()*g.Nx();
    dy.left = g.Nz();
    return dy;
}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid3d<double>& g, direction dir = centered){ return dy( g, g.bcy(), dir);}

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
SparseBlockMat dz( const Grid3d<double>& g, bc bcz, direction dir = centered)
{
    SparseBlockMat dz;
    dz = dx_normed( 1, g.Nz(), g.hz(), bcz, dir);
    dz.right = g.n()*g.n()*g.Nx()*g.Ny();
    return dz;

}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the stencil
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dz( const Grid3d<double>& g, direction dir = centered){ return dz( g, g.bcz(), dir);}



///@}

} //namespace create

} //namespace dg

#endif//_DG_DERIVATIVES_CUH_
