#ifndef _DG_DERIVATIVES_CUH_
#define _DG_DERIVATIVES_CUH_

#include <cusp/elementwise.h>

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
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */

SparseBlockMat dx( const Grid2d<double>& g, bc bcx, norm no = normed, direction dir = centered)
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
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dx( const Grid2d<double>& g, norm no = normed, direction dir = centered) { return dx( g, g.bcx(), no, dir);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid2d<double>& g, bc bcy, norm no = normed, direction dir = centered)
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
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid2d<double>& g, norm no = normed, direction dir = centered){ return dy( g, g.bcy(), no, dir);}

/**
 * @brief Matrix that contains 2d jump terms
 *
 * @tparam T value type
 * @param g grid
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 * @param no the norm
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat jump2d( const Grid2d<double>& g, bc bcx, bc bcy, norm no);
//separate into jumpX and jumpY functions

/**
 * @brief Matrix that contains 2d jump terms taking boundary conditions from the grid
 *
 * @tparam T value type
 * @param g grid
 * @param no the norm
 *
 * @return A host matrix in coordinate format
 */
//SparseBlockMat jump2d( const Grid2d<double>& g, norm no)
//{
//    return jump2d( g, g.bcx(), g.bcy(), no);
//}

///////////////////////////////////////////3D VERSIONS//////////////////////
/**
 * @brief Matrix that contains 2d jump terms
 *
 * @tparam T value type
 * @param g grid
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 * @param no the norm
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat jump2d( const Grid3d<double>& g, bc bcx, bc bcy, norm no);

/**
 * @brief Matrix that contains 2d jump terms taking boundary conditions from the grid
 *
 * @tparam T value type
 * @param g grid
 * @param no the norm
 *
 * @return A host matrix in coordinate format
 */
//SparseBlockMat jump2d( const Grid3d<double>& g, norm no)
//{
//    return jump2d( g, g.bcx(), g.bcy(), no);
//}

/**
 * @brief Create 3d derivative in x-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dx( const Grid3d<double>& g, bc bcx, norm no = normed, direction dir = centered)
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
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dx( const Grid3d<double>& g, norm no = normed, direction dir = centered) { return dx( g, g.bcx(), no, dir);}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid3d<double>& g, bc bcy, norm no = normed, direction dir = centered)
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
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dy( const Grid3d<double>& g, norm no = normed, direction dir = centered){ return dy( g, g.bcy(),no, dir);}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @tparam T value-type
 * @param g The grid on which to create dz
 * @param bcz The boundary condition
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the stencil
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dz( const Grid3d<double>& g, bc bcz, norm no = normed, direction dir = centered)
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
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the stencil
 *
 * @return A host matrix in coordinate format
 */
SparseBlockMat dz( const Grid3d<double>& g, norm no = normed, direction dir = centered){ return dz( g, g.bcz(), no, dir);}



///@}

} //namespace create

} //namespace dg

#endif//_DG_DERIVATIVES_CUH_
