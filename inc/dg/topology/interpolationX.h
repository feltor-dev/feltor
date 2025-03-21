#pragma once
//#include <iomanip>

#include "interpolation.h"
#include "gridX.h"

/*! @file

  @brief 1D, 2D and 3D interpolation matrix creation functions on X-point topology
  */

namespace dg{

namespace create{
///@addtogroup interpolation
///@{
/**
 * @brief Create interpolation matrix
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates
 * @param x X-coordinates of interpolation points
 * @param g The Grid on which to operate
 *
 * @return interpolation matrix
 */
template<class real_type>
dg::SparseMatrix<int, real_type, thrust::host_vector> interpolation( const thrust::host_vector<real_type>& x, const RealGridX1d<real_type>& g)
{
    return interpolation( x, g.grid());
}

/**
 * @brief Create interpolation matrix
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points
 * @param g The Grid on which to operate
 * @param globalbcz NEU for common interpolation. DIR for zeros at Box
 *
 * @return interpolation matrix
 */
template<class real_type>
dg::SparseMatrix<int, real_type, thrust::host_vector> interpolation( const thrust::host_vector<real_type>& x, const thrust::host_vector<real_type>& y, const aRealTopologyX2d<real_type>& g , dg::bc globalbcz = dg::NEU)
{
    return interpolation( x,y, g.grid(), globalbcz);
}



/**
 * @brief Create interpolation matrix
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates. In z-direction only a nearest neighbor interpolation is used
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points
 * @param z Z-coordinates of interpolation points
 * @param g The Grid on which to operate
 * @param globalbcz determines what to do if values lie exactly on the boundary
 *
 * @return interpolation matrix
 * @note The values of x, y and z must lie within the boundaries of g
 */
template<class real_type>
dg::SparseMatrix<int, real_type, thrust::host_vector> interpolation( const thrust::host_vector<real_type>& x, const thrust::host_vector<real_type>& y, const thrust::host_vector<real_type>& z, const aRealTopologyX3d<real_type>& g, dg::bc globalbcz= dg::NEU)
{
    return interpolation( x,y,z, g.grid(), globalbcz);
}

/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 *
 * @param g_new The new points
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
template<class real_type>
dg::SparseMatrix<int, real_type, thrust::host_vector> interpolation( const RealGridX1d<real_type>& g_new, const RealGridX1d<real_type>& g_old)
{
    return interpolation( g_new.grid(), g_old.grid());
}
/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 *
 * @param g_new The new points
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
template<class real_type>
dg::SparseMatrix<int, real_type, thrust::host_vector> interpolation( const aRealTopologyX2d<real_type>& g_new, const aRealTopologyX2d<real_type>& g_old)
{
    return interpolation( g_new.grid(), g_old.grid());
}

/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 *
 * @param g_new The new points
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
template<class real_type>
dg::SparseMatrix<int, real_type, thrust::host_vector> interpolation( const aRealTopologyX3d<real_type>& g_new, const aRealTopologyX3d<real_type>& g_old)
{
    return interpolation( g_new.grid(), g_old.grid());
}
///@}

/**
 * @brief Transform a vector from XSPACE to LSPACE
 *
 * @param in input
 * @param g grid
 *
 * @return the vector in LSPACE
 */
template<class real_type>
thrust::host_vector<real_type> forward_transform( const thrust::host_vector<real_type>& in, const aRealTopologyX2d<real_type>& g)
{
    return forward_transform( in, g.grid());
}
}//namespace create

/**
 * @brief Interpolate a vector on a single point on a 2d Grid
 *
 * @param sp Indicate whether the elements of the vector
 * v are in xspace or lspace
 *  (choose dg::xspace if you don't know what is going on here,
 *      It is faster to interpolate in dg::lspace so consider
 *      transforming v using dg::forward_transform( )
 *      if you do it very many times)
 * @param v The vector to interpolate in dg::xspace, or dg::lspace s.a. dg::forward_transform( )
 * @param x X-coordinate of interpolation point
 * @param y Y-coordinate of interpolation point
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @param bcy analogous to \c bcx, applies to y direction
 *
 * @ingroup interpolation
 * @return interpolated point
 * @note \c g.contains(x,y) must return true
 */
template<class real_type>
real_type interpolate(
    dg::space sp,
    const thrust::host_vector<real_type>& v,
    real_type x, real_type y,
    const aRealTopologyX2d<real_type>& g,
    dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU )
{
    return interpolate( sp, v, x, y, g.grid(), bcx, bcy);
}

} //namespace dg
