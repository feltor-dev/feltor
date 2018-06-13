#pragma once
//#include <iomanip>

#include "interpolation.cuh"
#include "gridX.h"

/*! @file

  @brief contains 1D, 2D and 3D interpolation matrix creation functions on X-point topology
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
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const thrust::host_vector<real_type>& x, const BasicGridX1d<real_type>& g)
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
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const thrust::host_vector<real_type>& x, const thrust::host_vector<real_type>& y, const aBasicTopologyX2d<real_type>& g , dg::bc globalbcz = dg::NEU)
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
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const thrust::host_vector<real_type>& x, const thrust::host_vector<real_type>& y, const thrust::host_vector<real_type>& z, const aBasicTopologyX3d<real_type>& g, dg::bc globalbcz= dg::NEU)
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
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const BasicGridX1d<real_type>& g_new, const BasicGridX1d<real_type>& g_old)
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
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aBasicTopologyX2d<real_type>& g_new, const aBasicTopologyX2d<real_type>& g_old)
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
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aBasicTopologyX3d<real_type>& g_new, const aBasicTopologyX3d<real_type>& g_old)
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
thrust::host_vector<real_type> forward_transform( const thrust::host_vector<real_type>& in, const aBasicTopologyX2d<real_type>& g)
{
    return forward_transform( in, g.grid());
}
}//namespace create

/**
 * @brief Interpolate a single point
 *
 * @param x X-coordinate of interpolation point
 * @param y Y-coordinate of interpolation point
 * @param v The vector to interpolate in LSPACE
 * @param g The Grid on which to operate
 *
 * @return interpolated point
 */
template<class real_type>
real_type interpolate( real_type x, real_type y,  const thrust::host_vector<real_type>& v, const aBasicTopologyX2d<real_type>& g )
{
    return interpolate( x,y,v,g.grid());
}

} //namespace dg
