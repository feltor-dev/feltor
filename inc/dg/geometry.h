#pragma once

#include <cassert>
#include "thrust/host_vector.h"
#include "backend/evaluation.cuh"
#include "backend/weights.cuh"
#ifdef MPI_VERSION
#include "backend/mpi_vector.h"
#include "backend/mpi_evaluation.h"
#include "backend/mpi_precon.h"
#endif//MPI_VERSION
#include "geometry/geometry_traits.h"
#include "geometry/cartesian.h"
#include "geometry/curvilinear.h"
#include "geometry/cartesianX.h"
#ifdef MPI_VERSION
#include "geometry/mpi_grids.h"
#include "geometry/mpi_curvilinear.h"
#endif//MPI_VERSION
#include "geometry/tensor.h"


/*!@file 
 *
 * geometry functions
 */

namespace dg{

/*! @brief Geometry routines 
 *
 * @ingroup geometry
 * Only those routines that are actually called need to be implemented.
 * Don't forget to specialize in the dg namespace.
 */
namespace geo{
///@addtogroup geometry
///@{

/**
 * @brief Multiply the input with the volume element without the dG weights!
 *
 * Computes \f$ f = \sqrt{g}f\f$ 
 * @tparam container container class 
 * @param inout input (contains result on output)
 * @param metric the metric object
 */
template<class container>
void multiplyVolume( container& inout, const SharedContainers<container>& metric)
{
    if( metric.isSet(0))
        dg::blas1::pointwiseDot( metric.getValue(0), inout, inout);
}

/**
 * @brief Divide the input vector with the volume element without the dG weights
 *
 * Computes \f$ v = v/ \sqrt{g}\f$ 
 * @tparam container container class 
 * @param inout input (contains result on output)
 * @param metric the metric
 */
template<class container>
void divideVolume( container& inout, const SharedContainers<container>& metric)
{
    if( metric.isSet(0))
        dg::blas1::pointwiseDivide( inout, metric.getValue(0), inout);
}

/**
 * @brief Raises the index of a covariant vector with the help of the projection tensor in 2d
 *
 * Compute \f$ v^i = g^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the first two dimensions 
 * @tparam container the container class
 * @param covX (input) covariant first component (undefined content on output)
 * @param covY (input) covariant second component (undefined content on output)
 * @param contraX (output) contravariant first component 
 * @param contraY (output) contravariant second component 
 * @param metric The metric
 * @note no alias allowed 
 */
template<class container>
void raisePerpIndex( container& covX, container& covY, container& contraX, container& contraY, const SharedContainers<container>& metric)
{
    assert( &covX != &contraX);
    assert( &covY != &contraY);
    assert( &covY != &covX);
    dg::detail::multiply( metric, covX, covY, contraX, contraY);
}

/**
 * @brief Multiplies the two-dimensional volume element
 *
 * Computes \f$ f = \sqrt{g_\perp}f\f$ where the perpendicualar volume is computed from the 2x2 submatrix of g in the first two coordinates.
 * @tparam container the container class
 * @param inout input (contains result on output)
 * @param metric The metric  
 * @note if metric is two-dimensional this function will have the same result as multiplyVolume()
 */
template<class container>
void multiplyPerpVolume( container& inout, const SharedContainers<container>& metric)
{
    if( metric.isSet(1))
        dg::blas1::pointwiseDot( metric.getValue(1), inout, inout);
}

/**
 * @brief Divides the two-dimensional volume element
 *
 * Computes \f$ f = f /\sqrt{g_\perp}\f$ where the perpendicualar volume is computed from the 2x2 submatrix of g in the first two coordinates.
 * @tparam container the container class
 * @param inout input (contains result on output)
 * @param metric The metric tensor
 * @note if metric is two-dimensional this function will have the same result as divideVolume()
 */
template<class container>
void dividePerpVolume( container& inout, const SharedContainers<container>& metric)
{
    if( metric.isSet(1))
        dg::blas1::pointwiseDivide( metric.getValue(1), inout, inout);
}


///@}
}//namespace geo

namespace create{
///@addtogroup geometry
///@{

/**
 * @brief Create the volume element on the grid (including weights!!)
 *
 * This is the same as the weights multiplied by the volume form \f$ \sqrt{g}\f$
 * @tparam Geometry any Geometry class
 * @param g Geometry object
 *
 * @return  The volume form
 */
template< class Geometry>
typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector volume( const Geometry& g)
{
    typedef typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector host_vector;
    SharedContainers<host_vector> metric = g.compute_metric();
    host_vector temp = dg::create::weights( g);
    dg::geo::multiplyVolume( temp, metric);
    return temp;
}

/**
 * @brief Create the inverse volume element on the grid (including weights!!)
 *
 * This is the same as the inv_weights divided by the volume form \f$ \sqrt{g}\f$
 * @tparam Geometry any Geometry class
 * @param g Geometry object
 *
 * @return  The inverse volume form
 */
template< class Geometry>
typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector inv_volume( const Geometry& g)
{
    typedef typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector host_vector;
    host_vector temp = volume(g);
    dg::blas1::transform(temp,temp,dg::INVERT<double>());
    return temp;
}

///@}
}//namespace create
}//namespace dg
