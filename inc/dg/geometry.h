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
#include "geometry/cartesianX.h"
#include "geometry/cylindrical.h"
#ifdef MPI_VERSION
#include "geometry/mpi_grids.h"
#endif//MPI_VERSION


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
 * @tparam Geometry Geometry class
 * @param inout input (contains result on output)
 * @param g the geometry object
 */
template<class container, class Geometry>
void multiplyVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doMultiplyVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}

/**
 * @brief Divide the input vector with the volume element without the dG weights
 *
 * Computes \f$ v = v/ \sqrt{g}\f$ 
 * @tparam container container class 
 * @tparam Geometry Geometry class
 * @param inout input (contains result on output)
 * @param g the geometry object
 */
template<class container, class Geometry>
void divideVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doDivideVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}

/**
 * @brief Raises the index of a covariant vector in 2d
 *
 * Computes \f$ v^i = g^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the two dimensions of a 2x1 product space
 * @tparam container the container class
 * @tparam Geometry the geometry class
 * @param covX (input) covariant first component (may get destroyed!!)
 * @param covY (input) covariant second component (may get destroyed!!)
 * @param contraX (output) contravariant first component
 * @param contraY (output) contravariant second component
 * @param g The geometry object
 * @note covX, covY, contraX and contraY may not be the same
 */
template<class container, class Geometry>
void raisePerpIndex( container& covX, container& covY, container& contraX, container& contraY, const Geometry& g)
{
    assert( &covX != &contraX);
    assert( &covY != &contraY);
    assert( &covY != &covX);
    dg::geo::detail::doRaisePerpIndex( covX, covY, contraX, contraY, g, typename dg::GeometryTraits<Geometry>::metric_category());

}
/**
 * @brief Raises the index of a covariant vector in 2d and multiplies the perpendicular volume
 *
 * Computes \f$ v^i = \sqrt{g/g_{zz}} g^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the two dimensions of a 2x1 product space. This special 
 * form occurs in the discretization of elliptic operators which is why it get's a special function.
 * @tparam container the container class
 * @tparam Geometry the geometry class
 * @param covX (input) covariant first component (may get destroyed!!)
 * @param covY (input) covariant second component (may get destroyed!!)
 * @param contraX (output) contravariant first component
 * @param contraY (output) contravariant second component
 * @param g The geometry object
 * @note covX, covY, contraX and contraY may not be the same
 */
template<class container, class Geometry>
void volRaisePerpIndex( container& covX, container& covY, container& contraX, container& contraY, const Geometry& g)
{
    assert( &covX != &contraX);
    assert( &covY != &contraY);
    assert( &covY != &covX);
    dg::geo::detail::doVolRaisePerpIndex( covX, covY, contraX, contraY, g, typename dg::GeometryTraits<Geometry>::metric_category());

}

/**
 * @brief Multiplies the two-dimensional volume element
 *
 * Computes \f$ f = \sqrt{g/g_{zz}}f\f$ on a 2x1 product space
 * @tparam container the container class
 * @tparam Geometry the geometry class
 * @param inout input (contains result on output)
 * @param g The geometry object
 */
template<class container, class Geometry>
void multiplyPerpVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doMultiplyPerpVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}

/**
 * @brief Divides the two-dimensional volume element
 *
 * Computes \f$ f = f /\sqrt{g/g_{zz}}\f$ on a 2x1 product space
 * @tparam container the container class
 * @tparam Geometry the geometry class
 * @param inout input (contains result on output)
 * @param g The geometry object
 */
template<class container, class Geometry>
void dividePerpVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doDividePerpVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}

/**
 * @brief Push forward a vector from cylindrical or Cartesian to a new coordinate system
 *
 * Computes \f[ v^x(x,y) = x_R (x,y) v^R(R(x,y), Z(x,y)) + x_Z v^Z(R(x,y), Z(x,y)) \\
               v^y(x,y) = y_R (x,y) v^R(R(x,y), Z(x,y)) + y_Z v^Z(R(x,y), Z(x,y)) \f]
   where \f$ x_R = \frac{\partial x}{\partial R}\f$, ... 
 * @tparam Geometry The Geometry class
 * @param vR input R-component in cylindrical coordinates
 * @param vZ input Z-component in cylindrical coordinates
 * @param vx x-component of vector (gets properly resized)
 * @param vy y-component of vector (gets properly resized)
 * @param g The geometry object
 */
template<class Functor1, class Functor2, class Geometry> 
void pushForwardPerp( Functor1 vR, Functor2 vZ, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& vx, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& vy,
        const Geometry& g)
{
    dg::geo::detail::doPushForwardPerp( vR, vZ, vx, vy, g, typename GeometryTraits<Geometry>::metric_category() ); 
}

///@cond
template<class Geometry> 
void pushForwardPerp(
        double(fR)(double,double,double), double(fZ)(double, double, double), 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g)
{
    pushForwardPerp<double(double, double, double), double(double, double, double), Geometry>( fR, fZ, out1, out2, g); 
}
///@endcond

/**
 * @brief Push forward a symmetric 2d tensor from cylindrical or Cartesian to a new coordinate system
 *
 * Computes \f[ 
 \chi^{xx}(x,y) = x_R x_R \chi^{RR} + 2x_Rx_Z \chi^{RZ} + x_Zx_Z\chi^{ZZ} \\
 \chi^{xy}(x,y) = x_R x_R \chi^{RR} + (x_Ry_Z+y_Rx_Z) \chi^{RZ} + x_Zx_Z\chi^{ZZ} \\
 \chi^{yy}(x,y) = y_R y_R \chi^{RR} + 2y_Ry_Z \chi^{RZ} + y_Zy_Z\chi^{ZZ} \\
               \f]
   where \f$ x_R = \frac{\partial x}{\partial R}\f$, ... 
 * @tparam Geometry The Geometry class
 * @param chiRR input RR-component in cylindrical coordinates
 * @param chiRZ input RZ-component in cylindrical coordinates
 * @param chiZZ input ZZ-component in cylindrical coordinates
 * @param chixx xx-component of tensor (gets properly resized)
 * @param chixy xy-component of tensor (gets properly resized)
 * @param chiyy yy-component of tensor (gets properly resized)
 * @param g The geometry object
 */
template<class FunctorRR, class FunctorRZ, class FunctorZZ, class Geometry> 
void pushForwardPerp( FunctorRR chiRR, FunctorRZ chiRZ, FunctorZZ chiZZ,
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chixx, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chixy,
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chiyy, 
        const Geometry& g)
{
    dg::geo::detail::doPushForwardPerp( chiRR, chiRZ, chiZZ, chixx, chixy, chiyy, g, typename GeometryTraits<Geometry>::metric_category() ); 
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
 * @tparam Geometry Geometry class
 * @param g Geometry object
 *
 * @return  The volume form
 */
template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector volume( const Geometry& g)
{
    return detail::doCreateVolume( g, typename GeometryTraits<Geometry>::metric_category());
}

/**
 * @brief Create the inverse volume element on the grid (including weights!!)
 *
 * This is the same as the inv_weights divided by the volume form \f$ \sqrt{g}\f$
 * @tparam Geometry Geometry class
 * @param g Geometry object
 *
 * @return  The inverse volume form
 */
template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector inv_volume( const Geometry& g)
{
    return detail::doCreateInvVolume( g, typename GeometryTraits<Geometry>::metric_category());
}

///@}
}//namespace create

/**
 * @brief This function pulls back a function defined in cartesian coordinates to the curvilinear coordinate system
 *
 * @ingroup geometry
 * i.e. F(x,y) = f(R(x,y), Z(x,y)) in 2d 
 * @tparam Functor The (binary or ternary) function object 
 * @param f The function defined in cartesian coordinates
 * @param g The grid
 * @note Template deduction will fail if you overload functions with different dimensionality (e.g. double sine( double x) and double sine(double x, double y) )
 * You will want to rename those uniquely
 *
 * @return A set of points representing F
 */
template< class Functor, class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector 
    pullback( Functor f, const Geometry& g)
{
    return dg::detail::doPullback( f, g, typename GeometryTraits<Geometry>::metric_category(), typename GeometryTraits<Geometry>::dimensionality(), typename GeometryTraits<Geometry>::memory_category() );
}


}//namespace dg
