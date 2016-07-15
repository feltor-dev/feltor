#pragma once

#include "../backend/functions.h"
#include "../backend/grid.h"
#include "../backend/evaluation.cuh"
#include "geometry_traits.h"

namespace dg
{
///@addtogroup basicgrids
///@{

/**
 * @brief three-dimensional Grid with Cartesian metric
 * 
 * @tparam container The container class for the volume element
 */
template<class container>
struct CylindricalGrid : public Grid3d<double>
{
    typedef OrthonormalCylindricalTag metric_category; 
    typedef CartesianGrid2d perpendicular_grid;
    /**
     * @brief Construct a 3D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param z0 lower boundary in z
     * @param z1 upper boundary in z 
     * @param n  # of polynomial coefficients per (x-,y-) dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param Nz # of points in z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     * @attention # of polynomial coefficients in z direction is always 1
     */
    CylindricalGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): 
        Grid3d<double>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz),
        R_(dg::evaluate( dg::coo1, *this)){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    //is this constructor a good idea?? You could construct a Cylindrical Grid from any other Grid Type that derives from Grid3d<double>
    CylindricalGrid( const Grid3d<double>& grid):
        Grid3d<double>(grid),
        R_(dg::evaluate( dg::coo1, *this)){}
    /**
     * @brief The volume element
     *
     * @return the volume element
     */

    perpendicular_grid perp_grid() const { return CartesianGrid2d( x0(), x1(), y0(), y1(), n(), Nx(), Ny(), bcx(), bcy());}
    const container& vol()const {return R_;}
    private:
    container R_;
};

///@}
/**
 * @brief evaluates a cylindrical function 
 *
 * same as evaluate, i.e. assumes that function is given in cylindrical coordinates
 * @ingroup pullbacks
 * @tparam TernaryOp Ternary function object
 * @tparam container The container class of the Cylindrical Grid
 * @param f functor
 * @param g geometry
 *
 * @return new instance of thrust vector
 */
template<class TernaryOp, class container>
thrust::host_vector<double> pullback( TernaryOp f, const CylindricalGrid<container>& g)
{
    return evaluate( f, g);
}
///@cond
template<class container>
thrust::host_vector<double> pullback( double(f)(double,double,double), const CylindricalGrid<container>& g)
{
    return pullback<double(double,double,double),container>( f, g);
}
///@endcond

} //namespace dg
