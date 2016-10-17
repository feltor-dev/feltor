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
struct CylindricalGrid : public dg::Grid3d
{
    typedef OrthonormalCylindricalTag metric_category; 
    typedef dg::CartesianGrid2d perpendicular_grid;
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
        dg::Grid3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz),
        R_(dg::evaluate( dg::cooX3d, *this)){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    //is this constructor a good idea?? You could construct a Cylindrical Grid from any other Grid Type that derives from Grid3d
    CylindricalGrid( const dg::Grid3d& grid):
        dg::Grid3d(grid),
        R_(dg::evaluate( dg::cooX3d, *this)){}
    /**
     * @brief The volume element
     *
     * @return the volume element
     */

    perpendicular_grid perp_grid() const { return dg::CartesianGrid2d( x0(), x1(), y0(), y1(), n(), Nx(), Ny(), bcx(), bcy());}
    const container& vol()const {return R_;}
    private:
    container R_;
};

///@}

} //namespace dg

