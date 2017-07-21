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
struct CylindricalGrid3d : public dg::Grid3d
{
    typedef OrthonormalCylindricalTag metric_category; 
    typedef dg::CartesianGrid2d perpendicular_grid;
    ///@copydoc Grid3d()
    CylindricalGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): 
        dg::Grid3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz),
        R_(dg::evaluate( dg::cooX3d, *this)){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    //is this constructor a good idea?? You could construct a Cylindrical Grid from any other Grid Type that derives from Grid3d
    CylindricalGrid3d( const dg::Grid3d& grid):
        dg::Grid3d(grid),
        R_(dg::evaluate( dg::cooX3d, *this)){}

    /**
    * @brief Return the grid of the R-Z planes
    * @return a Cartesian 2d grid of the R-Z plane
    */
    perpendicular_grid perp_grid() const { return dg::CartesianGrid2d( x0(), x1(), y0(), y1(), n(), Nx(), Ny(), bcx(), bcy());}
    /**
     * @brief The volume element R
     * @return the volume element R
     */
    const container& vol()const {return R_;}
    void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        dg::Grid3d::set(new_n,new_Nx,new_Ny,new_Nz);
        R_=dg::evaluate(dg::cooX3d, *this);
    }
    private:
    container R_;
};

///@}

} //namespace dg

