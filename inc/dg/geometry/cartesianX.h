#pragma once

#include "../backend/gridX.h"
#include "geometry_traits.h"

namespace dg
{

///@addtogroup basicgrids
///@{
//

/**
 * @brief two-dimensional Grid with Cartesian metric
 */
struct CartesianGridX2d: public dg::GridX2d
{
    typedef OrthonormalTag metric_category; 
    ///@copydoc GridX2d::GridX2d()
    CartesianGridX2d( double x0, double x1, double y0, double y1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = NEU):dg::GridX2d(x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     * @param grid existing grid class
     */
    CartesianGridX2d( const dg::GridX2d& grid):dg::GridX2d(grid){}
};

/**
 * @brief three-dimensional Grid with Cartesian metric
 */
struct CartesianGridX3d: public dg::GridX3d
{
    typedef OrthonormalTag metric_category; 
    ///@copydoc GridX3d::GridX3d()
    CartesianGridX3d( double x0, double x1, double y0, double y1, double z0, double z1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = NEU, bc bcz = PER): dg::GridX3d(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Construct from existing topology
     * @param grid existing grid class
     */
    CartesianGridX3d( const dg::GridX3d& grid):dg::GridX3d(grid){}
};

///@}

} //namespace dg
