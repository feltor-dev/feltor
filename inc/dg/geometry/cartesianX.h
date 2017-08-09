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

///@}

} //namespace dg
