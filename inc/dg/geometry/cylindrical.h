#pragma once

#include "../backend/functions.h"
#include "../backend/grid.h"
#include "../backend/evaluation.cuh"
#include "geometry_traits.h"

namespace dg
{

template<class container>
struct CylindricalGrid : public Grid3d<double>
{
    typedef OrthonormalCylindricalTag metric_category; 
    CylindricalGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): g3d_(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){
        R_ = dg::evaluate( dg::coo1, *this);
    
    }
    CylindricalGrid( const Grid3d<double>& grid):Grid3d<double>(grid){
        R_ = dg::evaluate( dg::coo1, *this);
    }
    const container& vol(){return R_;}
    private:
    container R_;
};

template<class TernaryOp, class container>
thrust::host_vector<double> pullback( TernaryOp f, const CylindricalGrid<container>& g)
{
    return evaluate( f, g);
}

} //namespace dg
