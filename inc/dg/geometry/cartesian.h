#pragma once

#include "../backend/grid.h"

namespace dg
{

struct CartesianGrid1d: public Grid1d<double>
{
    CartesianGrid2d( double x0, double x1, unsigned n, unsigned N, bc bcx = PER): Grid1d<double>(x0,x1,n,N,bcx){}
    CartesianGrid2d( const Grid1d<double>& grid):Grid1d<double>(grid){}
};

struct CartesianGrid2d: public Grid2d<double>
{
    CartesianGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): Grid2d<double>(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    CartesianGrid2d( const Grid2d<double>& grid):Grid2d<double>(grid){}
};

struct CartesianGrid3d: public Grid3d<double>
{
    CartesianGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): Grid3d<double>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    CartesianGrid3d( const Grid3d<double>& grid):Grid3d<double>(grid){}
};

///@cond
template<>
class GeometryTraits<CartesianGrid1d>
{
    typedef CartesianTag metric_category;
};
template<>
class GeometryTraits<CartesianGrid2d>
{
    typedef CartesianTag metric_category;
};
template<>
class GeometryTraits<CartesianGrid3d>
{
    typedef CartesianTag metric_category;
};

namespace geo{
namespace detail{

template <class container, class Geometry>
void doAttachVolume( container& inout, const Geometry& g, CartesianTag)
{
};

template <class container, class Geometry>
void doRaiseIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, CartesianTag)
{
};


}//namespace detail 
}//namespace geo

///@endcond

} //namespace dg
