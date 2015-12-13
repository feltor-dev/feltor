#pragma once

#include "../backend/functions.h"
#include "../backend/grid.h"

namespace dg
{

template<class container>
struct CylindricalGrid : public Grid3d<double>
{
    CylindricalGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): g3d_(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){
        R_ = dg::evaluate( dg::coo1, *this);
    
    }
    CylindricalGrid( const Grid3d<double>& grid):Grid3d<double>(grid){}
    private:
    container R_;

};

///@cond
template<>
class GeometryTraits<CylindricalGrid>
{
    typedef CylindricalTag metric_category;
};
namespace geo{
namespace detail{

template <class container, class Geometry>
void doAttachVolume( container& inout, const Geometry& g, CylindricalTag)
{
    dg::blas1::pointwiseDot( inout, g.vol(), inout);
};
template <class container, class Geometry>
void doRaiseIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, CylindricalTag)
{
};


}//namespace detail 
}//namespace geo

} //namespace dg
