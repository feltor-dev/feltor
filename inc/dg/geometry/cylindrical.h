#pragma once

#include "../backend/functions.h"
#include "../backend/grid.h"
#include "geometry_traits.h"

namespace dg
{

template<class container>
struct CylindricalGrid : public Grid3d<double>
{
    typedef CylindricalTag metric_category; 
    CylindricalGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): g3d_(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){
        R_ = dg::evaluate( dg::coo1, *this);
    
    }
    CylindricalGrid( const Grid3d<double>& grid):Grid3d<double>(grid){}
    const container& vol(){return R_;}
    private:
    container R_;

};

template<class TernaryOp>
thrust::host_vector<double> pullback( dg::system sys, TernaryOp f, const CartesianGrid3d& g)
{
    if(sys == dg::cylindrical) return evaluate( f, g);
    thrust::host_vector<double> v( g.size());
    if(sys == dg::cartesian) 
    {
        unsigned n= g.n();
        //TODO: opens dlt.dat three times...!!
        Grid1d<double> gx( g.x0(), g.x1(), n, g.Nx()); 
        Grid1d<double> gy( g.y0(), g.y1(), n, g.Ny());
        Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
        thrust::host_vector<double> absx = create::abscissas( gx);
        thrust::host_vector<double> absy = create::abscissas( gy);
        thrust::host_vector<double> absz = create::abscissas( gz);

        for( unsigned s=0; s<gz.N(); s++)
            for( unsigned i=0; i<gy.N(); i++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned j=0; j<gx.N(); j++)
                        for( unsigned l=0; l<n; l++)
                        {
                            double R = absx[j*n+l], Z = absy[i*n+k], P = absz[s];
                            double X = R*cos(P);
                            double Y = R*sin(P);
                            v[ (((s*gy.N()+i)*n+k)*g.Nx() + j)*n + l] = f(X,Y,Z);
                        }
    }
    return v;

}



} //namespace dg
