#pragma once

#include "../backend/grid.h"
#include "geometry_traits.h"

namespace dg
{

struct CartesianGrid1d: public Grid1d<double>
{
    typedef CartesianTag metric_category; 
    CartesianGrid2d( double x0, double x1, unsigned n, unsigned N, bc bcx = PER): Grid1d<double>(x0,x1,n,N,bcx){}
    CartesianGrid2d( const Grid1d<double>& grid):Grid1d<double>(grid){}
};

struct CartesianGrid2d: public Grid2d<double>
{
    typedef CartesianTag metric_category; 
    CartesianGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): Grid2d<double>(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    CartesianGrid2d( const Grid2d<double>& grid):Grid2d<double>(grid){}
};

struct CartesianGrid3d: public Grid3d<double>
{
    typedef CartesianTag metric_category; 
    CartesianGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): Grid3d<double>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    CartesianGrid3d( const Grid3d<double>& grid):Grid3d<double>(grid){}
};


template<class TernaryOp>
thrust::host_vector<double> pullback( dg::system sys, TernaryOp f, const CartesianGrid3d& g)
{
    if(sys == dg::cartesian) return evaluate( f, g);
    thrust::host_vector<double> v( g.size());
    if(sys == dg::cylindrical) 
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
                            double X = absx[j*n+l], Y = absy[i*n+k], Z = absz[s];
                            double R = sqrt(X*X+Y*Y);
                            double phi = 0;
                            if( X==0 && Y==0)phi = 0;
                            else if( X >= 0) phi = arcsin(Y/R);
                            else if( X > 0) phi = arctan(Y/X);
                            else phi = -arcsin(Y/R) + M_PI;
                            v[ (((s*gy.N()+i)*n+k)*g.Nx() + j)*n + l] = f( R,Z,phi);
                        }
    }
    return v;

}


} //namespace dg
