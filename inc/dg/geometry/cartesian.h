#pragma once

#include "../backend/grid.h"
#include "geometry_traits.h"

namespace dg
{

    ///@addtogroup
/**
 * @brief one-dimensional Grid with Cartesian metric
 */
struct CartesianGrid1d: public Grid1d<double>
{
    typedef OrthonormalTag metric_category; 
    /**
     * @brief Constructor is the same as 1d Grid
     *
     @param x0 left boundary
     @param x1 right boundary
     @param n # of polynomial coefficients
     @param N # of cells
     @param bcx boundary conditions
     */
    CartesianGrid1d( double x0, double x1, unsigned n, unsigned N, bc bcx = PER): Grid1d<double>(x0,x1,n,N,bcx){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGrid1d( const Grid1d<double>& grid):Grid1d<double>(grid){}
};

/**
 * @brief two-dimensional Grid with Cartesian metric
 */
struct CartesianGrid2d: public Grid2d<double>
{
    typedef OrthonormalTag metric_category; 
    /**
     * @brief Construct a 2D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param n  # of polynomial coefficients per dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     */
    CartesianGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): Grid2d<double>(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGrid2d( const Grid2d<double>& grid):Grid2d<double>(grid){}
};

/**
 * @brief evaluates a two-dimensional function 
 *
 * same as evaluate
 * @tparam BinaryOp Binaryy function object
 * @param f functor
 * @param g geometry
 *
 * @return new instance of thrust vector
 */
template<class BinaryOp>
MPI_Vector<thrust::host_vector<double> > pullback( BinaryOp f, const CartesianGrid2d& g)
{
    return evaluate( f, g);
}
///@cond
MPI_Vector<thrust::host_vector<double> > pullback( double(f)(double,double), const CartesianGrid2d& g)
{
    return pullback<double(double,double),container>( f, g);
}
///@endcond

/**
 * @brief three-dimensional Grid with Cartesian metric
 */
struct CartesianGrid3d: public Grid3d<double>
{
    typedef OrthonormalTag metric_category; 
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
    CartesianGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): Grid3d<double>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGrid3d( const Grid3d<double>& grid):Grid3d<double>(grid){}
};

//if a pullback is ever needed write an adapter class and use evaluate functions

} //namespace dg
