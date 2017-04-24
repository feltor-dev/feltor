#pragma once

#include "../backend/gridX.h"
#include "geometry_traits.h"

namespace dg
{

///@addtogroup basicgrids
///@{
//
/**
 * @brief one-dimensional Grid with Cartesian metric
 */
struct CartesianGridX1d: public dg::GridX1d
{
    typedef OrthonormalTag metric_category; 
    /**
     * @brief 1D grid
     * 
     @param x0 left boundary
     @param x1 right boundary
     @param f factor 0<f<0.5 divides the domain
     @param n # of polynomial coefficients
     @param N # of cells
     @param bcx boundary conditions
     */
    CartesianGridX1d( double x0, double x1, double f, unsigned n, unsigned N, bc bcx = NEU):
        dg::GridX1d(x0,x1,f,n,N,bcx){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGridX1d( const dg::GridX1d& grid):dg::GridX1d(grid){}
};

/**
 * @brief two-dimensional Grid with Cartesian metric
 */
struct CartesianGridX2d: public dg::GridX2d
{
    typedef OrthonormalTag metric_category; 
    /**
     * @brief Construct a 2D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param fx factor for x-direction (fx*Nx must be a natural number)
     * @param fy factor for y-direction (fy*Ny must be a natural number)
     * @param n  # of polynomial coefficients per dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     */
    CartesianGridX2d( double x0, double x1, double y0, double y1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = NEU):dg::GridX2d(x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     *
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
    /**
     * @brief Construct a 3D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param z0 lower boundary in z
     * @param z1 upper boundary in z 
     * @param fx factor for x-direction
     * @param fy factor for y-direction
     * @param n  # of polynomial coefficients per (x-,y-) dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param Nz # of points in z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     * @attention # of polynomial coefficients in z direction is always 1
     */
    CartesianGridX3d( double x0, double x1, double y0, double y1, double z0, double z1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = NEU, bc bcz = PER): dg::GridX3d(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGridX3d( const dg::GridX3d& grid):dg::GridX3d(grid){}
};

///@}

} //namespace dg
