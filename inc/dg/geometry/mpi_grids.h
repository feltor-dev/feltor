#pragma once

#include "geometry_traits.h"
#include "../backend/mpi_grid.h"
#include "cylindrical.h"

namespace dg
{
///@addtogroup basicgrids
///@{

struct CartesianMPIGrid2d : public MPI_Grid2d
{
    typedef OrthonormalTag metric_category; 

    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): MPI_Grid2d( x0, x1, y0, y1, n, Nx, Ny, comm){}

    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):MPI_Grid2d( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    CartesianMPIGrid2d( const MPI_Grid2d& grid ):MPI_Grid2d( grid){}
};

///@}
/**
 * @brief evaluates a two-dimensional function 
 *
 * same as evaluate
 * @ingroup pullbacks
 * @tparam BinaryOp Binary function object
 * @param f functor
 * @param g geometry
 *
 * @return new instance of thrust vector
 */
template<class BinaryOp>
MPI_Vector<thrust::host_vector<double> > pullback( BinaryOp f, const CartesianMPIGrid2d& g)
{
    return evaluate( f, g);
}
///@cond
MPI_Vector<thrust::host_vector<double> > pullback( double(f)(double,double), const CartesianMPIGrid2d& g)
{
    return pullback<double(double,double)>( f, g);
}
///@endcond

///@addtogroup basicgrids
///@{
struct CartesianMPIGrid3d : public MPI_Grid3d
{
    typedef OrthonormalTag metric_category; 

    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, comm){}

    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    CartesianMPIGrid3d( const MPI_Grid3d& grid ):MPI_Grid3d( grid){}
};


/**
 * @brief MPI version of Cylindrical grid
 *
 * @tparam container The MPI Vector container
 */
template<class container>
struct CylindricalMPIGrid : public MPI_Grid3d
{
    typedef OrthonormalCylindricalTag metric_category; 
    typedef CartesianMPIGrid2d perpendicular_grid;

    CylindricalMPIGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): 
        MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, comm), 
        R_( dg::evaluate( dg::coo1, *this)) { }

    CylindricalMPIGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm),
        R_( dg::evaluate( dg::coo1, *this))
        {}

    CylindricalMPIGrid( const MPI_Grid3d& grid ):
        MPI_Grid3d( grid),
        R_( dg::evaluate( dg::coo1, *this))
    {}

    const container& vol() const { return R_;}
    perpendicular_grid perp_grid() const { 
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( communicator(), remain_dims, &planeComm);
        return CartesianMPIGrid2d( global().x0(), global().x1(), global().y0(), global().y1(), global().n(), global().Nx(), global().Ny(), global().bcx(), global().bcy(), planeComm);
    }
    private:
    container R_;
};
///@}

/**
 * @brief evaluates a cylindrical function 
 *
 * same as evaluate, i.e. assumes that the function is given in cylindrical coordinates
 * @ingroup pullbacks
 * @tparam TernaryOp Ternary function object
 * @tparam container The container class of the Cylindrical Grid
 * @param f functor
 * @param g geometry
 *
 * @return new instance of thrust vector
 */
template<class TernaryOp, class container>
MPI_Vector<thrust::host_vector<double> > pullback( TernaryOp f, const CylindricalMPIGrid<container>& g)
{
    return evaluate( f, g);
}
///@cond
template<class container>
MPI_Vector<thrust::host_vector<double> > pullback( double(f)(double,double,double), const CylindricalMPIGrid<container>& g)
{
    return pullback<double(double,double,double),container>( f, g);
}
///@endcond


}//namespace dg
