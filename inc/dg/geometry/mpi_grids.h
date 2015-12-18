#pragma once

#include "geometry_traits.h"
#include "../backend/mpi_grid.h"

namespace dg
{

struct CartesianMPIGrid2d : public MPI_Grid2d
{
    typedef OrthonormalTag metric_category; 

    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): MPI_Grid2d( x0, x1, y0, y1, n, Nx, Ny, comm){}

    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):MPI_Grid2d( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    CartesianMPIGrid2d( const MPI_Grid2d& grid ):MPI_Grid2d( grid){}
};

struct CartesianMPIGrid3d : public MPI_Grid3d
{
    typedef OrthonormalTag metric_category; 

    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, comm){}

    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    CartesianMPIGrid3d( const MPI_Grid3d& grid ):MPI_Grid3d( grid){}
};

template<class container>
struct CylindricalMPIGrid : public MPI_Grid3d
{
    typedef OrthonormalCylindricalTag metric_category; 

    CylindricalMPIGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): 
        MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, comm), 
        R_( dg::evaluate( dg::coo1, *this) { }

    CylindricalMPIGrid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm),
        R_( dg::evaluate( dg::coo1, *this)
        {}

    CylindricalMPIGrid( const MPI_Grid3d& grid ):
        MPI_Grid3d( grid),
        R_( dg::evaluate( dg::coo1, *this)
    {}

    const container& vol() const { return R_;}
    private:
    container R_;
};

}//namespace dg
