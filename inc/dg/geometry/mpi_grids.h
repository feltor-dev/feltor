#pragma once

#include "geometry_traits.h"
#include "../backend/mpi_grid.h"
#include "curvilinear_cylindrical.h"

namespace dg
{


///@addtogroup basicgrids
///@{

/**
 * @brief The mpi version of a cartesian grid
 */
struct CartesianMPIGrid2d : public dg::MPIGrid2d
{
    typedef OrthonormalTag metric_category; 

    /**
     * @copydoc Grid2d::Grid2d()
     * @param comm a two-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): dg::MPIGrid2d( x0, x1, y0, y1, n, Nx, Ny, comm){}

    /**
     * @copydoc Grid2d::Grid2d()
     * @param comm a two-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):dg::MPIGrid2d( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    CartesianMPIGrid2d( const dg::MPIGrid2d& grid ):MPIGrid2d( grid){}
};

/**
 * @brief The mpi version of a cartesian grid
 */
struct CartesianMPIGrid3d : public dg::MPIGrid3d
{
    typedef OrthonormalTag metric_category; 

    /**
     * @copydoc Grid3d::Grid3d()
     * @param comm a three-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): dg::MPIGrid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, comm){}

    /**
     * @copydoc Grid3d::Grid3d()
     * @param comm a three-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):dg::MPIGrid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    CartesianMPIGrid3d( const dg::MPIGrid3d& grid ): dg::MPIGrid3d( grid){}
};

///@}

///@cond
/////////////////////////////////////////////////////MPI pullbacks/////////////////////////////////////////////////
namespace detail{
template< class Geometry>
MPI_Vector< thrust::host_vector<double> > doPullback( double(f)(double,double), const Geometry& g, CurvilinearTag, TwoDimensionalTag, MPITag)
{
    return doPullback<double(double,double), Geometry>( f, g);
}
template< class Geometry>
MPI_Vector< thrust::host_vector<double> > pullback( double(f)(double,double,double), const Geometry& g, CurvilinearTag, ThreeDimensionalTag, MPITag)
{
    return doPullback<double(double,double,double), Geometry>( f, g);
}

template< class BinaryOp, class Geometry>
MPI_Vector< thrust::host_vector<double> > doPullback( BinaryOp f, const Geometry& g, CurvilinearTag, TwoDimensionalTag, MPITag)
{
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( g.r().data()[i], g.z().data()[i]);
    MPI_Vector<thrust::host_vector<double> > v( vec, g.communicator());
    return v;
}

template< class TernaryOp, class Geometry>
MPI_Vector< thrust::host_vector<double> > doPullback( TernaryOp f, const Geometry& g, CurvilinearPerpTag, ThreeDimensionalTag, MPITag)
{
    thrust::host_vector<double> vec( g.size());
    unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
    Grid1d gz( g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> absz = create::abscissas( gz);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<size2d; i++)
            vec[k*size2d+i] = f( g.r().data()[k*size2d+i], g.z().data()[k*size2d+i], absz[k]);
    MPI_Vector<thrust::host_vector<double> > v( vec, g.communicator());
    return v;
}
template< class BinaryOp, class Geometry>
MPI_Vector< thrust::host_vector<double> > doPullback( BinaryOp f, const Geometry& g, OrthonormalTag, TwoDimensionalTag, MPITag)
{
    return evaluate( f, g);
}
template< class TernaryOp, class Geometry>
MPI_Vector< thrust::host_vector<double> > doPullback( TernaryOp f, const Geometry& g, OrthonormalTag, ThreeDimensionalTag, MPITag)
{
    return evaluate( f,g);
}

} //namespace detail
///@endcond

}//namespace dg
