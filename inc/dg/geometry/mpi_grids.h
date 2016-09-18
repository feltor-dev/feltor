#pragma once

#include "geometry_traits.h"
#include "../backend/mpi_grid.h"
#include "cylindrical.h"

namespace dg
{
///@addtogroup basicgrids
///@{

namespace cartesian
{
struct MPIGrid2d : public dg::MPI_Grid2d
{
    typedef OrthonormalTag metric_category; 

    MPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): dg::MPI_Grid2d( x0, x1, y0, y1, n, Nx, Ny, comm){}

    MPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):dg::MPI_Grid2d( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    MPIGrid2d( const dg::MPI_Grid2d& grid ):MPI_Grid2d( grid){}
};

struct MPIGrid3d : public dg::MPI_Grid3d
{
    typedef OrthonormalTag metric_category; 

    MPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): dg::MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, comm){}

    MPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):dg::MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    MPIGrid3d( const dg::MPI_Grid3d& grid ): dg::MPI_Grid3d( grid){}
};

} //namespace cartesian

namespace cylindrical
{
/**
 * @brief MPI version of Cylindrical grid
 *
 * @tparam container The MPI Vector container
 */
template<class container>
struct MPIGrid3d : public MPI_Grid3d
{
    typedef OrthonormalCylindricalTag metric_category; 
    typedef dg::cartesian::MPIGrid2d perpendicular_grid;

    MPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): 
        dg::MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, comm), 
        R_( dg::evaluate( dg::coo1, *this)) { }

    MPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        dg::MPI_Grid3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm),
        R_( dg::evaluate( dg::coo1, *this))
        {}

    MPIGrid3d( const MPI_Grid3d& grid ):
        MPI_Grid3d( grid),
        R_( dg::evaluate( dg::coo1, *this))
    {}

    const container& vol() const { return R_;}
    perpendicular_grid perp_grid() const { 
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( communicator(), remain_dims, &planeComm);
        return dg::cartesian::MPIGrid2d( global().x0(), global().x1(), global().y0(), global().y1(), global().n(), global().Nx(), global().Ny(), global().bcx(), global().bcy(), planeComm);
    }
    private:
    container R_;
};
}//namespace cylindrical
///@}

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
MPI_Vector< thrust::host_vector<double> > doPullback( TernaryOp f, const Geometry& g, CurvilinearTag, ThreeDimensionalTag, MPITag)
{
    thrust::host_vector<double> vec( g.size());
    unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
    Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> absz = create::abscissas( gz);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<size2d; i++)
            vec[k*size2d+i] = f( g.r().data()[k*size2d+i], g.z().data()[k*size2d+i], absz[k]);
    MPI_Vector<thrust::host_vector<double> > v( vec, g.communicator());
    return v;
}
template< class BinaryOp, class Geometry>
MPI_Vector< thrust::host_vector<double> > doPullback( BinaryOp f, const Geometry& g, OrthonormalCylindricalTag, TwoDimensionalTag, MPITag)
{
    return evaluate( f, g);
}
template< class TernaryOp, class Geometry>
MPI_Vector< thrust::host_vector<double> > doPullback( TernaryOp f, const Geometry& g, OrthonormalCylindricalTag, ThreeDimensionalTag, MPITag)
{
    return evaluate( f,g);
}

} //namespace detail

}//namespace dg
