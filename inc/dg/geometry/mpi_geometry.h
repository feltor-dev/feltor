#pragma once

#include "dg/backend/mpi_grid.h"
#include "tensor.h"

namespace dg
{

///@addtogroup geometry
///@{

/**
 * @brief This is the abstract interface class for a two-dimensional Geometry
 */
struct aMPIGeometry2d : public aMPITopology2d
{
    typedef MPI_Vector<thrust::host_vector<double> > host_vec;
    const SharedContainers<host_vec >& map()const{return map_;}
    SharedContainers<host_vec > compute_metric()const {
        return do_compute_metric();
    }
    ///allow deletion through base class pointer
    virtual ~aMPIGeometry2d(){}
    protected:
    aMPIGeometry2d(const SharedContainers<host_vec >& map, const SharedContainers<container >& metric): map_(map), metric_(metric){}
    aMPIGeometry2d( const aMPIGeometry2d& src):map_(src.map_), metric_(src.metric_){}
    aMPIGeometry2d& operator=( const aMPIGeometry2d& src){
        map_=src.map_;
        metric_=src.metric_;
    }
    SharedContainers<host_vec >& map(){return map_;}
    private:
    SharedContainers<host_vec > map_;
    virtual SharedContainers<host_vec > do_compute_metric()const=0;
};

/**
 * @brief This is the abstract interface class for a three-dimensional MPIGeometry
 */
struct aMPIGeometry3d : public aMPITopology3d
{
    typedef MPI_Vector<thrust::host_vector<double> > host_vec;
    const SharedContainers<host_vec >& map()const{return map_;}
    SharedContainers<host_vec > compute_metric()const {
        return do_compute_metric();
    }
    ///allow deletion through base class pointer
    virtual ~aMPIGeometry3d(){}
    protected:
    aMPIGeometry3d(const SharedContainers<host_vec >& map): map_(map){}
    aMPIGeometry3d( const aMPIGeometry2d& src):map_(src.map_){}
    aMPIGeometry3d& operator=( const aMPIGeometry3d& src){
        map_=src.map_;
    }
    SharedContainers<host_vec >& map(){return map_;}
    private:
    SharedContainers<host_vec > map_;
    virtual SharedContainers<host_vec > do_compute_metric()const=0;
};

namespace create
{

SharedContainers<host_vec > metric( const aMPIGeometry2d& g)
{
    return g.compute_metric();
}
SharedContainers<host_vec > metric( const aMPIGeometry3d& g)
{
    return g.compute_metric();
}

}//namespace create


///@}

///@addtogroup basicgrids
///@{

/**
 * @brief The mpi version of a cartesian grid
 */
struct CartesianMPIGrid2d : public aMPIGeometry2d
{
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
struct CartesianMPIGrid3d : public aMPIGeometry3d
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

}//namespace dg
