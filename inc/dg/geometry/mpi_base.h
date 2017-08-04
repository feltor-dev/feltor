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
    ///Geometries are cloneable
    virtual aMPIGeometry2d* clone()const=0;
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
    ///Geometries are cloneable
    virtual aMPIGeometry3d* clone()const=0;
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

///@}
namespace create
{
///@addtogroup metric
///@{

SharedContainers<host_vec > metric( const aMPIGeometry2d& g)
{
    return g.compute_metric();
}
SharedContainers<host_vec > metric( const aMPIGeometry3d& g)
{
    return g.compute_metric();
}

///@}
}//namespace create



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
    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): aMPIGeometry2d( x0, x1, y0, y1, n, Nx, Ny, dg::PER,dg::PER,comm){}

    /**
     * @copydoc Grid2d::Grid2d()
     * @param comm a two-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):dg::aMPIGeometry2d( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    CartesianMPIGrid2d( const dg::MPIGrid2d& g): aMPIGeometry2d( g.x0(),g.x1(),g.y0(),g.y1(),g.n(),g.Nx(),g.Ny(),g.bcx(),g.bcy(),g.comm()){}
};

/**
 * @brief The mpi version of a cartesian grid
 */
struct CartesianMPIGrid3d : public aMPIGeometry3d
{
    /**
     * @copydoc MPIGrid3d::MPIGrid3d()
     * @param comm a three-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, dg::PER,dg::PER,dg::PER comm){}

    /**
     * @copydoc MPIGrid3d::MPIGrid3d()
     * @param comm a three-dimensional Cartesian communicator
     * @note the paramaters given in the constructor are global parameters 
     */
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}

    CartesianMPIGrid3d( const dg::MPIGrid3d& g): aMPIGeometry3d( g.x0(),g.x1(),g.y0(),g.y1(),g.z0(),g.z1(),g.n(),g.Nx(),g.Ny(),g.Nz(),g.bcx(),g.bcy(),g.bcz(),g.comm()){}
};

/**
 * @brief three-dimensional Grid with Cylindrical metric
 */
struct CylindricalMPIGrid3d: public aMPIGeometry3d
{
    CylindricalMPIGrid3d( double R0, double R1, double Z0, double Z1, double phi0, double phi1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcR, bc bcZ, bc bcphi, MPI_Comm comm): dg::aGeometry3d(R0,R1,Z0,Z1,phi0,phi1,n,NR,NZ,Nphi,bcR,bcZ,bcphi,comm){}
    ///take PER for bcphi
    CylindricalMPIGrid3d( double R0, double R1, double Z0, double Z1, double phi0, double phi1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcR, bc bcZ, MPI_Comm comm): dg::aGeometry3d(R0,R1,Z0,Z1,phi0,phi1,n,NR,NZ,Nphi,bcR,bcZ,dg::PER,comm){}

    virtual CylindricalMPIGrid3d* clone()const{return new CylindricalMPIGrid3d(*this);}
    private:
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const{

        std::vector<MPI_Vector<thrust::host_vector<double> > > values(2, size());
        MPI_Vector<thrust::host_vector<double> > R = dg::evaluate(dg::coo1, *this);
        unsigned size2d = n()*n()*Nx()*Ny();
        for( unsigned i = 0; i<Nz(); i++)
        for( unsigned j = 0; j<size2d; j++)
        {
            unsigned idx = i*size2d+j;
            values[1].data()[idx] = R.data()[j];
            values[0].data()[idx] = 1./R.data()[j]/R.data()[j];
        }
        Operator<int> mat_idx(3,-1); mat_idx(2,2) = 0;
        std::vector<int> vec_idx(3,-1); vec_idx[0] = 1;
        return SharedContainers<thrust::host_vector<double> >( mat_idx, vec_idx, values);
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aMPITopology2d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

///@}

}//namespace dg
