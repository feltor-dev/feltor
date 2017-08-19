#pragma once

#include "dg/backend/mpi_grid.h"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

/**
 * @brief This is the abstract interface class for a two-dimensional Geometry
 */
struct aMPIGeometry2d : public aMPITopology2d
{
    typedef MPI_Vector<thrust::host_vector<double> > host_vector;
    ///@copydoc aGeometry2d::jacobian()
    SparseTensor<host_vector > jacobian()const {
        return do_compute_jacobian();
    }
    ///@copydoc aGeometry2d::metric()
    SparseTensor<host_vector > metric()const {
        return do_compute_metric();
    }
    ///@copydoc aGeometry2d::map()
    std::vector<host_vector > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aMPIGeometry2d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aMPIGeometry2d(){}
    protected:
    aMPIGeometry2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):
        aMPITopology2d( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy, comm)
    { }
    aMPIGeometry2d( const aMPIGeometry2d& src):aMPITopology2d(src){}
    aMPIGeometry2d& operator=( const aMPIGeometry2d& src){
        aMPITopology2d::operator=(src);
        return *this;
    }
    private:
    virtual SparseTensor<host_vector > do_compute_metric()const {
        return SparseTensor<host_vector >();
    }
    virtual SparseTensor<host_vector > do_compute_jacobian()const {
        return SparseTensor<host_vector >();
    }
    virtual std::vector<host_vector > do_compute_map()const{
        std::vector<host_vector> map(2);
        map[0] = dg::evaluate(dg::cooX2d, *this);
        map[1] = dg::evaluate(dg::cooY2d, *this);
        return map;
    }
};

/**
 * @brief This is the abstract interface class for a three-dimensional MPIGeometry
 */
struct aMPIGeometry3d : public aMPITopology3d
{
    typedef MPI_Vector<thrust::host_vector<double> > host_vector;
    ///@copydoc aGeometry3d::jacobian()
    SparseTensor<host_vector > jacobian()const{
        return do_compute_jacobian();
    }
    ///@copydoc aGeometry3d::metric()
    SparseTensor<host_vector > metric()const { 
        return do_compute_metric(); 
    }
    ///@copydoc aGeometry3d::map()
    std::vector<host_vector > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aMPIGeometry3d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aMPIGeometry3d(){}
    protected:
    aMPIGeometry3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        aMPITopology3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    aMPIGeometry3d( const aMPIGeometry3d& src):aMPITopology3d(src){}
    aMPIGeometry3d& operator=( const aMPIGeometry3d& src){
        aMPITopology3d::operator=(src);
        return *this;
    }
    private:
    virtual SparseTensor<host_vector > do_compute_metric()const {
        return SparseTensor<host_vector >();
    }
    virtual SparseTensor<host_vector > do_compute_jacobian()const {
        return SparseTensor<host_vector >();
    }
    virtual std::vector<host_vector > do_compute_map()const{
        std::vector<host_vector> map(3);
        map[0] = dg::evaluate(dg::cooX3d, *this);
        map[1] = dg::evaluate(dg::cooY3d, *this);
        map[2] = dg::evaluate(dg::cooZ3d, *this);
        return map;
    }
};

///@}

///@addtogroup geometry
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
    CartesianMPIGrid2d( const dg::MPIGrid2d& g): aMPIGeometry2d( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().bcx(),g.global().bcy(),g.communicator()){}
    virtual CartesianMPIGrid2d* clone()const{return new CartesianMPIGrid2d(*this);}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny){
        aMPITopology2d::do_set(new_n,new_Nx,new_Ny);
    }

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
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, dg::PER,dg::PER,dg::PER, comm){}

    /**
     * @copydoc MPIGrid3d::MPIGrid3d()
     * @param comm a three-dimensional Cartesian communicator
     * @note the paramaters given in the constructor are global parameters 
     */
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}

    CartesianMPIGrid3d( const dg::MPIGrid3d& g): aMPIGeometry3d( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().z0(),g.global().z1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().Nz(),g.global().bcx(),g.global().bcy(),g.global().bcz(),g.communicator()){}
    virtual CartesianMPIGrid3d* clone()const{return new CartesianMPIGrid3d(*this);}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aMPITopology3d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

/**
 * @brief three-dimensional Grid with Cylindrical metric
 */
struct CylindricalMPIGrid3d: public aMPIGeometry3d
{
    CylindricalMPIGrid3d( double R0, double R1, double Z0, double Z1, double phi0, double phi1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcR, bc bcZ, bc bcphi, MPI_Comm comm): dg::aMPIGeometry3d(R0,R1,Z0,Z1,phi0,phi1,n,NR,NZ,Nphi,bcR,bcZ,bcphi,comm){}
    ///take PER for bcphi
    CylindricalMPIGrid3d( double R0, double R1, double Z0, double Z1, double phi0, double phi1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcR, bc bcZ, MPI_Comm comm): dg::aMPIGeometry3d(R0,R1,Z0,Z1,phi0,phi1,n,NR,NZ,Nphi,bcR,bcZ,dg::PER,comm){}

    virtual CylindricalMPIGrid3d* clone()const{return new CylindricalMPIGrid3d(*this);}
    private:
    virtual SparseTensor<host_vector > do_compute_metric()const{
        SparseTensor<host_vector> metric(1);
        host_vector R = dg::evaluate(dg::cooX3d, *this);
        for( unsigned i = 0; i<size(); i++)
            R.data()[i] = 1./R.data()[i]/R.data()[i];
        metric.idx(2,2)=0;
        metric.value(0) = R;
        return metric;
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aMPITopology3d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

///@}

}//namespace dg
