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
    ///@copydoc aMPITopology2d::aMPITopology2d()
    aMPIGeometry2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):
        aMPITopology2d( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy, comm)
    { }
    ///@copydoc aMPITopology2d::aMPITopology2d(const aMPITopology2d&)
    aMPIGeometry2d( const aMPIGeometry2d& src):aMPITopology2d(src){}
    ///@copydoc aMPITopology2d::operator=(const aMPITopology2d&)
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
    ///@copydoc aMPITopology3d::aMPITopology3d()
    aMPIGeometry3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        aMPITopology3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    ///@copydoc aMPITopology3d::aMPITopology3d(const aMPITopology3d&)
    aMPIGeometry3d( const aMPIGeometry3d& src):aMPITopology3d(src){}
    ///@copydoc aMPITopology3d::operator=(const aMPITopology3d&)
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
 * @brief The mpi version of CartesianGrid2d
 */
struct CartesianMPIGrid2d : public aMPIGeometry2d
{
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_comm_parameters2d
    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): aMPIGeometry2d( x0, x1, y0, y1, n, Nx, Ny, dg::PER,dg::PER,comm){}

    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    ///@copydoc hide_comm_parameters2d
    CartesianMPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):dg::aMPIGeometry2d( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    ///@brief Implicit type conversion from MPIGrid2d
    ///@param g existing grid object
    CartesianMPIGrid2d( const dg::MPIGrid2d& g): aMPIGeometry2d( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().bcx(),g.global().bcy(),g.communicator()){}
    virtual CartesianMPIGrid2d* clone()const{return new CartesianMPIGrid2d(*this);}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny){
        aMPITopology2d::do_set(new_n,new_Nx,new_Ny);
    }

};

/**
 * @brief The mpi version of CartesianGrid3d
 */
struct CartesianMPIGrid3d : public aMPIGeometry3d
{
    typedef CartesianMPIGrid2d perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_comm_parameters3d
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, dg::PER,dg::PER,dg::PER, comm){}

    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    CartesianMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}

    ///@brief Implicit type conversion from MPIGrid3d
    ///@param g existing grid object
    CartesianMPIGrid3d( const dg::MPIGrid3d& g): aMPIGeometry3d( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().z0(),g.global().z1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().Nz(),g.global().bcx(),g.global().bcy(),g.global().bcz(),g.communicator()){}
    virtual CartesianMPIGrid3d* clone()const{return new CartesianMPIGrid3d(*this);}
    /*!
     * @brief The grid made up by the first two dimensions in space and process topology
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid with the perpendicular communicator
     */
    CartesianMPIGrid2d perp_grid()const{ 
        return CartesianMPIGrid2d( global().x0(), global().x1(), global().y0(), global().y1(), global().n(), global().Nx(), global().Ny(), global().bcx(), global().bcy(), get_perp_comm( communicator() ));
    }

    private:
    MPI_Comm get_perp_comm( MPI_Comm src) const
    {
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( src, remain_dims, &planeComm);
        return planeComm;
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aMPITopology3d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

/**
 * @brief the mpi version of CylindricalGrid3d
 */
struct CylindricalMPIGrid3d: public aMPIGeometry3d
{
    typedef CartesianMPIGrid2d perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    CylindricalMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters2d
    ///@note bcz is dg::PER
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    CylindricalMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, MPI_Comm comm):aMPIGeometry3d( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, dg::PER, comm){}

    virtual CylindricalMPIGrid3d* clone()const{return new CylindricalMPIGrid3d(*this);}
    ///@copydoc CartesianMPIGrid3d::perp_grid()const
    CartesianMPIGrid2d perp_grid()const{ 
        return CartesianMPIGrid2d( global().x0(), global().x1(), global().y0(), global().y1(), global().n(), global().Nx(), global().Ny(), global().bcx(), global().bcy(), get_perp_comm( communicator() ));
    }
    private:
    MPI_Comm get_perp_comm( MPI_Comm src) const
    {
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( src, remain_dims, &planeComm);
        return planeComm;
    }
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
