#pragma once

#include "mpi_grid.h"
#include "base_geometry.h"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

/**
 * @brief This is the abstract interface class for a two-dimensional Geometry
 */
template<class real_type>
struct aBasicMPIGeometry2d : public aBasicMPITopology2d<real_type>
{
    typedef MPI_Vector<thrust::host_vector<real_type> > host_vector;
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
    virtual aBasicMPIGeometry2d* clone()const=0;
    ///Construct the global non-MPI geometry
    virtual aBasicGeometry2d<real_type>* global_geometry()const =0;
    ///allow deletion through base class pointer
    virtual ~aBasicMPIGeometry2d(){}
    protected:
    ///@copydoc aBasicMPITopology2d<real_type>::aBasicMPITopology2d<real_type>()
    aBasicMPIGeometry2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):
        aBasicMPITopology2d<real_type>( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy, comm)
    { }
    ///@copydoc aBasicMPITopology2d<real_type>::aBasicMPITopology2d<real_type>(const aBasicMPITopology2d<real_type>&)
    aBasicMPIGeometry2d( const aBasicMPIGeometry2d& src):aBasicMPITopology2d<real_type>(src){}
    ///@copydoc aBasicMPITopology2d<real_type>::operator=(const aBasicMPITopology2d<real_type>&)
    aBasicMPIGeometry2d& operator=( const aBasicMPIGeometry2d& src){
        aBasicMPITopology2d<real_type>::operator=(src);
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
template<class real_type>
struct aBasicMPIGeometry3d : public aBasicMPITopology3d<real_type>
{
    typedef MPI_Vector<thrust::host_vector<real_type> > host_vector;
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
    virtual aBasicMPIGeometry3d* clone()const=0;
    ///Construct the global non-MPI geometry
    virtual aBasicGeometry3d<real_type>* global_geometry()const =0;
    ///allow deletion through base class pointer
    virtual ~aBasicMPIGeometry3d(){}
    protected:
    ///@copydoc aBasicMPITopology3d<real_type>::aBasicMPITopology3d<real_type>()
    aBasicMPIGeometry3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        aBasicMPITopology3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    ///@copydoc aBasicMPITopology3d<real_type>::aBasicMPITopology3d<real_type>(const aBasicMPITopology3d<real_type>&)
    aBasicMPIGeometry3d( const aBasicMPIGeometry3d& src):aBasicMPITopology3d<real_type>(src){}
    ///@copydoc aBasicMPITopology3d<real_type>::operator=(const aBasicMPITopology3d<real_type>&)
    aBasicMPIGeometry3d& operator=( const aBasicMPIGeometry3d& src){
        aBasicMPITopology3d<real_type>::operator=(src);
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

///@brief a 3d product space MPI Geometry
template<class real_type>
struct aBasicProductMPIGeometry3d : public aBasicMPIGeometry3d<real_type>
{
    /*!
     * @brief The grid made up by the first two dimensions
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid
     */
    aBasicMPIGeometry2d<real_type>* perp_grid()const{
        return do_perp_grid();
    }
    ///allow deletion through base class pointer
    virtual ~aBasicProductMPIGeometry3d(){}
    ///Geometries are cloneable
    virtual aBasicProductMPIGeometry3d* clone()const=0;
    protected:
    ///@copydoc aBasicMPITopology3d<real_type>::aBasicMPITopology3d<real_type>()
    aBasicProductMPIGeometry3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        aBasicMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    ///@copydoc aBasicMPITopology3d<real_type>::aBasicMPITopology3d<real_type>(const aBasicMPITopology3d<real_type>&)
    aBasicProductMPIGeometry3d( const aBasicProductMPIGeometry3d& src):aBasicMPIGeometry3d<real_type>(src){}
    ///@copydoc aBasicMPITopology3d<real_type>::operator=(const aBasicMPITopology3d<real_type>&)
    aBasicProductMPIGeometry3d& operator=( const aBasicProductMPIGeometry3d& src){
        aBasicMPIGeometry3d<real_type>::operator=(src);
        return *this;
    }
    private:
    virtual aBasicMPIGeometry2d<real_type>* do_perp_grid()const=0;
};

using aMPIGeometry2d = aBasicMPIGeometry2d<double>;
using aMPIGeometry3d = aBasicMPIGeometry3d<double>;
using aProductMPIGeometry3d = aBasicProductMPIGeometry3d<double>;
///@}

///@addtogroup geometry
///@{

/**
 * @brief The mpi version of BasicCartesianGrid2d
 */
template<class real_type>
struct BasicCartesianMPIGrid2d : public aBasicMPIGeometry2d<real_type>
{
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_comm_parameters2d
    BasicCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): aBasicMPIGeometry2d<real_type>( x0, x1, y0, y1, n, Nx, Ny, dg::PER,dg::PER,comm){}

    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    ///@copydoc hide_comm_parameters2d
    BasicCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):dg::aBasicMPIGeometry2d<real_type>( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    ///@brief Implicit type conversion from MPIGrid2d
    ///@param g existing grid object
    BasicCartesianMPIGrid2d( const dg::BasicMPIGrid2d<real_type>& g): aBasicMPIGeometry2d<real_type>( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().bcx(),g.global().bcy(),g.communicator()){}
    virtual BasicCartesianMPIGrid2d* clone()const{return new BasicCartesianMPIGrid2d(*this);}
    virtual BasicCartesianGrid2d<real_type>* global_geometry()const{
        return new BasicCartesianGrid2d<real_type>(
                this->global().x0(), this->global().x1(),
                this->global().y0(), this->global().y1(),
                this->global().n(),  this->global().Nx(), this->global().Ny(),
                this->global().bcx(), this->global().bcy());
    }
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny){
        aBasicMPITopology2d<real_type>::do_set(new_n,new_Nx,new_Ny);
    }

};

/**
 * @brief The mpi version of BasicCartesianGrid3d
 */
template<class real_type>
struct BasicCartesianMPIGrid3d : public aBasicProductMPIGeometry3d<real_type>
{
    typedef BasicCartesianMPIGrid2d<real_type> perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_comm_parameters3d
    BasicCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): aBasicProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, dg::PER,dg::PER,dg::PER, comm){}

    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    BasicCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aBasicProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}

    ///@brief Implicit type conversion from MPIGrid3d
    ///@param g existing grid object
    BasicCartesianMPIGrid3d( const dg::BasicMPIGrid3d<real_type>& g): aBasicProductMPIGeometry3d<real_type>( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().z0(),g.global().z1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().Nz(),g.global().bcx(),g.global().bcy(),g.global().bcz(),g.communicator()){}
    virtual BasicCartesianMPIGrid3d* clone()const{return new BasicCartesianMPIGrid3d(*this);}
    virtual BasicCartesianGrid3d<real_type>* global_geometry()const{
        return new BasicCartesianGrid3d<real_type>(
                this->global().x0(), this->global().x1(),
                this->global().y0(), this->global().y1(),
                this->global().z0(), this->global().z1(),
                this->global().n(), this->global().Nx(), this->global().Ny(), this->global().Nz(),
                this->global().bcx(), this->global().bcy(), this->global().bcz());
    }

    private:
    virtual BasicCartesianMPIGrid2d<real_type>* do_perp_grid()const{
        return new BasicCartesianMPIGrid2d<real_type>( this->global().x0(), this->global().x1(), this->global().y0(), this->global().y1(), this->global().n(), this->global().Nx(), this->global().Ny(), this->global().bcx(), this->global().bcy(), this->get_perp_comm( ));
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aBasicMPITopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

/**
 * @brief the mpi version of BasicCylindricalGrid3d
 */
template<class real_type>
struct BasicCylindricalMPIGrid3d: public aBasicProductMPIGeometry3d<real_type>
{
    typedef BasicCartesianMPIGrid2d<real_type> perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    BasicCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aBasicProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters2d
    ///@note bcz is dg::PER
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    BasicCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, MPI_Comm comm):aBasicProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, dg::PER, comm){}

    virtual BasicCylindricalMPIGrid3d<real_type>* clone()const{return new BasicCylindricalMPIGrid3d(*this);}
    virtual BasicCylindricalGrid3d<real_type>* global_geometry()const{
        return new BasicCylindricalGrid3d<real_type>(
                this->global().x0(), this->global().x1(),
                this->global().y0(), this->global().y1(),
                this->global().z0(), this->global().z1(),
                this->global().n(), this->global().Nx(), this->global().Ny(), this->global().Nz(),
                this->global().bcx(), this->global().bcy(), this->global().bcz());
    }
    private:
    using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
    virtual BasicCartesianMPIGrid2d<real_type>* do_perp_grid()const override final{
        return new BasicCartesianMPIGrid2d<real_type>( this->global().x0(), this->global().x1(), this->global().y0(), this->global().y1(), this->global().n(), this->global().Nx(), this->global().Ny(), this->global().bcx(), this->global().bcy(), this->get_perp_comm( ));
    }
    virtual SparseTensor<host_vector > do_compute_metric()const override final{
        SparseTensor<host_vector> metric(1);
        host_vector R = dg::evaluate(dg::cooX3d, *this);
        for( unsigned i = 0; i<this->local().size(); i++)
            R.data()[i] = 1./R.data()[i]/R.data()[i];
        metric.idx(2,2)=0;
        metric.values()[0] = R;
        return metric;
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final{
        aBasicMPITopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};
using CartesianMPIGrid2d = BasicCartesianMPIGrid2d<double>;
using CartesianMPIGrid3d = BasicCartesianMPIGrid3d<double>;
using CylindricalMPIGrid3d = BasicCylindricalMPIGrid3d<double>;

///@}

}//namespace dg
