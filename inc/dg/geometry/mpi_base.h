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
struct aRealMPIGeometry2d : public aRealMPITopology2d<real_type>
{
    typedef MPI_Vector<thrust::host_vector<real_type> > host_vector;
    ///@copydoc aRealGeometry2d::jacobian()
    SparseTensor<host_vector > jacobian()const {
        return do_compute_jacobian();
    }
    ///@copydoc aRealGeometry2d::metric()
    SparseTensor<host_vector > metric()const {
        return do_compute_metric();
    }
    ///@copydoc aRealGeometry2d::map()
    std::vector<host_vector > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aRealMPIGeometry2d* clone()const=0;
    ///Construct the global non-MPI geometry
    virtual aRealGeometry2d<real_type>* global_geometry()const =0;
    ///allow deletion through base class pointer
    virtual ~aRealMPIGeometry2d() = default;
    protected:
    using aRealMPITopology2d<real_type>::aRealMPITopology2d;
    ///@copydoc aRealMPITopology2d::aRealMPITopology2d(const aRealMPITopology2d&)
    aRealMPIGeometry2d( const aRealMPIGeometry2d& src) = default;
    ///@copydoc aRealMPITopology2d::operator=(const aRealMPITopology2d&)
    aRealMPIGeometry2d& operator=( const aRealMPIGeometry2d& src) = default;
    private:
    virtual SparseTensor<host_vector > do_compute_metric()const {
        return SparseTensor<host_vector >(*this);
    }
    virtual SparseTensor<host_vector > do_compute_jacobian()const {
        return SparseTensor<host_vector >(*this);
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
struct aRealMPIGeometry3d : public aRealMPITopology3d<real_type>
{
    typedef MPI_Vector<thrust::host_vector<real_type> > host_vector;
    ///@copydoc aRealGeometry3d::jacobian()
    SparseTensor<host_vector > jacobian()const{
        return do_compute_jacobian();
    }
    ///@copydoc aRealGeometry3d::metric()
    SparseTensor<host_vector > metric()const {
        return do_compute_metric();
    }
    ///@copydoc aRealGeometry3d::map()
    std::vector<host_vector > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aRealMPIGeometry3d* clone()const=0;
    ///Construct the global non-MPI geometry
    virtual aRealGeometry3d<real_type>* global_geometry()const =0;
    ///allow deletion through base class pointer
    virtual ~aRealMPIGeometry3d() = default;
    protected:
    using aRealMPITopology3d<real_type>::aRealMPITopology3d;
    ///@copydoc aRealMPITopology3d::aRealMPITopology3d(const aRealMPITopology3d&)
    aRealMPIGeometry3d( const aRealMPIGeometry3d& src) = default;
    ///@copydoc aRealMPITopology3d::operator=(const aRealMPITopology3d&)
    aRealMPIGeometry3d& operator=( const aRealMPIGeometry3d& src) = default;
    private:
    virtual SparseTensor<host_vector > do_compute_metric()const {
        return SparseTensor<host_vector >(*this);
    }
    virtual SparseTensor<host_vector > do_compute_jacobian()const {
        return SparseTensor<host_vector >(*this);
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
struct aRealProductMPIGeometry3d : public aRealMPIGeometry3d<real_type>
{
    /*!
     * @brief The grid made up by the first two dimensions
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid
     */
    aRealMPIGeometry2d<real_type>* perp_grid()const{
        return do_perp_grid();
    }
    ///allow deletion through base class pointer
    virtual ~aRealProductMPIGeometry3d() = default;
    ///Geometries are cloneable
    virtual aRealProductMPIGeometry3d* clone()const=0;
    protected:
    using aRealMPIGeometry3d<real_type>::aRealMPIGeometry3d;
    ///@copydoc aRealMPITopology3d::aRealMPITopology3d(const aRealMPITopology3d&)
    aRealProductMPIGeometry3d( const aRealProductMPIGeometry3d& src) = default;
    ///@copydoc aRealMPITopology3d::operator=(const aRealMPITopology3d&)
    aRealProductMPIGeometry3d& operator=( const aRealProductMPIGeometry3d& src) = default;
    private:
    virtual aRealMPIGeometry2d<real_type>* do_perp_grid()const=0;
};

///@}

///@addtogroup geometry
///@{

/**
 * @brief The mpi version of RealCartesianGrid2d
 */
template<class real_type>
struct RealCartesianMPIGrid2d : public aRealMPIGeometry2d<real_type>
{
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_comm_parameters2d
    RealCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): aRealMPIGeometry2d<real_type>( x0, x1, y0, y1, n, Nx, Ny, dg::PER,dg::PER,comm){}

    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    ///@copydoc hide_comm_parameters2d
    RealCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):dg::aRealMPIGeometry2d<real_type>( x0, x1, y0, y1, n, Nx, Ny,bcx, bcy, comm){}
    ///@brief Implicit type conversion from MPIGrid2d
    ///@param g existing grid object
    RealCartesianMPIGrid2d( const dg::RealMPIGrid2d<real_type>& g): aRealMPIGeometry2d<real_type>( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().bcx(),g.global().bcy(),g.communicator()){}
    virtual RealCartesianMPIGrid2d* clone()const override final{return new RealCartesianMPIGrid2d(*this);}
    virtual RealCartesianGrid2d<real_type>* global_geometry()const override final{
        return new RealCartesianGrid2d<real_type>(
                this->global().x0(), this->global().x1(),
                this->global().y0(), this->global().y1(),
                this->global().n(),  this->global().Nx(), this->global().Ny(),
                this->global().bcx(), this->global().bcy());
    }
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny) override final{
        aRealMPITopology2d<real_type>::do_set(new_n,new_Nx,new_Ny);
    }

};

/**
 * @brief The mpi version of RealCartesianGrid3d
 */
template<class real_type>
struct RealCartesianMPIGrid3d : public aRealProductMPIGeometry3d<real_type>
{
    typedef RealCartesianMPIGrid2d<real_type> perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_comm_parameters3d
    RealCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): aRealProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, dg::PER,dg::PER,dg::PER, comm){}

    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    RealCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aRealProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}

    ///@brief Implicit type conversion from RealMPIGrid3d
    ///@param g existing grid object
    RealCartesianMPIGrid3d( const dg::RealMPIGrid3d<real_type>& g): aRealProductMPIGeometry3d<real_type>( g.global().x0(),g.global().x1(),g.global().y0(),g.global().y1(),g.global().z0(),g.global().z1(),g.global().n(),g.global().Nx(),g.global().Ny(),g.global().Nz(),g.global().bcx(),g.global().bcy(),g.global().bcz(),g.communicator()){}
    virtual RealCartesianMPIGrid3d* clone()const override final{
        return new RealCartesianMPIGrid3d(*this);
    }
    virtual RealCartesianGrid3d<real_type>* global_geometry()const override final{
        return new RealCartesianGrid3d<real_type>(
                this->global().x0(), this->global().x1(),
                this->global().y0(), this->global().y1(),
                this->global().z0(), this->global().z1(),
                this->global().n(), this->global().Nx(), this->global().Ny(), this->global().Nz(),
                this->global().bcx(), this->global().bcy(), this->global().bcz());
    }

    private:
    virtual RealCartesianMPIGrid2d<real_type>* do_perp_grid()const override final{
        return new RealCartesianMPIGrid2d<real_type>( this->global().x0(), this->global().x1(), this->global().y0(), this->global().y1(), this->global().n(), this->global().Nx(), this->global().Ny(), this->global().bcx(), this->global().bcy(), this->get_perp_comm( ));
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz)override final{
        aRealMPITopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

/**
 * @brief the mpi version of RealCylindricalGrid3d
 */
template<class real_type>
struct RealCylindricalMPIGrid3d: public aRealProductMPIGeometry3d<real_type>
{
    typedef RealCartesianMPIGrid2d<real_type> perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    RealCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aRealProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm){}
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters2d
    ///@note bcz is dg::PER
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    RealCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, MPI_Comm comm):aRealProductMPIGeometry3d<real_type>( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, dg::PER, comm){}

    virtual RealCylindricalMPIGrid3d<real_type>* clone()const override final{
        return new RealCylindricalMPIGrid3d(*this);
    }
    virtual RealCylindricalGrid3d<real_type>* global_geometry()const override final{
        return new RealCylindricalGrid3d<real_type>(
                this->global().x0(), this->global().x1(),
                this->global().y0(), this->global().y1(),
                this->global().z0(), this->global().z1(),
                this->global().n(), this->global().Nx(), this->global().Ny(), this->global().Nz(),
                this->global().bcx(), this->global().bcy(), this->global().bcz());
    }
    private:
    using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
    virtual RealCartesianMPIGrid2d<real_type>* do_perp_grid()const override final{
        return new RealCartesianMPIGrid2d<real_type>( this->global().x0(), this->global().x1(), this->global().y0(), this->global().y1(), this->global().n(), this->global().Nx(), this->global().Ny(), this->global().bcx(), this->global().bcy(), this->get_perp_comm( ));
    }
    virtual SparseTensor<host_vector > do_compute_metric()const override final{
        SparseTensor<host_vector> metric(*this);
        host_vector R = dg::evaluate(dg::cooX3d, *this);
        for( unsigned i = 0; i<this->local().size(); i++)
            R.data()[i] = 1./R.data()[i]/R.data()[i];
        metric.idx(2,2)=2;
        metric.values().push_back(R);
        return metric;
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final{
        aRealMPITopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

///@}
///@addtogroup gridtypes
///@{
using aMPIGeometry2d        = dg::aRealMPIGeometry2d<double>;
using aMPIGeometry3d        = dg::aRealMPIGeometry3d<double>;
using aProductMPIGeometry3d = dg::aRealProductMPIGeometry3d<double>;
using CartesianMPIGrid2d    = dg::RealCartesianMPIGrid2d<double>;
using CartesianMPIGrid3d    = dg::RealCartesianMPIGrid3d<double>;
using CylindricalMPIGrid3d  = dg::RealCylindricalMPIGrid3d<double>;
///@}

}//namespace dg
