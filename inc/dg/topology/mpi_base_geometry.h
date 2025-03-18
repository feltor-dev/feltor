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
    ///@copydoc aRealGeometry2d::jacobian()
    SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > jacobian()const {
        return do_compute_jacobian();
    }
    ///@copydoc aRealGeometry2d::metric()
    SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > metric()const {
        return do_compute_metric();
    }
    ///@copydoc aRealGeometry2d::map()
    std::vector<MPI_Vector<thrust::host_vector<real_type>> > map()const{
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
    ///@copydoc aRealMPITopology::aRealMPITopology(const aRealMPITopology&)
    aRealMPIGeometry2d( const aRealMPIGeometry2d& src) = default;
    ///@copydoc aRealMPITopology::operator=
    aRealMPIGeometry2d& operator=( const aRealMPIGeometry2d& src) = default;
    private:
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_metric()const {
        return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
    }
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_jacobian()const {
        return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
    }
    virtual std::vector<MPI_Vector<thrust::host_vector<real_type>> > do_compute_map()const{
        std::vector<MPI_Vector<thrust::host_vector<real_type>>> map(2);
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
    ///@copydoc aRealGeometry3d::jacobian()
    SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > jacobian()const{
        return do_compute_jacobian();
    }
    ///@copydoc aRealGeometry3d::metric()
    SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > metric()const {
        return do_compute_metric();
    }
    ///@copydoc aRealGeometry3d::map()
    std::vector<MPI_Vector<thrust::host_vector<real_type>> > map()const{
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
    ///@copydoc aRealMPITopology::aRealMPITopology(const aRealMPITopology&)
    aRealMPIGeometry3d( const aRealMPIGeometry3d& src) = default;
    ///@copydoc aRealMPITopology::operator=
    aRealMPIGeometry3d& operator=( const aRealMPIGeometry3d& src) = default;
    private:
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_metric()const {
        return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
    }
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_jacobian()const {
        return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
    }
    virtual std::vector<MPI_Vector<thrust::host_vector<real_type>> > do_compute_map()const{
        std::vector<MPI_Vector<thrust::host_vector<real_type>>> map(3);
        map[0] = dg::evaluate(dg::cooX3d, *this);
        map[1] = dg::evaluate(dg::cooY3d, *this);
        map[2] = dg::evaluate(dg::cooZ3d, *this);
        return map;
    }
};

///@copydoc aRealProductGeometry3d
template<class real_type>
struct aRealProductMPIGeometry3d : public aRealMPIGeometry3d<real_type>
{
    /*!
     * @brief The grid made up by the first two dimensions
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid
     * @attention The user takes ownership of the newly allocated grid
     * @code
     * dg::ClonePtr<aRealMPIGeometry2d<real_type>> perp_ptr = grid.perp_grid();
     * @endcode
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
    ///@copydoc aRealMPITopology::aRealMPITopology(const aRealMPITopology&)
    aRealProductMPIGeometry3d( const aRealProductMPIGeometry3d& src) = default;
    ///@copydoc aRealMPITopology::operator=
    aRealProductMPIGeometry3d& operator=( const aRealProductMPIGeometry3d& src) = default;
    private:
    virtual aRealMPIGeometry2d<real_type>* do_perp_grid()const=0;
};

/**
 * @brief The mpi version of RealCartesianGrid2d
 */
template<class real_type>
struct RealCartesianMPIGrid2d : public aRealMPIGeometry2d<real_type>
{
    RealCartesianMPIGrid2d() = default;
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_comm_parameters2d
    RealCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1,
        unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm):
        aRealMPIGeometry2d<real_type>( {x0,y0},{x1,y1}, {n,n}, {Nx,Ny}, {dg::PER, dg::PER},
            dg::mpi_cart_split_as<2>(comm))
            {}

    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    ///@copydoc hide_comm_parameters2d
    RealCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1,
        unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):
        aRealMPIGeometry2d<real_type>( {x0,y0},{x1,y1}, {n,n}, {Nx,Ny}, {bcx, bcy},
            dg::mpi_cart_split_as<2>(comm))
            {}
    ///@copydoc RealCartesianGrid2d::RealCartesianGrid2d(RealGrid1d<real_type>,RealGrid1d<real_type>)
    RealCartesianMPIGrid2d( RealMPIGrid1d<real_type> gx, RealMPIGrid1d<real_type> gy):
        aRealMPIGeometry2d<real_type>(std::array{gx,gy})
        {}
    ///@brief Implicit type conversion from MPIGrid2d
    ///@param g existing grid object
    RealCartesianMPIGrid2d( const dg::RealMPIGrid2d<real_type>& g):
        aRealMPIGeometry2d<real_type>( std::array{g.gx(), g.gy()})
        {}
    virtual RealCartesianMPIGrid2d* clone()const override final{return new RealCartesianMPIGrid2d(*this);}
    virtual RealCartesianGrid2d<real_type>* global_geometry()const override final{
        return new RealCartesianGrid2d<real_type>(
                this->global().gx(), this->global().gy());
    }
    private:
    virtual void do_set(std::array<unsigned,2> new_n, std::array<unsigned,2> new_N) override final{
        aRealMPITopology2d<real_type>::do_set(new_n,new_N);
    }
    virtual void do_set_pq( std::array<real_type,2> new_x0, std::array<real_type,2> new_x1) override final{
        aRealMPITopology<real_type,2>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,2> new_bcs) override final{
        aRealMPITopology<real_type,2>::do_set(new_bcs);
    }

};

/**
 * @brief The mpi version of RealCartesianGrid3d
 */
template<class real_type>
struct RealCartesianMPIGrid3d : public aRealProductMPIGeometry3d<real_type>
{
    using perpendicular_grid = RealCartesianMPIGrid2d<real_type>;
    RealCartesianMPIGrid3d() = default;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_comm_parameters3d
    RealCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1,
        unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm):
        aRealProductMPIGeometry3d<real_type>(
            {x0,y0,z0},{x1,y1,z1}, {n,n,1}, {Nx,Ny,Nz}, {dg::PER, dg::PER,dg::PER},
            dg::mpi_cart_split_as<3>(comm)){}

    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    RealCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1,
        unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        aRealProductMPIGeometry3d<real_type>(
            {x0,y0,z0},{x1,y1,z1}, {n,n,1}, {Nx,Ny,Nz}, {bcx, bcy, bcz},
            dg::mpi_cart_split_as<3>(comm)){}

    ///@brief Implicit type conversion from RealMPIGrid3d
    ///@param g existing grid object
    RealCartesianMPIGrid3d( const dg::RealMPIGrid3d<real_type>& g):
        aRealProductMPIGeometry3d<real_type>( std::array{g.gx(), g.gy(), g.gz()}){}
    virtual RealCartesianMPIGrid3d* clone()const override final{
        return new RealCartesianMPIGrid3d(*this);
    }
    ///@copydoc RealCartesianGrid3d::RealCartesianGrid3d(RealGrid1d<real_type>,RealGrid1d<real_type>,RealGrid1d<real_type>)
    RealCartesianMPIGrid3d( RealMPIGrid1d<real_type> gx, RealMPIGrid1d<real_type> gy,
        RealMPIGrid1d<real_type> gz):
        dg::aRealProductMPIGeometry3d<real_type>(std::array{gx,gy,gz})
        {}
    virtual RealCartesianGrid3d<real_type>* global_geometry()const override final{
        return new RealCartesianGrid3d<real_type>(
                this->global().gx(), this->global().gy(), this->global().gz());
    }

    private:
    virtual RealCartesianMPIGrid2d<real_type>* do_perp_grid()const override final{
        return new RealCartesianMPIGrid2d<real_type>( this->gx(), this->gy());
    }
    virtual void do_set(std::array<unsigned,3> new_n, std::array<unsigned,3> new_N)override final{
        aRealMPITopology<real_type,3>::do_set(new_n,new_N);
    }
    virtual void do_set_pq( std::array<real_type,3> new_x0, std::array<real_type,3> new_x1) override final{
        aRealMPITopology<real_type,3>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,3> new_bcs) override final{
        aRealMPITopology<real_type,3>::do_set(new_bcs);
    }
};

/**
 * @brief the mpi version of RealCylindricalGrid3d
 */
template<class real_type>
struct RealCylindricalMPIGrid3d: public aRealProductMPIGeometry3d<real_type>
{
    using perpendicular_grid = RealCartesianMPIGrid2d<real_type>;
    RealCylindricalMPIGrid3d() = default;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    RealCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        aRealProductMPIGeometry3d<real_type>( {x0,y0,z0},{x1,y1,z1}, {n,n,1},{Nx,Ny,Nz},{bcx,bcy,bcz}, dg::mpi_cart_split_as<3>(comm)){}
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters2d
    ///@note bcz is dg::PER
    ///@copydoc hide_comm_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    RealCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, MPI_Comm comm):
        aRealProductMPIGeometry3d<real_type>( {x0,y0,z0},{x1,y1,z1}, {n,n,1},{Nx,Ny,Nz},{bcx,bcy,dg::PER}, dg::mpi_cart_split_as<3>(comm)){}

    ///@copydoc RealCylindricalGrid3d::RealCylindricalGrid3d(RealGrid1d<real_type>,RealGrid1d<real_type>,RealGrid1d<real_type>)
    RealCylindricalMPIGrid3d( RealMPIGrid1d<real_type> gx, RealMPIGrid1d<real_type> gy, RealMPIGrid1d<real_type> gz): dg::aRealProductMPIGeometry3d<real_type>(std::array{gx,gy,gz}){}

    virtual RealCylindricalMPIGrid3d<real_type>* clone()const override final{
        return new RealCylindricalMPIGrid3d(*this);
    }
    virtual RealCylindricalGrid3d<real_type>* global_geometry()const override final{
        return new RealCylindricalGrid3d<real_type>(
                this->global().gx(), this->global().gy(), this->global().gz());
    }
    private:
    virtual RealCartesianMPIGrid2d<real_type>* do_perp_grid()const override final{
        return new RealCartesianMPIGrid2d<real_type>( this->gx(), this->gy());
    }
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_metric()const override final{
        SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> metric(*this);
        MPI_Vector<thrust::host_vector<real_type>> R = dg::evaluate(dg::cooX3d, *this);
        for( unsigned i = 0; i<this->local().size(); i++)
            R.data()[i] = 1./R.data()[i]/R.data()[i];
        metric.idx(2,2)=2;
        metric.values().push_back(R);
        return metric;
    }
    virtual void do_set(std::array<unsigned,3> new_n, std::array<unsigned,3> new_N) override final{
        aRealMPITopology<real_type,3>::do_set(new_n,new_N);
    }
    virtual void do_set_pq( std::array<real_type,3> new_x0, std::array<real_type,3> new_x1) override final{
        aRealMPITopology<real_type,3>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,3> new_bcs) override final{
        aRealMPITopology<real_type,3>::do_set(new_bcs);
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
namespace x{
using aGeometry2d           = aMPIGeometry2d           ;
using aGeometry3d           = aMPIGeometry3d           ;
using aProductGeometry3d    = aProductMPIGeometry3d    ;
using CartesianGrid2d       = CartesianMPIGrid2d       ;
using CartesianGrid3d       = CartesianMPIGrid3d       ;
using CylindricalGrid3d     = CylindricalMPIGrid3d     ;
}//namespace x
///@}

}//namespace dg
