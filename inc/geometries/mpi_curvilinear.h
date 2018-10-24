#pragma once

#include <mpi.h>

#include "dg/topology/mpi_evaluation.h"
#include "dg/topology/mpi_grid.h"
#include "dg/topology/mpi_base.h"
#include "curvilinear.h"
#include "generator.h"

namespace dg
{
namespace geo
{

///@cond
template<class real_type>
struct RealCurvilinearProductMPIGrid3d;
///@endcond
//
///@addtogroup grids
///@{
/**
 * @brief A two-dimensional MPI grid based on curvilinear coordinates
 */
template<class real_type>
struct RealCurvilinearMPIGrid2d : public dg::aRealMPIGeometry2d<real_type>
{
    /// @copydoc hide_grid_parameters2d
    /// @param comm a two-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters
    RealCurvilinearMPIGrid2d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy, MPI_Comm comm):
        dg::aRealMPIGeometry2d<real_type>( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, bcy, comm), handle_(generator)
    {
        //generate global 2d grid and then reduce to local
        RealCurvilinearGrid2d<real_type> g(generator, n, Nx, Ny);
        divide_and_conquer(g);
    }
    ///explicit conversion of 3d product grid to the perpendicular grid
    explicit RealCurvilinearMPIGrid2d( const RealCurvilinearProductMPIGrid3d<real_type>& g);

    ///read access to the generator
    const aRealGenerator2d<real_type>& generator() const{return handle_.get();}
    virtual RealCurvilinearMPIGrid2d* clone()const override final{return new RealCurvilinearMPIGrid2d(*this);}
    virtual RealCurvilinearGrid2d<real_type>* global_geometry()const override final{
        return new RealCurvilinearGrid2d<real_type>(
                handle_.get(),
                global().n(), global().Nx(), global().Ny(),
                global().bcx(), global().bcy());
    }
    //These are necessary to help compiler find inherited names
    using typename dg::aRealMPIGeometry2d<real_type>::host_vector;
    using dg::aRealMPIGeometry2d<real_type>::global;
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny) override final
    {
        dg::aRealMPITopology2d<real_type>::do_set(new_n, new_Nx, new_Ny);
        RealCurvilinearGrid2d<real_type> g( handle_.get(), new_n, new_Nx, new_Ny);
        divide_and_conquer(g);//distribute to processes
    }
    void divide_and_conquer(const RealCurvilinearGrid2d<real_type>& g_)
    {
        dg::SparseTensor<thrust::host_vector<real_type> > jacobian=g_.jacobian();
        dg::SparseTensor<thrust::host_vector<real_type> > metric=g_.metric();
        std::vector<thrust::host_vector<real_type> > map = g_.map();
        for( unsigned i=0; i<3; i++)
            for( unsigned j=0; j<3; j++)
            {
                metric_.idx(i,j) = metric.idx(i,j);
                jac_.idx(i,j) = jacobian.idx(i,j);
            }
        jac_.values().resize( jacobian.values().size());
        for( unsigned i=0; i<jacobian.values().size(); i++)
            jac_.values()[i] = global2local( jacobian.values()[i], *this);
        metric_.values().resize( metric.values().size());
        for( unsigned i=0; i<metric.values().size(); i++)
            metric_.values()[i] = global2local( metric.values()[i], *this);
        map_.resize(map.size());
        for( unsigned i=0; i<map.size(); i++)
            map_[i] = global2local( map[i], *this);
    }

    virtual SparseTensor<host_vector> do_compute_jacobian( ) const override final{
        return jac_;
    }
    virtual SparseTensor<host_vector> do_compute_metric( ) const override final{
        return metric_;
    }
    virtual std::vector<host_vector > do_compute_map()const override final{return map_;}
    dg::SparseTensor<host_vector > jac_, metric_;
    std::vector<host_vector > map_;
    dg::ClonePtr<aRealGenerator2d<real_type>> handle_;
};

/**
 * @brief A 2x1 curvilinear product space MPI grid

 * The base coordinate system is the cylindrical coordinate system R,Z,phi
 */
template<class real_type>
struct RealCurvilinearProductMPIGrid3d : public dg::aRealProductMPIGeometry3d<real_type>
{
    typedef dg::geo::RealCurvilinearMPIGrid2d<real_type> perpendicular_grid; //!< the two-dimensional grid
    /// @copydoc hide_grid_parameters3d
    /// @param comm a three-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters
    RealCurvilinearProductMPIGrid3d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        dg::aRealProductMPIGeometry3d<real_type>( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz, comm),
        handle_( generator)
    {
        map_.resize(3);
        RealCurvilinearMPIGrid2d<real_type> g(generator,n,Nx,Ny, bcx, bcy, this->get_perp_comm());
        constructPerp( g);
        constructParallel(this->local().Nz());
    }


    ///read access to the generator
    const aRealGenerator2d<real_type>& generator() const{return handle_.get();}
    virtual RealCurvilinearProductMPIGrid3d* clone()const{return new RealCurvilinearProductMPIGrid3d(*this);}
    virtual RealCurvilinearProductGrid3d<real_type>* global_geometry()const{
        return new RealCurvilinearProductGrid3d<real_type>(
                handle_.get(),
                global().n(), global().Nx(), global().Ny(), global().Nz(),
                global().bcx(), global().bcy(), global().bcz());
    }
    //These are necessary to help compiler find inherited names
    using typename dg::aRealMPIGeometry3d<real_type>::host_vector;
    using dg::aRealMPIGeometry3d<real_type>::global;
    private:
    virtual perpendicular_grid* do_perp_grid() const override final{ return new perpendicular_grid(*this);}
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final
    {
        dg::aRealMPITopology3d<real_type>::do_set(new_n, new_Nx, new_Ny, new_Nz);
        if( !( new_n == this->n() && new_Nx == global().Nx() && new_Ny == global().Ny() ) )
        {
            RealCurvilinearMPIGrid2d<real_type> g(handle_.get(),new_n,new_Nx,new_Ny, this->bcx(), this->bcy(), this->get_perp_comm());
            constructPerp( g);
        }
        constructParallel(this->local().Nz());
    }
    void constructPerp( RealCurvilinearMPIGrid2d<real_type>& g2d)
    {
        jac_=g2d.jacobian();
        map_=g2d.map();
    }
    void constructParallel( unsigned localNz )
    {
        map_.resize(3);
        map_[2]=dg::evaluate(dg::cooZ3d, *this);
        unsigned size = this->local().size();
        unsigned size2d = this->n()*this->n()*this->local().Nx()*this->local().Ny();
        //resize for 3d values
        for( unsigned r=0; r<6;r++)
        {
            jac_.values()[r].data().resize(size);
            jac_.values()[r].set_communicator( this->communicator());
        }
        map_[0].data().resize(size);
        map_[0].set_communicator( this->communicator());
        map_[1].data().resize(size);
        map_[1].set_communicator( this->communicator());
        //lift to 3D grid
        for( unsigned k=1; k<localNz; k++)
            for( unsigned i=0; i<size2d; i++)
            {
                for(unsigned r=0; r<6; r++)
                    jac_.values()[r].data()[k*size2d+i] = jac_.values()[r].data()[(k-1)*size2d+i];
                map_[0].data()[k*size2d+i] = map_[0].data()[(k-1)*size2d+i];
                map_[1].data()[k*size2d+i] = map_[1].data()[(k-1)*size2d+i];
            }
    }
    virtual SparseTensor<host_vector> do_compute_jacobian( ) const override final{
        return jac_;
    }
    virtual SparseTensor<host_vector> do_compute_metric( ) const override final{
        return detail::square( jac_, map_[0], handle_.get().isOrthogonal());
    }
    virtual std::vector<host_vector > do_compute_map()const override final{return map_;}
    dg::SparseTensor<host_vector > jac_;
    std::vector<host_vector > map_;
    ClonePtr<dg::geo::aRealGenerator2d<real_type>> handle_;
};
///@cond
template<class real_type>
RealCurvilinearMPIGrid2d<real_type>::RealCurvilinearMPIGrid2d( const RealCurvilinearProductMPIGrid3d<real_type>& g):
    dg::aRealMPIGeometry2d<real_type>( g.global().x0(), g.global().x1(), g.global().y0(), g.global().y1(), g.global().n(), g.global().Nx(), g.global().Ny(), g.global().bcx(), g.global().bcy(), g.get_perp_comm() ),
    handle_(g.generator())
{
    map_=g.map();
    jac_=g.jacobian();
    metric_=g.metric();
    //now resize to 2d
    map_.pop_back();
    unsigned s = this->local().size();
    for( unsigned i=0; i<jac_.values().size(); i++)
        jac_.values()[i].data().resize(s);
    for( unsigned i=0; i<metric_.values().size(); i++)
        metric_.values()[i].data().resize(s);
    for( unsigned i=0; i<map_.size(); i++)
        map_[i].data().resize(s);
}
///@endcond
//
using CurvilinearMPIGrid2d         = dg::geo::RealCurvilinearMPIGrid2d<double>;
using CurvilinearProductMPIGrid3d  = dg::geo::RealCurvilinearProductMPIGrid3d<double>;

///@}
}//namespace geo
}//namespace dg

