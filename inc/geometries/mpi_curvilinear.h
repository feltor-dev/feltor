#pragma once

#include <mpi.h>

#include "dg/algorithm.h"
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
        dg::aRealMPIGeometry2d<real_type>( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, bcy, comm), m_handle(generator)
    {
        //generate global 2d grid and then reduce to local
        RealCurvilinearGrid2d<real_type> g(generator, n, Nx, Ny);
        divide_and_conquer(g);
    }
    ///explicit conversion of 3d product grid to the perpendicular grid
    explicit RealCurvilinearMPIGrid2d( const RealCurvilinearProductMPIGrid3d<real_type>& g);

    ///read access to the generator
    const aRealGenerator2d<real_type>& generator() const{return *m_handle;}
    virtual RealCurvilinearMPIGrid2d* clone()const override final{return new RealCurvilinearMPIGrid2d(*this);}
    virtual RealCurvilinearGrid2d<real_type>* global_geometry()const override final{
        return new RealCurvilinearGrid2d<real_type>(
                *m_handle,
                global().n(), global().Nx(), global().Ny(),
                global().bcx(), global().bcy());
    }
    //These are necessary to help compiler find inherited names
    using dg::aRealMPIGeometry2d<real_type>::global;
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny) override final
    {
        dg::aRealMPITopology2d<real_type>::do_set(new_n, new_Nx, new_Ny);
        RealCurvilinearGrid2d<real_type> g( *m_handle, new_n, new_Nx, new_Ny);
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
                m_metric.idx(i,j) = metric.idx(i,j);
                m_jac.idx(i,j) = jacobian.idx(i,j);
            }
        // Here we set the communicator implicitly
        m_jac.values().resize( jacobian.values().size());
        for( unsigned i=0; i<jacobian.values().size(); i++)
            m_jac.values()[i] = global2local( jacobian.values()[i], *this);
        m_metric.values().resize( metric.values().size());
        for( unsigned i=0; i<metric.values().size(); i++)
            m_metric.values()[i] = global2local( metric.values()[i], *this);
        m_map.resize(map.size());
        for( unsigned i=0; i<map.size(); i++)
            m_map[i] = global2local( map[i], *this);
    }

    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_jacobian( ) const override final{
        return m_jac;
    }
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_metric( ) const override final{
        return m_metric;
    }
    virtual std::vector<MPI_Vector<thrust::host_vector<real_type>>> do_compute_map()const override final{return m_map;}
    dg::SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> m_jac, m_metric;
    std::vector<MPI_Vector<thrust::host_vector<real_type>>> m_map;
    dg::ClonePtr<aRealGenerator2d<real_type>> m_handle;
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
        m_handle( generator)
    {
        m_map.resize(3);
        RealCurvilinearMPIGrid2d<real_type> g(generator,n,Nx,Ny, bcx, bcy, this->get_perp_comm());
        constructPerp( g);
        constructParallel(this->local().Nz());
    }


    ///read access to the generator
    const aRealGenerator2d<real_type>& generator() const{return *m_handle;}
    virtual RealCurvilinearProductMPIGrid3d* clone()const{return new RealCurvilinearProductMPIGrid3d(*this);}
    virtual RealCurvilinearProductGrid3d<real_type>* global_geometry()const{
        return new RealCurvilinearProductGrid3d<real_type>(
                *m_handle,
                global().n(), global().Nx(), global().Ny(), global().Nz(),
                global().bcx(), global().bcy(), global().bcz());
    }
    //These are necessary to help compiler find inherited names
    using dg::aRealMPIGeometry3d<real_type>::global;
    private:
    virtual perpendicular_grid* do_perp_grid() const override final{ return new perpendicular_grid(*this);}
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final
    {
        dg::aRealMPITopology3d<real_type>::do_set(new_n, new_Nx, new_Ny, new_Nz);
        if( !( new_n == this->n() && new_Nx == global().Nx() && new_Ny == global().Ny() ) )
        {
            RealCurvilinearMPIGrid2d<real_type> g( *m_handle,new_n,new_Nx,new_Ny, this->bcx(), this->bcy(), this->get_perp_comm());
            constructPerp( g);
        }
        constructParallel(this->local().Nz());
    }
    void constructPerp( RealCurvilinearMPIGrid2d<real_type>& g2d)
    {
        m_jac=g2d.jacobian();
        m_map=g2d.map();
    }
    void constructParallel( unsigned localNz )
    {
        m_map.resize(3);
        m_map[2]=dg::evaluate(dg::cooZ3d, *this);
        unsigned size = this->local().size();
        unsigned size2d = this->n()*this->n()*this->local().Nx()*this->local().Ny();
        //resize for 3d values
        MPI_Comm comm = this->communicator(), comm_mod, comm_mod_reduce;
        exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_mod_reduce);
        for( unsigned r=0; r<6;r++)
        {
            m_jac.values()[r].data().resize(size);
            m_jac.values()[r].set_communicator( comm, comm_mod, comm_mod_reduce);
        }
        m_map[0].data().resize(size);
        m_map[0].set_communicator( comm, comm_mod, comm_mod_reduce);
        m_map[1].data().resize(size);
        m_map[1].set_communicator( comm, comm_mod, comm_mod_reduce);
        //lift to 3D grid
        for( unsigned k=1; k<localNz; k++)
            for( unsigned i=0; i<size2d; i++)
            {
                for(unsigned r=0; r<6; r++)
                    m_jac.values()[r].data()[k*size2d+i] = m_jac.values()[r].data()[(k-1)*size2d+i];
                m_map[0].data()[k*size2d+i] = m_map[0].data()[(k-1)*size2d+i];
                m_map[1].data()[k*size2d+i] = m_map[1].data()[(k-1)*size2d+i];
            }
    }
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_jacobian( ) const override final{
        return m_jac;
    }
    virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_metric( ) const override final{
        return detail::square( m_jac, m_map[0], m_handle->isOrthogonal());
    }
    virtual std::vector<MPI_Vector<thrust::host_vector<real_type>>> do_compute_map()const override final{return m_map;}
    dg::SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> m_jac;
    std::vector<MPI_Vector<thrust::host_vector<real_type>>> m_map;
    ClonePtr<dg::geo::aRealGenerator2d<real_type>> m_handle;
};
///@cond
template<class real_type>
RealCurvilinearMPIGrid2d<real_type>::RealCurvilinearMPIGrid2d( const RealCurvilinearProductMPIGrid3d<real_type>& g):
    dg::aRealMPIGeometry2d<real_type>( g.global().x0(), g.global().x1(), g.global().y0(), g.global().y1(), g.global().n(), g.global().Nx(), g.global().Ny(), g.global().bcx(), g.global().bcy(), g.get_perp_comm() ),
    m_handle(g.generator())
{
    m_map=g.map();
    m_jac=g.jacobian();
    m_metric=g.metric();
    //now resize to 2d
    m_map.pop_back();
    unsigned s = this->local().size();
    MPI_Comm comm = g.get_perp_comm(), comm_mod, comm_mod_reduce;
    exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_mod_reduce);
    for( unsigned i=0; i<m_jac.values().size(); i++)
    {
        m_jac.values()[i].data().resize(s);
        m_jac.values()[i].set_communicator( comm, comm_mod, comm_mod_reduce);
    }
    for( unsigned i=0; i<m_metric.values().size(); i++)
    {
        m_metric.values()[i].data().resize(s);
        m_metric.values()[i].set_communicator( comm, comm_mod, comm_mod_reduce);
    }
    // we rely on the fact that the 3d grid uses square to compute its metric
    // so the (2,2) entry is value 3 that we need to set to 1 (for the
    // create::volume function to work properly)
    dg::blas1::copy( 1., m_metric.values()[3]);
    for( unsigned i=0; i<m_map.size(); i++)
    {
        m_map[i].data().resize(s);
        m_map[i].set_communicator( comm, comm_mod, comm_mod_reduce);
    }
}
///@endcond
//
using CurvilinearMPIGrid2d         = dg::geo::RealCurvilinearMPIGrid2d<double>;
using CurvilinearProductMPIGrid3d  = dg::geo::RealCurvilinearProductMPIGrid3d<double>;
namespace x{
using CurvilinearGrid2d         = CurvilinearMPIGrid2d        ;
using CurvilinearProductGrid3d  = CurvilinearProductMPIGrid3d ;
}//namespace x

///@}
}//namespace geo
}//namespace dg

