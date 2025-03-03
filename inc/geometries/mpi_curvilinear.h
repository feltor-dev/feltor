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
    RealCurvilinearMPIGrid2d() = default;
    /// @copydoc hide_grid_parameters2d
    /// @param comm a two-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters
    RealCurvilinearMPIGrid2d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy, MPI_Comm comm):
        RealCurvilinearMPIGrid2d( generator, {n,Nx,bcx}, {n,Ny,bcy}, comm){}

    /// @copydoc hide_grid_product2d
    /// @param comm a two-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters
    RealCurvilinearMPIGrid2d( const aRealGenerator2d<real_type>& generator, Topology1d tx, Topology1d ty, MPI_Comm comm):
        dg::aRealMPIGeometry2d<real_type>( {0.,0.},
                {generator.width(), generator.height()},
                {tx.n,ty.n},{tx.N,ty.N}, { tx.b, ty.b},
                dg::mpi_cart_split_as<2>(comm)), m_handle(generator)
    {
        //generate global 2d grid and then reduce to local
        RealCurvilinearGrid2d<real_type> g(generator, tx, ty);
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
                {global().nx(), global().Nx(), global().bcx()},
                {global().ny(), global().Ny(), global().bcy()});
    }
    //These are necessary to help compiler find inherited names
    using dg::aRealMPIGeometry2d<real_type>::global;
    private:
    virtual void do_set(std::array<unsigned,2> new_n, std::array<unsigned,2> new_N) override final
    {
        dg::aRealMPITopology2d<real_type>::do_set( new_n, new_N);
        RealCurvilinearGrid2d<real_type> g( *m_handle,
                {new_n[0], new_N[0]}, {new_n[1], new_N[1]});
        divide_and_conquer(g);//distribute to processes
    }
    virtual void do_set(std::array<dg::bc,2> new_bc) override final
    {
        // TODO Do we change MPI periodic topology when we change bcs
        dg::aRealMPITopology2d<real_type>::do_set( new_bc);
    }
    virtual void do_set_pq(std::array<real_type,2> new_x0, std::array<real_type,2> new_x1) override final
    {
        throw dg::Error(dg::Message(_ping_)<<"This grid cannot change boundaries\n");
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
    RealCurvilinearProductMPIGrid3d() = default;
    /// @copydoc hide_grid_parameters3d
    /// @param comm a three-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters
    RealCurvilinearProductMPIGrid3d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        RealCurvilinearProductMPIGrid3d( generator, {n,Nx,bcx}, {n,Ny,bcy}, RealMPIGrid1d<real_type>{0.,2.*M_PI,1,Nz,bcz, dg::mpi_cart_split_as<3>(comm)[2]}, comm){}


    /// @copydoc hide_grid_product3d
    /// @param comm a three-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters
    RealCurvilinearProductMPIGrid3d( const aRealGenerator2d<real_type>& generator, Topology1d tx, Topology1d ty, RealMPIGrid1d<real_type> gz, MPI_Comm comm):
        dg::aRealProductMPIGeometry3d<real_type>(
                {0.,0., gz.x0()},{ generator.width(),
                generator.height(),gz.x1()}, {tx.n,ty.n, gz.n()},
                {tx.N, ty.N, gz.N()},{ tx.b, ty.b, gz.bcx()},
                dg::mpi_cart_split_as<3>(comm)), m_handle(generator)
    {
        m_map.resize(3);
        RealCurvilinearMPIGrid2d<real_type> g(generator,tx,ty,this->get_perp_comm());
        constructPerp( g);
        constructParallel(this->nz(), this->local().Nz());
    }


    ///read access to the generator
    const aRealGenerator2d<real_type>& generator() const{return *m_handle;}
    virtual RealCurvilinearProductMPIGrid3d* clone()const override final{return new RealCurvilinearProductMPIGrid3d(*this);}
    virtual RealCurvilinearProductGrid3d<real_type>* global_geometry()const{
        return new RealCurvilinearProductGrid3d<real_type>(
                *m_handle,
                {global().nx(), global().Nx(), global().bcx()},
                {global().ny(), global().Ny(), global().bcy()},
                {global().x0(), global().x1(), global().nz(), global().Nz(), global().bcz()});
    }
    //These are necessary to help compiler find inherited names
    using dg::aRealMPIGeometry3d<real_type>::global;
    private:
    virtual perpendicular_grid* do_perp_grid() const override final{ return new perpendicular_grid(*this);}
    virtual void do_set( std::array<unsigned,3> new_n, std::array<unsigned,3> new_N) override final
    {
        auto old_n = this->get_n(), old_N = this->get_N();
        dg::aRealMPITopology3d<real_type>::do_set( new_n, new_N);
        if( !( new_n[0] == old_n[0] && new_N[0] == old_N[0] &&
               new_n[1] == old_n[1] && new_N[1] == old_N[1] ) )
        {
            RealCurvilinearMPIGrid2d<real_type> g( *m_handle,
                    { new_n[0], new_N[0], this->bcx()},
                    { new_n[1], new_N[1], this->bcy()},
                    this->get_perp_comm());
            constructPerp( g);
        }
        constructParallel(this->nz(), this->local().Nz());
    }
    virtual void do_set(std::array<dg::bc,3> new_bc) override final
    {
        dg::aRealMPITopology3d<real_type>::do_set( new_bc);
    }
    virtual void do_set_pq(std::array<real_type,3> new_x0, std::array<real_type,3> new_x1) override final
    {
        throw dg::Error(dg::Message(_ping_)<<"This grid cannot change boundaries\n");
    }
    void constructPerp( RealCurvilinearMPIGrid2d<real_type>& g2d)
    {
        m_jac=g2d.jacobian();
        m_map=g2d.map();
    }
    void constructParallel( unsigned nz, unsigned localNz )
    {
        m_map.resize(3);
        m_map[2]=dg::evaluate(dg::cooZ3d, *this);
        unsigned size = this->local().size();
        unsigned size2d = this->nx()*this->ny()*this->local().Nx()*this->local().Ny();
        //resize for 3d values
        MPI_Comm comm = this->communicator();
        for( unsigned r=0; r<6;r++)
        {
            m_jac.values()[r].data().resize(size);
            m_jac.values()[r].set_communicator( comm);
        }
        m_map[0].data().resize(size);
        m_map[0].set_communicator( comm);
        m_map[1].data().resize(size);
        m_map[1].set_communicator( comm);
        //lift to 3D grid
        for( unsigned k=1; k<nz*localNz; k++)
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
    dg::aRealMPIGeometry2d<real_type>( std::array{g.gx(), g.gy()} ),
    m_handle(g.generator())
{
    m_map=g.map();
    m_jac=g.jacobian();
    m_metric=g.metric();
    //now resize to 2d
    m_map.pop_back();
    unsigned s = this->local().size();
    MPI_Comm comm = g.get_perp_comm();
    for( unsigned i=0; i<m_jac.values().size(); i++)
    {
        m_jac.values()[i].data().resize(s);
        m_jac.values()[i].set_communicator( comm);
    }
    for( unsigned i=0; i<m_metric.values().size(); i++)
    {
        m_metric.values()[i].data().resize(s);
        m_metric.values()[i].set_communicator( comm);
    }
    // we rely on the fact that the 3d grid uses square to compute its metric
    // so the (2,2) entry is value 3 that we need to set to 1 (for the
    // create::volume function to work properly)
    dg::blas1::copy( 1., m_metric.values()[3]);
    for( unsigned i=0; i<m_map.size(); i++)
    {
        m_map[i].data().resize(s);
        m_map[i].set_communicator( comm);
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

