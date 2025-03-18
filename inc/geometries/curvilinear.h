#pragma once

#include "dg/algorithm.h"
#include "generator.h"

namespace dg
{
namespace geo
{

/*!@class hide_grid_parameters3d
 * @brief Construct a 3D grid
 *
 * the coordinates of the computational space are called x,y,z
 * @param generator generate the perpendicular grid: the grid boundaries are [0, generator.width()] x [0, generator.height()] x [0, 2Pi]
 * @param n number of %Gaussian nodes in x and y (1<=n<=20 ), nz is set to 1
 * @param Nx number of cells in x
 * @param Ny number of cells in y
 * @param Nz  number of cells z
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 * @param bcz boundary condition in z
 *
 * @class hide_grid_product3d
 * @brief Construct the computational space as the product of three 1d grids
 *
 * @code
 * dg::CurvilinearGrid2d g2d( generator, {nx,Nx,bcx},{ny,Ny,bcy}, {z0,z1,nz,Nz,bcz});
 * @endcode
 * @param generator generate the grid: the grid boundaries are [0, generator.width()] x [0, generator.height()]
 * @param tx a Grid without boundaries in x - direction
 * @param ty a Grid without boundaries in y - direction
 * @param gz a Grid in z - direction
 */

/*!@class hide_grid_parameters2d
 * @brief Construct a 2D grid
 *
 * the coordinates of the computational space are called x,y
 * @param generator generate the grid: the grid boundaries are [0, generator.width()] x [0, generator.height()]
 * @param n number of %Gaussian nodes in x and y
 *  (1<=n<=20 )
 * @param Nx number of cells in x
 * @param Ny number of cells in y
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 *
 * @class hide_grid_product2d
 * @brief Construct the computational space as the product of two 1d grids
 *
 * @code
 * dg::CurvilinearGrid2d g2d( generator, {nx,Nx,bcx},{ny,Ny,bcy});
 * @endcode
 * @param generator generate the grid: the grid boundaries are [0, generator.width()] x [0, generator.height()]
 * @param tx a Grid without boundaries in x - direction
 * @param ty a Grid without boundaries in y - direction
 */


///@addtogroup grids
///@{
/** @brief Helper class for construction
 */
struct Topology1d
{
    unsigned n;
    unsigned N;
    bc b = dg::PER;
};


///@cond
template<class real_type>
struct RealCurvilinearProductGrid3d;
namespace detail
{
template<class host_vector>
dg::SparseTensor<host_vector> square( const dg::SparseTensor<host_vector >& jac, const host_vector& R, bool orthogonal)
{
    std::vector<host_vector> values( 5, R);
    {
        dg::blas1::scal( values[0], 0); //0
        dg::blas1::pointwiseDot( 1., jac.value(0,0), jac.value(0,0), 1., jac.value(0,1), jac.value(0,1), 0., values[1]); //xx
        dg::blas1::pointwiseDot( 1., jac.value(1,0), jac.value(1,0), 1., jac.value(1,1), jac.value(1,1), 0., values[2]); //yy
        dg::blas1::pointwiseDot( values[3], values[3], values[3]);
        dg::blas1::pointwiseDivide( 1., values[3], values[3]); //pp == 1/R^2

        dg::blas1::pointwiseDot( 1., jac.value(0,0), jac.value(1,0), 1., jac.value(0,1), jac.value(1,1), 0., values[4]); //xy
    }
    SparseTensor<host_vector> metric(values[0]); //unit tensor
    metric.values().pop_back(); //remove the one
    metric.idx(0,0) = 1; metric.values().push_back( values[1]);
    metric.idx(1,1) = 2; metric.values().push_back( values[2]);
    metric.idx(2,2) = 3; metric.values().push_back( values[3]);
    if( !orthogonal)
    {
        metric.idx(1,0) = metric.idx(0,1) = 4;
        metric.values().push_back( values[4]);
    }
    return metric;
}
}//namespace detail
///@endcond

//when we make a 3d grid with eta and phi swapped the metric structure and the transformation changes
//In practise it can only be orthogonal due to the projection tensor in the elliptic operator

/**
 * @brief A two-dimensional grid based on curvilinear coordinates
 */
template<class real_type>
struct RealCurvilinearGrid2d : public dg::aRealGeometry2d<real_type>
{
    RealCurvilinearGrid2d() = default;
    ///@copydoc hide_grid_parameters2d
    RealCurvilinearGrid2d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
        RealCurvilinearGrid2d( generator, {n,Nx,bcx}, {n,Ny,bcy}){}
    ///@copydoc hide_grid_product2d
    RealCurvilinearGrid2d( const aRealGenerator2d<real_type>& generator,
            Topology1d tx, Topology1d ty) :
        dg::aRealGeometry2d<real_type>( {0.,0.},
                {generator.width(), generator.height()},
                {tx.n,ty.n},{tx.N,ty.N}, { tx.b, ty.b}),
        m_handle(generator)
    {
        construct( tx.n, tx.N, ty.n, ty.N);
    }

    /**
     * @brief Explicitly convert 3d product grid to the perpendicular grid
     * @param g 3d product grid
     */
    explicit RealCurvilinearGrid2d( RealCurvilinearProductGrid3d<real_type> g);

    ///read access to the generator
    const aRealGenerator2d<real_type>& generator() const{return *m_handle;}
    virtual RealCurvilinearGrid2d* clone()const override final{return new RealCurvilinearGrid2d(*this);}
    private:
    virtual void do_set(std::array<unsigned,2> new_n, std::array<unsigned,2> new_N) override final
    {
        dg::aRealTopology2d<real_type>::do_set( new_n, new_N);
        construct( new_n[0], new_N[0], new_n[1], new_N[1]);
    }
    virtual void do_set(std::array<dg::bc,2> new_bc) override final
    {
        dg::aRealTopology2d<real_type>::do_set( new_bc);
    }
    virtual void do_set_pq(std::array<real_type,2> new_x0, std::array<real_type,2> new_x1) override final
    {
        throw dg::Error(dg::Message(_ping_)<<"This grid cannot change boundaries\n");
    }
    void construct( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny);
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian( ) const override final{
        return m_jac;
    }
    virtual SparseTensor<thrust::host_vector<real_type>> do_compute_metric( ) const override final{
        return m_metric;
    }
    virtual std::vector<thrust::host_vector<real_type>> do_compute_map()const override final{return m_map;}
    dg::SparseTensor<thrust::host_vector<real_type>> m_jac, m_metric;
    std::vector<thrust::host_vector<real_type>> m_map;
    dg::ClonePtr<aRealGenerator2d<real_type>> m_handle;
};


/**
 * @brief A 2x1 curvilinear product space grid

 * The base coordinate system is the cylindrical coordinate system R,Z,phi
 */
template<class real_type>
struct RealCurvilinearProductGrid3d : public dg::aRealProductGeometry3d<real_type>
{
    using perpendicular_grid = RealCurvilinearGrid2d<real_type>;
    RealCurvilinearProductGrid3d() = default;

    ///@copydoc hide_grid_parameters3d
    RealCurvilinearProductGrid3d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
        RealCurvilinearProductGrid3d( generator, {n,Nx,bcx}, {n,Ny,bcy}, {0., 2.*M_PI, 1,Nz,bcz}){}

    ///@copydoc hide_grid_product3d
    RealCurvilinearProductGrid3d( const aRealGenerator2d<real_type>& generator,
            Topology1d tx, Topology1d ty, RealGrid1d<real_type> gz):
        dg::aRealProductGeometry3d<real_type>(
                {0.,0., gz.x0()},{ generator.width(),
                generator.height(),gz.x1()}, {tx.n,ty.n, gz.n()},
                {tx.N, ty.N, gz.N()},{ tx.b, ty.b, gz.bcx()}), m_handle(generator)
    {
        m_map.resize(3);
        constructPerp( this->nx(), this->Nx(), this->ny(), this->Ny());
        constructParallel(this->nz(), this->Nz());
    }


    ///@copydoc RealCurvilinearGrid2d::generator()const
    const aRealGenerator2d<real_type> & generator() const{return *m_handle;}
    virtual RealCurvilinearProductGrid3d* clone()const override final{return new RealCurvilinearProductGrid3d(*this);}
    private:
    virtual RealCurvilinearGrid2d<real_type>* do_perp_grid() const override final;
    virtual void do_set( std::array<unsigned,3> new_n, std::array<unsigned,3> new_N) override final
    {
        auto old_n = this->get_n(), old_N = this->get_N();
        dg::aRealTopology3d<real_type>::do_set( new_n, new_N);
        if( !( new_n[0] == old_n[0] && new_N[0] == old_N[0] &&
               new_n[1] == old_n[1] && new_N[1] == old_N[1] ) )
            constructPerp( new_n[0], new_N[0], new_n[1],new_N[1]);
        constructParallel(new_n[2],new_N[2]);
    }
    virtual void do_set(std::array<dg::bc,3> new_bc) override final
    {
        dg::aRealTopology3d<real_type>::do_set( new_bc);
    }
    virtual void do_set_pq(std::array<real_type,3> new_x0, std::array<real_type,3> new_x1) override final
    {
        throw dg::Error(dg::Message(_ping_)<<"This grid cannot change boundaries\n");
    }
    //construct phi and lift rest to 3d
    void constructParallel(unsigned nz,unsigned Nz)
    {
        m_map[2]=dg::evaluate(dg::cooZ3d, *this);
        unsigned size = this->size();
        unsigned size2d = this->nx()*this->ny()*this->Nx()*this->Ny();
        //resize for 3d values
        for( unsigned r=0; r<6;r++)
            m_jac.values()[r].resize(size);
        m_map[0].resize(size);
        m_map[1].resize(size);
        //lift to 3D grid
        for( unsigned k=1; k<nz*Nz; k++)
            for( unsigned i=0; i<size2d; i++)
            {
                for(unsigned r=0; r<6; r++)
                    m_jac.values()[r][k*size2d+i] = m_jac.values()[r][(k-1)*size2d+i];
                m_map[0][k*size2d+i] = m_map[0][(k-1)*size2d+i];
                m_map[1][k*size2d+i] = m_map[1][(k-1)*size2d+i];
            }
    }
    //construct 2d plane
    void constructPerp( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny)
    {
        dg::Grid1d gX1d( this->x0(), this->x1(), nx, Nx);
        dg::Grid1d gY1d( this->y0(), this->y1(), ny, Ny);
        thrust::host_vector<real_type> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<real_type> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        m_jac = SparseTensor< thrust::host_vector<real_type>>( x_vec);//unit tensor
        m_jac.values().resize( 6);
        m_handle->generate( x_vec, y_vec, m_map[0], m_map[1], m_jac.values()[2], m_jac.values()[3], m_jac.values()[4], m_jac.values()[5]);
        m_jac.idx(0,0) = 2, m_jac.idx(0,1) = 3, m_jac.idx(1,0)=4, m_jac.idx(1,1) = 5;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian( ) const override final{
        return m_jac;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric( ) const override final
    {
        return detail::square( m_jac, m_map[0], m_handle->isOrthogonal());
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{return m_map;}
    std::vector<thrust::host_vector<real_type> > m_map;
    SparseTensor<thrust::host_vector<real_type> > m_jac;
    dg::ClonePtr<aRealGenerator2d<real_type>> m_handle;
};

using CurvilinearGrid2d         = dg::geo::RealCurvilinearGrid2d<double>;
using CurvilinearProductGrid3d  = dg::geo::RealCurvilinearProductGrid3d<double>;
#ifndef MPI_VERSION
namespace x{
using CurvilinearGrid2d         = CurvilinearGrid2d        ;
using CurvilinearProductGrid3d  = CurvilinearProductGrid3d ;
}
#endif

///@}
///@cond
template<class real_type>
RealCurvilinearGrid2d<real_type>::RealCurvilinearGrid2d( RealCurvilinearProductGrid3d<real_type> g):
    dg::aRealGeometry2d<real_type>( std::array{g.gx(), g.gy()} ), m_handle(g.generator())
{
    g.set( {this->nx(), this->ny(), 1}, {this->Nx(), this->Ny(),1}); //shouldn't trigger 2d grid generator
    m_map=g.map();
    m_jac=g.jacobian();
    m_metric=g.metric();
    // we rely on the fact that the 3d grid uses square to compute its metric
    // so the (2,2) entry is value 3 that we need to set to 1 (for the
    // create::volume function to work properly)
    dg::blas1::copy( 1., m_metric.values()[3]);
    m_map.pop_back();
}
template<class real_type>
void RealCurvilinearGrid2d<real_type>::construct( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny)
{
    RealCurvilinearProductGrid3d<real_type> g( *m_handle, {nx,Nx,this->bcx()}, {ny,Ny,this->bcy()}, {0., 2.*M_PI, 1,1});
    *this = RealCurvilinearGrid2d<real_type>(g);
}
template<class real_type>
typename RealCurvilinearProductGrid3d<real_type>::perpendicular_grid* RealCurvilinearProductGrid3d<real_type>::do_perp_grid() const { return new typename RealCurvilinearProductGrid3d<real_type>::perpendicular_grid(*this);}
///@endcond

}//namespace geo
}//namespace dg
