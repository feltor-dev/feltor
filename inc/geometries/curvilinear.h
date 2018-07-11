#pragma once

#include "dg/backend/memory.h"
#include "dg/blas1.h"
#include "dg/geometry/base_geometry.h"
#include "generator.h"

namespace dg
{
namespace geo
{

/*!@class hide_grid_parameters3d
 * @brief Construct a 3D grid
 *
 * the coordinates of the computational space are called x,y,z
 * @param generator generates the perpendicular grid
 * @param n number of %Gaussian nodes in x and y
 *  (1<=n<=20, note that the library is optimized for n=3 )
 * @attention # of polynomial coefficients in z direction is always 1
 * @param Nx number of cells in x
 * @param Ny number of cells in y
 * @param Nz  number of cells z
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 * @param bcz boundary condition in z
 */

/*!@class hide_grid_parameters2d
 * @brief Construct a 2D grid
 *
 * the coordinates of the computational space are called x,y,z
 * @param generator generates the grid
 * @param n number of %Gaussian nodes in x and y
 *  (1<=n<=20, note that the library is optimized for n=3 )
 * @param Nx number of cells in x
 * @param Ny number of cells in y
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 */


///@addtogroup grids
///@{

///@cond
template<class real_type>
struct CurvilinearProductGrid3d;
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
    SparseTensor<thrust::host_vector<real_type>> metric(values[0]); //unit tensor
    metric.values().pop_back(); //remove the one
    metric.idx(0,0) = 1; metric.values().push_back( values[1]);
    metric.idx(1,1) = 2; metric.values().push_back( values[2]);
    metric.idx(2,2) = 3; metric.values().push_back( values[3]);
    if( !orthogonal)
    {
        metric.idx(1,0) = metric.idx(0,1) = 4;
        metric.values().push_back( values[4]);
    }
}
}//namespace detail
///@endcond

//when we make a 3d grid with eta and phi swapped the metric structure and the transformation changes
//In practise it can only be orthogonal due to the projection tensor in the elliptic operator

/**
 * @brief A two-dimensional grid based on curvilinear coordinates
 *
 * @snippet flux_t.cu doxygen
 */
template<class real_type>
struct RealCurvilinearGrid2d : public dg::aRealGeometry2d<real_type>
{
    ///@copydoc hide_grid_parameters2d
    RealCurvilinearGrid2d( const aGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
        dg::aRealGeometry2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, dg::PER), handle_(generator)
    {
        construct( n,Nx,Ny);
    }

    /**
     * @brief Explicitly convert 3d product grid to the perpendicular grid
     * @param g 3d product grid
     */
    explicit RealCurvilinearGrid2d( RealCurvilinearProductGrid3d<real_type> g);

    ///read access to the generator
    const aGenerator2d<real_type>& generator() const{return handle_.get();}
    virtual RealCurvilinearGrid2d* clone()const override final{return new RealCurvilinearGrid2d(*this);}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny) override final
    {
        dg::aRealTopology2d<real_type>::do_set( new_n, new_Nx, new_Ny);
        construct( new_n, new_Nx, new_Ny);
    }
    void construct( unsigned n, unsigned Nx, unsigned Ny);
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian( ) const override final{
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric( ) const override final{
        return metric_;
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{return map_;}
    dg::SparseTensor<thrust::host_vector<real_type> > jac_, metric_;
    std::vector<thrust::host_vector<real_type> > map_;
    dg::ClonePtr<aGenerator2d<real_type>> handle_;
};


/**
 * @brief A 2x1 curvilinear product space grid

 * The base coordinate system is the cylindrical coordinate system R,Z,phi
 * @snippet hector_t.cu doxygen
 */
template<class real_type>
struct RealCurvilinearProductGrid3d : public dg::aRealProductGeometry3d<real_type>
{
    typedef RealCurvilinearGrid2d<real_type> perpendicular_grid;

    ///@copydoc hide_grid_parameters3d
    RealCurvilinearProductGrid3d( const aGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
        dg::aRealProductGeometry3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz)
    {
        map_.resize(3);
        handle_ = generator;
        constructPerp( n, Nx, Ny);
        constructParallel(Nz);
    }


    ///@copydoc CurvilinearGrid2d::generator()const
    const aGenerator2d<real_type> & generator() const{return handle_.get();}
    virtual RealCurvilinearProductGrid3d* clone()const override final{return new RealCurvilinearProductGrid3d(*this);}
    private:
    virtual RealCurvilinearGrid2d* do_perp_grid() const override final;
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny,unsigned new_Nz) override final{
        dg::aRealTopology3d<real_type>::do_set( new_n, new_Nx, new_Ny,new_Nz);
        if( !( new_n == n() && new_Nx == Nx() && new_Ny == Ny() ) )
            constructPerp( new_n, new_Nx, new_Ny);
        constructParallel(new_Nz);
    }
    //construct phi and lift rest to 3d
    void constructParallel(unsigned Nz)
    {
        map_[2]=dg::evaluate(dg::cooZ3d, *this);
        unsigned size = this->size();
        unsigned size2d = this->n()*this->n()*this->Nx()*this->Ny();
        //resize for 3d values
        for( unsigned r=0; r<6;r++)
            jac_.values()[r].resize(size);
        map_[0].resize(size);
        map_[1].resize(size);
        //lift to 3D grid
        for( unsigned k=1; k<Nz; k++)
            for( unsigned i=0; i<size2d; i++)
            {
                for(unsigned r=0; r<6; r++)
                    jac_.values()[r][k*size2d+i] = jac_.values()[r][(k-1)*size2d+i];
                map_[0][k*size2d+i] = map_[0][(k-1)*size2d+i];
                map_[1][k*size2d+i] = map_[1][(k-1)*size2d+i];
            }
    }
    //construct 2d plane
    void constructPerp( unsigned n, unsigned Nx, unsigned Ny)
    {
        dg::Grid1d gX1d( x0(), x1(), n, Nx);
        dg::Grid1d gY1d( y0(), y1(), n, Ny);
        thrust::host_vector<real_type> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<real_type> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        jac_ = SparseTensor< thrust::host_vector<real_type>>( x_vec);//unit tensor
        jac_.values().resize( 6);
        handle_.get().generate( x_vec, y_vec, map_[0], map_[1], jac_.values()[2], jac_.values()[3], jac_.values()[4], jac_.values()[5]);
        jac_.idx(0,0) = 2, jac_.idx(0,1) = 3, jac_.idx(1,0)=4, jac_.idx(1,1) = 5;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian( ) const override final{
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric( ) const override final
    {
        return detail::square( jac_, map_[0], handle_.get().isOrthogonal());
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{return map_;}
    std::vector<thrust::host_vector<real_type> > map_;
    SparseTensor<thrust::host_vector<real_type> > jac_;
    dg::ClonePtr<aGenerator2d<real_type>> handle_;
};

using CurvilinearGrid2d         = dg::RealCurvilinearGrid2d<double>;
using CurvilinearProductGrid3d  = dg::RealCurvilinearProductGrid3d<double>;

///@}
///@cond
template<class real_type>
RealCurvilinearGrid2d<real_type>::RealCurvilinearGrid2d( RealCurvilinearProductGrid3d<real_type> g):
    dg::aRealGeometry2d<real_type>( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy() ), handle_(g.generator())
{
    g.set( n(), Nx(), Ny(), 1); //shouldn't trigger 2d grid generator
    map_=g.map();
    jac_=g.jacobian().perp();
    metric_=g.metric().perp();
    map_.pop_back();
}
template<class real_type>
void CurvilinearGrid2d<real_type>::construct( unsigned n, unsigned Nx, unsigned Ny)
{
    RealCurvilinearProductGrid3d<real_type> g( handle_.get(), n,Nx,Ny,1,bcx());
    *this = RealCurvilinearGrid2d<real_type>(g);
}
template<class real_type>
RealCurvilinearProductGrid3d<real_type>::perpendicular_grid* RealCurvilinearProductGrid3d::do_perp_grid() const override final { return new RealCurvilinearProductGrid3d<real_type>::perpendicular_grid(*this);}
///@endcond

}//namespace geo
}//namespace dg
