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
struct CurvilinearGrid2d; 
namespace detail
{
void square( const dg::SparseTensor<thrust::host_vector<double> >& jac, const thrust::host_vector<double>& R, dg::SparseTensor<thrust::host_vector<double> >& metric, bool orthogonal)
{
    thrust::host_vector<double> tempxx( R.size()), tempxy(R.size()), tempyy(R.size()), temppp(R.size());
    for( unsigned i=0; i<R.size(); i++)
    {
        tempxx[i] = (jac.value(0,0)[i]*jac.value(0,0)[i]+jac.value(0,1)[i]*jac.value(0,1)[i]);
        tempxy[i] = (jac.value(0,0)[i]*jac.value(1,0)[i]+jac.value(0,1)[i]*jac.value(1,1)[i]);
        tempyy[i] = (jac.value(1,0)[i]*jac.value(1,0)[i]+jac.value(1,1)[i]*jac.value(1,1)[i]);
        temppp[i] = 1./R[i]/R[i]; //1/R^2
    }
    metric.idx(0,0) = 0; metric.value(0) = tempxx;
    metric.idx(1,1) = 1; metric.value(1) = tempyy;
    metric.idx(2,2) = 2; metric.value(2) = temppp;
    if( !orthogonal)
    {
        metric.idx(0,1) = metric.idx(1,0) = 3; 
        metric.value(3) = tempxy;
    }
}
}//namespace detail
///@endcond

//when we make a 3d grid with eta and phi swapped the metric structure and the transformation changes 
//In practise it can only be orthogonal due to the projection tensor in the elliptic operator


/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 * 
 * The base coordinate system is the cylindrical coordinate system R,Z,phi
 */
struct CurvilinearProductGrid3d : public dg::aGeometry3d
{
    typedef CurvilinearGrid2d perpendicular_grid;

    ///@copydoc hide_grid_parameters3d
    CurvilinearProductGrid3d( const aGenerator2d& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
        dg::aGeometry3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz)
    { 
        map_.resize(3);
        handle_ = generator;
        constructPerp( n, Nx, Ny);
        constructParallel(Nz);
    }

    /*!
     * @brief The grid made up by the first two dimensions
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid
     */
    perpendicular_grid perp_grid() const;// { return perpendicular_grid(*this);}

    ///@copydoc CurvilinearGrid2d::generator()const
    const aGenerator2d & generator() const{return handle_.get();}
    virtual CurvilinearProductGrid3d* clone()const{return new CurvilinearProductGrid3d(*this);}
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny,unsigned new_Nz){
        dg::aTopology3d::do_set( new_n, new_Nx, new_Ny,new_Nz);
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
        for( unsigned r=0; r<4;r++)
            jac_.value(r).resize(size);
        map_[0].resize(size);
        map_[1].resize(size);
        //lift to 3D grid
        for( unsigned k=1; k<Nz; k++)
            for( unsigned i=0; i<size2d; i++)
            {
                for(unsigned r=0; r<4; r++)
                    jac_.value(r)[k*size2d+i] = jac_.value(r)[(k-1)*size2d+i];
                map_[0][k*size2d+i] = map_[0][(k-1)*size2d+i];
                map_[1][k*size2d+i] = map_[1][(k-1)*size2d+i];
            }
    }
    //construct 2d plane
    void constructPerp( unsigned n, unsigned Nx, unsigned Ny)
    {
        dg::Grid1d gX1d( x0(), x1(), n, Nx);
        dg::Grid1d gY1d( y0(), y1(), n, Ny);
        thrust::host_vector<double> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        handle_.get().generate( x_vec, y_vec, map_[0], map_[1], jac_.value(0), jac_.value(1), jac_.value(2), jac_.value(3));
        jac_.idx(0,0) = 0, jac_.idx(0,1) = 1, jac_.idx(1,0)=2, jac_.idx(1,1) = 3;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian( ) const {
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric( ) const
    {
        SparseTensor<thrust::host_vector<double> > metric;
        detail::square( jac_, map_[0], metric, handle_.get().isOrthogonal());
        return metric;
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{return map_;}
    std::vector<thrust::host_vector<double> > map_;
    SparseTensor<thrust::host_vector<double> > jac_;
    dg::Handle<aGenerator2d> handle_;
};

/**
 * @brief A two-dimensional grid based on curvilinear coordinates
 */
struct CurvilinearGrid2d : public dg::aGeometry2d
{
    ///@copydoc hide_grid_parameters2d
    CurvilinearGrid2d( const aGenerator2d& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
        dg::aGeometry2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, dg::PER), handle_(generator)
    {
        construct( n,Nx,Ny);
    }

    /**
     * @brief Explicitly convert 3d product grid to the perpendicular grid
     * @param g 3d product grid
     */
    explicit CurvilinearGrid2d( CurvilinearProductGrid3d g):
        dg::aGeometry2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy() ), handle_(g.generator())
    {
        g.set( n(), Nx(), Ny(), 1); //shouldn't trigger 2d grid generator
        map_=g.map();
        jac_=g.jacobian().perp();
        metric_=g.metric().perp();
        map_.pop_back();
    }

    ///read access to the generator 
    const aGenerator2d& generator() const{return handle_.get();}
    virtual CurvilinearGrid2d* clone()const{return new CurvilinearGrid2d(*this);}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny)
    {
        dg::aTopology2d::do_set( new_n, new_Nx, new_Ny);
        construct( new_n, new_Nx, new_Ny);
    }
    void construct( unsigned n, unsigned Nx, unsigned Ny)
    {
        CurvilinearProductGrid3d g( handle_.get(), n,Nx,Ny,1,bcx());
        *this = CurvilinearGrid2d(g);
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian( ) const {
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric( ) const {
        return metric_;
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{return map_;}
    dg::SparseTensor<thrust::host_vector<double> > jac_, metric_;
    std::vector<thrust::host_vector<double> > map_;
    dg::Handle<aGenerator2d> handle_;
};

///@}
///@cond
CurvilinearProductGrid3d::perpendicular_grid CurvilinearProductGrid3d::perp_grid() const { return CurvilinearProductGrid3d::perpendicular_grid(*this);}
///@endcond

}//namespace geo
}//namespace dg
