#pragma once

#include "dg/backend/manage.h"
#include "dg/blas1.h"
#include "base_geometry.h"
#include "generator.h"

namespace dg
{
///@addtogroup basicgrids
///@{

///@cond
struct CurvilinearGrid2d; 
///@endcond

//when we make a 3d grid with eta and phi swapped the metric structure and the transformation changes 
//In practise it can only be orthogonal due to the projection matrix in the elliptic operator


/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 * 
 * The base coordinate system is the cylindrical coordinate system R,Z,phi
 */
struct CylindricalProductGrid3d : public dg::aGeometry3d
{
    typedef CurvilinearGrid2d perpendicular_grid;

    /*!@brief Constructor
    
     * the coordinates of the computational space are called x,y,z
     * @param generator must generate a grid
     * @param n number of %Gaussian nodes in x and y
     * @param Nx number of cells in x
     * @param Ny number of cells in y 
     * @param Nz  number of cells z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     */
    CylindricalProductGrid3d( const aGenerator2d& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
        dg::aGeometry3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz)
    { 
        map_.resize(3);
        handle_ = generator;
        constructPerp( n, Nx, Ny);
        constructParallel(Nz);
    }

    perpendicular_grid perp_grid() const;// { return perpendicular_grid(*this);}

    const aGenerator2d & generator() const{return handle_.get();}
    virtual CylindricalProductGrid3d* clone()const{return new CylindricalProductGrid3d(*this);}
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
        dg::Grid1d gX1d( 0., x0(), n, Nx);
        dg::Grid1d gY1d( 0., y0(), n, Ny);
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
        thrust::host_vector<double> tempxx( size()), tempxy(size()), tempyy(size()), temppp(size());
        for( unsigned i=0; i<size(); i++)
        {
            tempxx[i] = (jac_.value(0,0)[i]*jac_.value(0,0)[i]+jac_.value(0,1)[i]*jac_.value(0,1)[i]);
            tempxy[i] = (jac_.value(0,0)[i]*jac_.value(1,0)[i]+jac_.value(0,1)[i]*jac_.value(1,1)[i]);
            tempyy[i] = (jac_.value(1,0)[i]*jac_.value(1,0)[i]+jac_.value(1,1)[i]*jac_.value(1,1)[i]);
            temppp[i] = 1./map_[2][i]/map_[2][i]; //1/R^2
        }
        SparseTensor<thrust::host_vector<double> > metric;
        metric.idx(0,0) = 0; metric.value(0) = tempxx;
        metric.idx(1,1) = 1; metric.value(1) = tempyy;
        metric.idx(2,2) = 2; metric.value(2) = temppp;
        if( !handle_.get().isOrthogonal())
        {
            metric.idx(0,1) = metric.idx(1,0) = 3; 
            metric.value(3) = tempxy;
        }
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
    /*!@brief Constructor
    
     * @param generator must generate an orthogonal grid (class takes ownership of the pointer)
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     * @param Ny number of cells in second coordinate
     * @param bcx boundary condition in first coordinate
     * @param bcy boundary condition in second coordinate
     */
    CurvilinearGrid2d( const aGenerator2d& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
        dg::aGeometry2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, dg::PER), handle_(generator)
    {
        construct( n,Nx,Ny);
    }
    explicit CurvilinearGrid2d( CylindricalProductGrid3d g):
        dg::aGeometry2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy() ), handle_(g.generator())
    {
        g.set( n(), Nx(), Ny(), 1); //shouldn't trigger 2d grid generator
        map_=g.map();
        jac_=g.jacobian().perp();
        metric_=g.metric().perp();
        map_.pop_back();
    }

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
        CylindricalProductGrid3d g( handle_.get(), n,Nx,Ny,1,bcx());
        jac_=g.jacobian();
        map_=g.map();
        metric_=g.metric();
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
CylindricalProductGrid3d::perpendicular_grid CylindricalProductGrid3d::perp_grid() const { return CylindricalProductGrid3d::perpendicular_grid(*this);}
///@endcond

}//namespace dg
