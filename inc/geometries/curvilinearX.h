#pragma once

#include "dg/backend/gridX.h"
#include "dg/backend/evaluationX.cuh"
#include "dg/backend/functions.h"
#include "dg/blas1.h"
#include "dg/geometry/base_geometryX.h"
#include "generatorX.h"
#include "curvilinear.h"

namespace dg
{
namespace geo
{
///@addtogroup grids
///@{

/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 * 
 * The base coordinate system is the cylindrical coordinate system R,Z,phi
 */
struct CurvilinearProductGridX3d : public dg::aGeometryX3d
{
    /*!@brief Constructor
    
     * the coordinates of the computational space are called x,y,z
     * @param generator must generate a grid
     * @param n number of %Gaussian nodes in x and y
     * @param fx
     * @param fy
     * @param Nx number of cells in x
     * @param Ny number of cells in y 
     * @param Nz  number of cells z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     */
    CurvilinearProductGridX3d( const aGeneratorX2d& generator, 
        double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
        dg::aGeometryX3d( generator.zeta0(fx), generator.zeta1(fx), generator.eta0(fy), generator.eta1(fy), 0., 2.*M_PI, fx,fy,n, Nx, Ny, Nz, bcx, bcy, bcz)
    { 
        map_.resize(3);
        handle_ = generator;
        constructPerp( n, Nx, Ny);
        constructParallel(Nz);
    }

    const aGeneratorX2d & generator() const{return handle_.get();}
    virtual CurvilinearProductGridX3d* clone()const{return new CurvilinearProductGridX3d(*this);}
    private:
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
        dg::GridX1d gY1d( y0(), y1(), fy(), n, Ny);
        thrust::host_vector<double> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        handle_.get().generate( x_vec, y_vec, gY1d.n()*gY1d.outer_N(), gY1d.n()*(gY1d.inner_N()+gY1d.outer_N()), map_[0], map_[1], jac_.value(0), jac_.value(1), jac_.value(2), jac_.value(3));
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
    dg::Handle<aGeneratorX2d> handle_;
};

/**
 * @brief A two-dimensional grid based on curvilinear coordinates
 */
struct CurvilinearGridX2d : public dg::aGeometryX2d
{
    /*!@brief Constructor
    
     * @param generator must generate an orthogonal grid (class takes ownership of the pointer)
     * @param fx
     * @param fy
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     * @param Ny number of cells in second coordinate
     * @param bcx boundary condition in first coordinate
     * @param bcy boundary condition in second coordinate
     */
    CurvilinearGridX2d( const aGeneratorX2d& generator, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
        dg::aGeometryX2d( generator.zeta0(fx), generator.zeta1(fx), generator.eta0(fy), generator.eta1(fy),fx,fy, n, Nx, Ny, bcx, bcy), handle_(generator)
    {
        construct(fx,fy, n,Nx,Ny);
    }

    const aGeneratorX2d& generator() const{return handle_.get();}
    virtual CurvilinearGridX2d* clone()const{return new CurvilinearGridX2d(*this);}
    private:
    void construct( double fx, double fy, unsigned n, unsigned Nx, unsigned Ny)
    {
        CurvilinearProductGridX3d g( handle_.get(),fx,fy,n,Nx,Ny,1,bcx());
        map_=g.map();
        jac_=g.jacobian().perp();
        metric_=g.metric().perp();
        map_.pop_back();
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
    dg::Handle<aGeneratorX2d> handle_;
};

///@}

} //namespace geo
} //namespace dg
