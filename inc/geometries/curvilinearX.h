#pragma once

#include "dg/algorithm.h"
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
template<class real_type>
struct RealCurvilinearProductGridX3d : public dg::aRealGeometryX3d<real_type>
{
    /*!@brief Constructor

     * the coordinates of the computational space are called x,y,z
     * @param generator must generate a grid
     * @param fx a rational number indicating partition of the x - direction
     * @param fy a rational number indicating partition of the y - direction
     * @param n number of %Gaussian nodes in x and y
     * @param Nx number of cells in x
     * @param Ny number of cells in y
     * @param Nz  number of cells z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     */
    RealCurvilinearProductGridX3d( const aRealGeneratorX2d<real_type>& generator,
        real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
        dg::aRealGeometryX3d<real_type>( generator.zeta0(fx), generator.zeta1(fx), generator.eta0(fy), generator.eta1(fy), 0., 2.*M_PI, fx,fy,n, Nx, Ny, Nz, bcx, bcy, bcz)
    {
        map_.resize(3);
        handle_ = generator;
        constructPerp( n, Nx, Ny);
        constructParallel(Nz);
    }

    const aRealGeneratorX2d<real_type> & generator() const{return *handle_;}
    virtual RealCurvilinearProductGridX3d* clone()const override final{return new RealCurvilinearProductGridX3d(*this);}
    private:
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
        dg::Grid1d gX1d( this->x0(), this->x1(), n, Nx);
        dg::GridX1d gY1d( this->y0(), this->y1(), this->fy(), n, Ny);
        thrust::host_vector<real_type> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<real_type> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        jac_ = SparseTensor< thrust::host_vector<real_type>>( x_vec);//unit tensor
        jac_.values().resize( 6);
        handle_->generate( x_vec, y_vec, gY1d.n()*gY1d.outer_N(), gY1d.n()*(gY1d.inner_N()+gY1d.outer_N()), map_[0], map_[1], jac_.values()[2], jac_.values()[3], jac_.values()[4], jac_.values()[5]);
        jac_.idx(0,0) = 2, jac_.idx(0,1) = 3, jac_.idx(1,0)=4, jac_.idx(1,1) = 5;
    }
    virtual SparseTensor<thrust::host_vector<real_type>> do_compute_jacobian( ) const override final{
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<real_type>> do_compute_metric( ) const override final
    {
        return detail::square( jac_, map_[0], handle_->isOrthogonal());
    }
    virtual std::vector<thrust::host_vector<real_type>> do_compute_map()const override final{return map_;}
    std::vector<thrust::host_vector<real_type>> map_;
    SparseTensor<thrust::host_vector<real_type>> jac_;
    dg::ClonePtr<aRealGeneratorX2d<real_type>> handle_;
};

/**
 * @brief A two-dimensional grid based on curvilinear coordinates
 */
template<class real_type>
struct RealCurvilinearGridX2d : public dg::aRealGeometryX2d<real_type>
{
    /*!@brief Constructor

     * @param generator must generate an orthogonal grid
     * @param fx a rational number indicating partition of the x - direction
     * @param fy a rational number indicating partition of the y - direction
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     * @param Ny number of cells in second coordinate
     * @param bcx boundary condition in first coordinate
     * @param bcy boundary condition in second coordinate
     */
    RealCurvilinearGridX2d( const aRealGeneratorX2d<real_type>& generator, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
        dg::aRealGeometryX2d<real_type>( generator.zeta0(fx), generator.zeta1(fx), generator.eta0(fy), generator.eta1(fy),fx,fy, n, Nx, Ny, bcx, bcy), handle_(generator)
    {
        construct(fx,fy, n,Nx,Ny);
    }

    const aRealGeneratorX2d<real_type>& generator() const{return *handle_;}
    virtual RealCurvilinearGridX2d* clone()const override final{return new RealCurvilinearGridX2d(*this);}
    private:
    void construct( real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny)
    {
        RealCurvilinearProductGridX3d<real_type> g( *handle_,fx,fy,n,Nx,Ny,1,this->bcx());
        map_=g.map();
        jac_=g.jacobian();
        metric_=g.metric();
        dg::blas1::copy( 1., metric_.values()[3]); //set pp to 1
        map_.pop_back();
    }
    virtual SparseTensor<thrust::host_vector<real_type>> do_compute_jacobian( ) const override final{
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<real_type>> do_compute_metric( ) const override final{
        return metric_;
    }
    virtual std::vector<thrust::host_vector<real_type>> do_compute_map()const override final{return map_;}
    dg::SparseTensor<thrust::host_vector<real_type>> jac_, metric_;
    std::vector<thrust::host_vector<real_type>> map_;
    dg::ClonePtr<aRealGeneratorX2d<real_type>> handle_;
};


using CurvilinearGridX2d        = dg::geo::RealCurvilinearGridX2d<double>;
using CurvilinearProductGridX3d = dg::geo::RealCurvilinearProductGridX3d<double>;

///@}

} //namespace geo
} //namespace dg
