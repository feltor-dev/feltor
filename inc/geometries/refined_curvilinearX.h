#pragma once

#include "dg/geometry/refined_gridX.h"
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
struct RealCurvilinearRefinedProductGridX3d : public dg::aRealGeometryX3d<real_type>
{
    /*!@brief Constructor

     * the coordinates of the computational space are called x,y,z
     * @param ref a X-point refinement
     * @param generator must generate a grid
     * @param fx a rational number indicating partition of the x - direction
     * @param fy a rational number indicating partition of the y - direction
     * @param n number of %Gaussian nodes in x and y
     * @param Nx number of cells in x
     * @param Ny number of cells in y
     * @param Nz  number of cells z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y (may not be periodic)
     * @param bcz boundary condition in z
     */
    RealCurvilinearRefinedProductGridX3d( const aRealRefinementX2d<real_type>& ref, const aRealGeneratorX2d<real_type>& generator,
        real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::NEU, bc bcz=dg::PER):
        dg::aRealGeometryX3d<real_type>( generator.zeta0(fx), generator.zeta1(fx), generator.eta0(fy), generator.eta1(fy), 0., 2.*M_PI, ref.fx_new(Nx,fx),ref.fy_new(Ny,fy),n, ref.nx_new(Nx,fx), ref.ny_new(Ny,fy), Nz, bcx, bcy, bcz), map_(3)
    {
        handle_ = generator;
        ref_=ref;
        constructPerp( fx,fy,n, Nx, Ny);
        constructParallel(Nz);
    }

    const aRealGeneratorX2d<real_type> & generator() const{return handle_.get();}
    virtual RealCurvilinearRefinedProductGridX3d* clone()const override final{return new RealCurvilinearRefinedProductGridX3d(*this);}
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
    void constructPerp( real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny)
    {
        std::vector<thrust::host_vector<real_type> > w(2),abs(2);
        GridX2d g( this->x0(),this->x1(),this->y0(),this->y1(),fx,fy,n,Nx,Ny,this->bcx(),this->bcy());
        ref_.get().generate(g,w[0],w[1],abs[0],abs[1]);
        thrust::host_vector<real_type> x_vec(this->n()*this->Nx()), y_vec(this->n()*this->Ny());
        for( unsigned i=0; i<x_vec.size(); i++)
        {
            x_vec[i] = abs[0][i];
            //std::cout << abs[0][i] <<"\n";
        }
        for( unsigned i=0; i<y_vec.size(); i++)
        {
            y_vec[i] = abs[1][i*x_vec.size()];
            //std::cout << abs[1][i*x_vec.size()] <<"\n";
        }
        jac_ = SparseTensor< thrust::host_vector<real_type>>( x_vec);//unit tensor
        jac_.values().resize( 6);
        handle_.get().generate( x_vec, y_vec, this->n()*this->outer_Ny(), this->n()*(this->inner_Ny()+this->outer_Ny()), map_[0], map_[1], jac_.values()[0], jac_.values()[1], jac_.values()[2], jac_.values()[3]);
        //multiply by weights
        dg::blas1::pointwiseDot( jac_.values()[0], w[0], jac_.values()[0]);
        dg::blas1::pointwiseDot( jac_.values()[1], w[0], jac_.values()[1]);
        dg::blas1::pointwiseDot( jac_.values()[2], w[1], jac_.values()[2]);
        dg::blas1::pointwiseDot( jac_.values()[3], w[1], jac_.values()[3]);
        jac_.idx(0,0) = 0, jac_.idx(0,1) = 1, jac_.idx(1,0)=2, jac_.idx(1,1) = 3;
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
    dg::ClonePtr<aRealGeneratorX2d<real_type>> handle_;
    dg::ClonePtr<aRefinementX2d> ref_;
};

/**
 * @brief A two-dimensional grid based on curvilinear coordinates
 */
template<class real_type>
struct RealCurvilinearRefinedGridX2d : public dg::aRealGeometryX2d<real_type>
{
    /*!@brief Constructor

     * @param ref a X-point refinement
     * @param generator must generate an orthogonal grid (class takes ownership of the pointer)
     * @param fx a rational number indicating partition of the x - direction
     * @param fy a rational number indicating partition of the y - direction
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     * @param Ny number of cells in second coordinate
     * @param bcx boundary condition in first coordinate
     * @param bcy boundary condition in second coordinate (may not be periodic)
     */
    RealCurvilinearRefinedGridX2d( const aRealRefinementX2d<real_type>& ref, const aRealGeneratorX2d<real_type>& generator, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::NEU):
        dg::aRealGeometryX2d<real_type>( generator.zeta0(fx), generator.zeta1(fx), generator.eta0(fy), generator.eta1(fy),ref.fx_new(Nx,fx),ref.fy_new(Ny,fy),n, ref.nx_new(Nx,fx), ref.ny_new(Ny,fy), bcx, bcy)
    {
        handle_ = generator;
        ref_=ref;
        construct( fx,fy,n,Nx,Ny);
    }

    const aRealGeneratorX2d<real_type>& generator() const{return handle_.get();}
    virtual RealCurvilinearRefinedGridX2d* clone()const override final{return new RealCurvilinearRefinedGridX2d(*this);}
    private:
    void construct(real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny)
    {
        RealCurvilinearRefinedProductGridX3d<real_type> g( ref_.get(), handle_.get(),fx,fy,n,Nx,Ny,1,this->bcx(), this->bcy());
        map_=g.map();
        jac_=g.jacobian();
        metric_=g.metric();
        dg::blas1::copy( 1., metric_.values()[3]); //set pp to 1
        map_.pop_back();
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian( ) const override final{
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric( ) const override final{
        return metric_;
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{return map_;}
    dg::SparseTensor<thrust::host_vector<real_type> > jac_, metric_;
    std::vector<thrust::host_vector<real_type> > map_;
    dg::ClonePtr<aRealGeneratorX2d<real_type>> handle_;
    dg::ClonePtr<aRealRefinementX2d<real_type>> ref_;
};
using CurvilinearRefinedGridX2d        = dg::geo::RealCurvilinearRefinedGridX2d<double>;
using CurvilinearRefinedProductGridX3d = dg::geo::RealCurvilinearRefinedProductGridX3d<double>;

///@}


} //namespace geo
} //namespace dg

