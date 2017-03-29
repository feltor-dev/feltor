#pragma once

#include "dg/backend/gridX.h"
#include "dg/backend/evaluationX.cuh"
#include "dg/blas1.h"
#include "dg/geometry/geometry_traits.h"

namespace dg
{

///@cond
template< class container>
struct OrthogonalGridX2d; 
///@endcond

/**
 * @brief Orthogonal X-point grid in three dimensions
 * @ingroup grids
 */
template< class container>
struct OrthogonalGridX3d : public dg::GridX3d
{
    typedef dg::OrthogonalTag metric_category;
    typedef OrthogonalGridX2d<container> perpendicular_grid;

    /*!@brief Constructor
    
     * @tparam GeneratorX models aGeneratorX
     * @param generator isOrthogonal() must return true
     * @param psi_0
     * @param fx
     * @param fy
     * @param n 
     * @param Nx
     @param Ny
     @param Nz 
     @param bcx
     @param bcy
     */
    template< class GeneratorX>
    OrthogonalGridX3d( GeneratorX generator, double psi_0, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, dg::bc bcy):
        dg::GridX3d( 0,1, -2.*M_PI*fy/(1.-2.*fy), 2.*M_PI*(1.+fy/(1.-2.*fy)), 0., 2*M_PI, fx, fy, n, Nx, Ny, Nz, bcx, bcy, dg::PER),
        r_(this->size()), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_)
    {
        assert( generator.isOrthogonal());
        construct( generator, psi_0, fx, n, Nx, Ny);
    }

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    perpendicular_grid perp_grid() const { return OrthogonalGridX2d<container>(*this);}
    private:
    template<class GeneratorX>
    void construct( GeneratorX generator, double psi_0, double fx, unsigned n, unsigned Nx, unsigned Ny )
    {
        const double x_0 = generator.f0()*psi_0;
        const double x_1 = -fx/(1.-fx)*x_0;
        init_X_boundaries( x_0, x_1);
        dg::Grid1d gX1d( this->x0(), this->x1(), n, Nx, dg::DIR);
        thrust::host_vector<double> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        dg::GridX1d gY1d( -this->fy()*2.*M_PI/(1.-2.*this->fy()), 2*M_PI+this->fy()*2.*M_PI/(1.-2.*this->fy()), this->fy(), this->n(), this->Ny(), dg::DIR);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        thrust::host_vector<double> rvec, zvec, yrvec, yzvec, xrvec, xzvec;
        generator( x_vec, y_vec, gY1d.n()*gY1d.outer_N(), gY1d.n()*(gY1d.inner_N()+gY1d.outer_N()), rvec, zvec, xrvec, xzvec, yrvec, yzvec);

        unsigned Mx = this->n()*this->Nx(), My = this->n()*this->Ny();
        //now lift to 3D grid
        for( unsigned k=0; k<this->Nz(); k++)
            for( unsigned i=0; i<Mx*My; i++)
            {
                r_[k*Mx*My+i] = rvec[i];
                z_[k*Mx*My+i] = zvec[i];
                yr_[k*Mx*My+i] = yrvec[i];
                yz_[k*Mx*My+i] = yzvec[i];
                xr_[k*Mx*My+i] = xrvec[i];
                xz_[k*Mx*My+i] = xzvec[i];
            }
        construct_metric();
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric()
    {
        //std::cout << "CONSTRUCTING METRIC\n";
        thrust::host_vector<double> tempxx( r_), tempxy(r_), tempyy(r_), tempvol(r_);
        for( unsigned idx=0; idx<this->size(); idx++)
        {
            tempxx[idx] = (xr_[idx]*xr_[idx]+xz_[idx]*xz_[idx]);
            tempxy[idx] = (yr_[idx]*xr_[idx]+yz_[idx]*xz_[idx]);
            tempyy[idx] = (yr_[idx]*yr_[idx]+yz_[idx]*yz_[idx]);
            tempvol[idx] = r_[idx]/sqrt(tempxx[idx]*tempyy[idx]-tempxy[idx]*tempxy[idx]);
        }
        g_xx_=tempxx, g_xy_=tempxy, g_yy_=tempyy, vol_=tempvol;
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        vol2d_ = tempvol;
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
};

/**
 * @brief Orthogonal X-point grid in two dimensions
 * @ingroup grids
 */
template< class container>
struct OrthogonalGridX2d : public dg::GridX2d
{
    typedef dg::OrthogonalTag metric_category;
    /*!@brief Constructor
    
     * @tparam GeneratorX models aGeneratorX
     * @param generator isOrthogonal() must return true
     * @param psi_0 left flux surface
     * @param fx
     * @param fy
     * @param n 
     * @param Nx
     @param Ny
     @param bcx
     @param bcy
     */
    template<class Generator>
    OrthogonalGridX2d(Generator generator, double psi_0, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy):
        dg::GridX2d( 0, 1,-fy*2.*M_PI/(1.-2.*fy), 2*M_PI+fy*2.*M_PI/(1.-2.*fy), fx, fy, n, Nx, Ny, bcx, bcy)
    {
        OrthogonalGridX3d<container> g( generator, psi_0, fx,fy, n,Nx,Ny,1,bcx,bcy);
        init_X_boundaries( g.x0(),g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    OrthogonalGridX2d( const OrthogonalGridX3d<container>& g):
        dg::GridX2d( g.x0(), g.x1(), g.y0(), g.y1(), g.fx(), g.fy(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
    {
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        { r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.g_xy().begin(), g.g_xy().begin()+s, g_xy_.begin());
        thrust::copy( g.g_yy().begin(), g.g_yy().begin()+s, g_yy_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }
    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}

    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, vol2d_;
};

}//namespace dg

