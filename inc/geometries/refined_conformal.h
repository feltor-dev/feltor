#pragma once

#include "dg/geometry/refined_grid.h"
#include "conformal.h"

namespace dg
{

///@cond
template< class container>
struct ConformalRefinedGrid2d; 
///@endcond

///@addtogroup grids
///@{

/**
 * @brief A conformal refined grid
 */
template< class container>
struct ConformalRefinedGrid3d : public dg::RefinedGrid3d
{
    typedef dg::ConformalCylindricalTag metric_category;
    typedef ConformalRefinedGrid2d<container> perpendicular_grid;

    template<class Generator>
    ConformalRefinedGrid3d( unsigned multiple_x, unsigned multiple_y, const Generator& generator, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx): 
        dg::RefinedGrid3d( multiple_x, multiple_y, 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, n_old, Nx, Ny, Nz, bcx, dg::PER, dg::PER),
        g_assoc_( generator, n_old, Nx, Ny, Nz, bcx)
    { 
        assert( generator.isConformal());
        construct( generator);
    }


    perpendicular_grid perp_grid() const { return perpendicular_grid(*this);}
    const dg::ConformalGrid3d<container>& associated() const{ return g_assoc_;}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_xx_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    private:
    template< class Generator>
    void construct( Generator generator)
    {
        init_X_boundaries( 0., generator.width());
        unsigned sizeY = this->n()*this->Ny();
        unsigned sizeX = this->n()*this->Nx();
        thrust::host_vector<double> y_vec(sizeY), x_vec(sizeX);
        for(unsigned i=0; i<sizeY; i++) y_vec[i] = this->abscissasY()[i*sizeX];
        for(unsigned i=0; i<sizeX; i++) x_vec[i] = this->abscissasX()[i];
        generator( x_vec, y_vec, r_, z_, xr_, xz_, yr_, yz_);
        lift3d( ); //lift to 3D grid
        construct_metric();
    }
    void lift3d( )
    {
        //lift to 3D grid
        unsigned size = this->size();
        r_.resize( size), z_.resize(size), xr_.resize(size), yr_.resize( size), xz_.resize( size), yz_.resize(size);
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        thrust::host_vector<double> wx = this->weightsX();
        thrust::host_vector<double> wy = this->weightsY();
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                r_[k*Nx*Ny+i] = r_[(k-1)*Nx*Ny+i];
                z_[k*Nx*Ny+i] = z_[(k-1)*Nx*Ny+i];
                yr_[k*Nx*Ny+i] = yr_[(k-1)*Nx*Ny+i]*wy[k*Nx*Ny+i];
                yz_[k*Nx*Ny+i] = yz_[(k-1)*Nx*Ny+i]*wy[k*Nx*Ny+i];
                xr_[k*Nx*Ny+i] = xr_[(k-1)*Nx*Ny+i]*wx[k*Nx*Ny+i];
                xz_[k*Nx*Ny+i] = xz_[(k-1)*Nx*Ny+i]*wx[k*Nx*Ny+i];
            }
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric( )
    {
        thrust::host_vector<double> tempxx( r_), tempvol(r_);
        for( unsigned i = 0; i<this->size(); i++)
        {
            tempxx[i] = (xr_[i]*xr_[i]+xz_[i]*xz_[i]);
            tempvol[i] = r_[i]/ tempxx[i];
        }
        dg::blas1::transfer( tempxx, g_xx_);
        dg::blas1::transfer( tempvol, vol_);
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        dg::blas1::transfer( tempvol, vol2d_);
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_; 
    container g_xx_, g_pp_, vol_, vol2d_;
    dg::ConformalGrid3d<container> g_assoc_;

};

/**
 * @brief A conformal refined grid
 */
template< class container>
struct ConformalRefinedGrid2d : public dg::RefinedGrid2d
{
    typedef dg::ConformalCylindricalTag metric_category;
    template< class Generator>
    ConformalRefinedGrid2d( unsigned multiple_x, unsigned multiple_y, const Generator& generator, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx):
        dg::RefinedGrid2d( multiple_x, multiple_y, 0, 1., 0., 2*M_PI, n,n_old,Nx,Ny, bcx, dg::PER),
        g_assoc_( generator, n_old, Nx, Ny, bcx) 
    {
        dg::ConformalRefinedGrid3d<container> g( multiple_x, multiple_y, generator, n,n_old,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx();
        vol2d_=g.perpVol();
    }

    ConformalRefinedGrid2d( const ConformalRefinedGrid3d<container>& g):
        dg::RefinedGrid2d( g ), g_assoc_(g.associated())
    {
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s),
        g_xx_.resize( s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        {r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }

    const dg::ConformalGrid2d<container>& associated()const{return g_assoc_;}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_xx_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container g_xx_, vol2d_;
    dg::ConformalGrid2d<container> g_assoc_;
};

///@}
}//namespace dg
