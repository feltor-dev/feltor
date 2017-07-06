#pragma once

#include "dg/backend/grid.h"
#include "dg/blas1.h"
#include "dg/geometry/geometry_traits.h"

namespace dg
{
///@addtogroup grids
///@{

///@cond
template< class container>
struct ConformalGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on conformal coordintes 
 *
 * @tparam container Vector class that holds metric coefficients (models aContainer)
 */
template< class container>
struct ConformalGrid3d : public dg::Grid3d
{
    typedef dg::ConformalCylindricalTag metric_category; //!< metric tag
    typedef ConformalGrid2d<container> perpendicular_grid; //!< the two-dimensional grid type

    /**
     * @brief 
     *
     * @tparam Generator models aGenerator
     * @param generator must generate a conformal grid
     * @param n
     * @param Nx
     * @param Ny
     * @param Nz
     * @param bcx
     */
    template< class Generator>
    ConformalGrid3d( const Generator& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx=dg::DIR) :
        dg::Grid3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    {
        construct( generator, n,Nx, Ny, Nz, bcx);
    }
    perpendicular_grid perp_grid() const { return ConformalGrid2d<container>(*this);}
    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return gradU2_;}
    const container& g_yy()const{return gradU2_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    private:
    template< class Generator>
    void construct( const Generator& hector, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx) 
    {
        assert( hector.isConformal());
        dg::Grid1d gu( 0., hector.width(), n, Nx);
        dg::Grid1d gv( 0., hector.height(), n, Ny);
        const thrust::host_vector<double> u1d = dg::evaluate( dg::cooX1d, gu);
        const thrust::host_vector<double> v1d = dg::evaluate( dg::cooX1d, gv);
        hector( u1d, v1d, r_, z_, xr_, xz_, yr_, yz_);
        init_X_boundaries( 0., hector.width());
        lift3d( ); //lift to 3D grid
        construct_metric();
    }
    void lift3d( )
    {
        //lift to 3D grid
        unsigned size = this->size();
        r_.resize( size), z_.resize(size), xr_.resize(size), yr_.resize( size), xz_.resize( size), yz_.resize(size);
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                r_[k*Nx*Ny+i] = r_[(k-1)*Nx*Ny+i];
                z_[k*Nx*Ny+i] = z_[(k-1)*Nx*Ny+i];
                xr_[k*Nx*Ny+i] = xr_[(k-1)*Nx*Ny+i];
                xz_[k*Nx*Ny+i] = xz_[(k-1)*Nx*Ny+i];
                yr_[k*Nx*Ny+i] = yr_[(k-1)*Nx*Ny+i];
                yz_[k*Nx*Ny+i] = yz_[(k-1)*Nx*Ny+i];
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
        dg::blas1::transfer( tempxx, gradU2_);
        dg::blas1::transfer( tempvol, vol_);
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        dg::blas1::transfer( tempvol, vol2d_);
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_; //3d vector
    container gradU2_, g_pp_, vol_, vol2d_;

};

/**
 * @brief A two-dimensional grid based on the conformal coordinates  (models aContainer)
 */
template< class container>
struct ConformalGrid2d : public dg::Grid2d
{
    typedef dg::ConformalCylindricalTag metric_category;
    /**
     * @brief 
     *
     * @tparam Generator models aGenerator
     * @param generator must generate a conformal grid
     * @param n
     * @param Nx
     * @param Ny
     * @param bcx
     */
    template< class Generator>
    ConformalGrid2d( const Generator& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR):
        dg::Grid2d( 0, generator.width(), 0., generator.height(), n,Nx,Ny, bcx, dg::PER)
    {
        ConformalGrid3d<container> g( generator, n,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        gradU2_=g.g_xx();
        vol2d_=g.perpVol();
    }
    ConformalGrid2d( const ConformalGrid3d<container>& g):
        dg::Grid2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
    {
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        gradU2_.resize( s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        { r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, gradU2_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }


    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return gradU2_;}
    const container& g_yy()const{return gradU2_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container gradU2_, vol2d_;
};

///@}
}//namespace
