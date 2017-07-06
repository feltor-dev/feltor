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
struct OrthogonalGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on orthogonal coordinates
 * @tparam container models aContainer
 */
template< class container>
struct OrthogonalGrid3d : public dg::Grid3d
{
    typedef dg::OrthogonalTag metric_category;
    typedef OrthogonalGrid2d<container> perpendicular_grid;

    /*!@brief Constructor
    
     * @tparam Generator models aGenerator
     * @param generator must generate an orthogonal grid
     * @param n 
     * @param Nx
     @param Ny
     @param Nz 
     @param bcx
     */
    template< class Generator>
    OrthogonalGrid3d( const Generator& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx=dg::DIR):
        dg::Grid3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    { 
        assert( generator.isOrthogonal());
        construct( generator, n, Nx, Ny);
    }

    perpendicular_grid perp_grid() const { return OrthogonalGrid2d<container>(*this);}
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
    private:
    template< class Generator>
    void construct( Generator generator, unsigned n, unsigned Nx, unsigned Ny)
    {
        dg::Grid1d gY1d( 0, generator.height(), n, Ny, dg::PER);
        dg::Grid1d gX1d( 0., generator.width(), n, Nx);
        thrust::host_vector<double> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        generator( x_vec, y_vec, r_, z_, xr_, xz_, yr_, yz_);
        init_X_boundaries( 0., generator.width());
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
        thrust::host_vector<double> tempxx( r_), tempxy(r_), tempyy(r_), tempvol(r_);
        for( unsigned i = 0; i<this->size(); i++)
        {
            tempxx[i] = (xr_[i]*xr_[i]+xz_[i]*xz_[i]);
            tempxy[i] = (yr_[i]*xr_[i]+yz_[i]*xz_[i]);
            tempyy[i] = (yr_[i]*yr_[i]+yz_[i]*yz_[i]);
            tempvol[i] = r_[i]/sqrt( tempxx[i]*tempyy[i] );
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
 * @brief A three-dimensional grid based on orthogonal coordinates
 @tparam container models aContainer
 */
template< class container>
struct OrthogonalGrid2d : public dg::Grid2d
{
    typedef dg::OrthogonalTag metric_category;
    /*!@brief Constructor
    
     * @tparam Generator models aGenerator
     * @param generator must generate an orthogonal grid
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     @param Ny number of cells in second coordinate
     @param bcx boundary condition in first coordinate
     */
    template< class Generator>
    OrthogonalGrid2d( const Generator& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR):
        dg::Grid2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, dg::PER)
    {
        OrthogonalGrid3d<container> g( generator, n,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();

    }
    OrthogonalGrid2d( const OrthogonalGrid3d<container>& g):
        dg::Grid2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
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

///@}
}//namespace dg
