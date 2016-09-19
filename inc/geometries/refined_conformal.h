#pragma once

#include "dg/geometry/refined_grid.h"
#include "conformal.h"



namespace dg
{
namespace refined
{
namespace conformal
{

template< class container>
struct RingGrid2d; 

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid3d : public dg::refined::Grid3d
{
    typedef dg::ConformalCylindricalTag metric_category;
    typedef RingGrid2d<container> perpendicular_grid;

    /**
     * @brief Construct 
     *
     * @param gp The geometric parameters define the magnetic field
     * @param psi_0 lower boundary for psi
     * @param psi_1 upper boundary for psi
     * @param n The dG number of polynomials
     * @param Nx The number of points in x-direction
     * @param Ny The number of points in y-direction
     * @param Nz The number of points in z-direction
     * @param bcx The boundary condition in x (y,z are periodic)
     */
    RingGrid3d( unsigned multiple_x, unsigned multiple_y, solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx): 
        dg::refined::Grid3d( multiple_x, multiple_y, 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, n_old, Nx, Ny, Nz, bcx, dg::PER, dg::PER),
        g_assoc_( gp, psi_0, psi_1, n_old, Nx, Ny, Nz, bcx)
    { 
        solovev::Psip psip(gp); 
        solovev::PsipR psipR(gp); solovev::PsipZ psipZ(gp);
        solovev::LaplacePsip lapPsip(gp); 
        construct( psip, psipR, psipZ, lapPsip, psi_0, psi_1, gp.R_0, 0);

    }
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    RingGrid3d( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, 
            double psi_0, double psi_1, double x0, double y0, unsigned multiple_x, unsigned multiple_y, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx):
        dg::refined::Grid3d( multiple_x, multiple_y, 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, n_old, Nx, Ny, Nz, bcx, dg::PER, dg::PER),
        g_assoc_( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0, n_old, Nx, Ny, Nz, bcx)
    { 
        construct( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0);
    }


    perpendicular_grid perp_grid() const { return dg::refined::conformal::RingGrid2d<container>(*this);}
    const dg::conformal::RingGrid3d<container>& associated() const{ return g_assoc_;}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    private:
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    void construct( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi,
            double psi_0, double psi_1, 
            double x0, double y0)
    {
        unsigned sizeY = this->n()*this->Ny();
        unsigned sizeX = this->n()*this->Nx();
        thrust::host_vector<double> y_vec(sizeY);
        for(unsigned i=0; i<sizeY; i++) y_vec[i] = this->abscissasY()[i*sizeX];
        dg::Hector generator( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0);
        double x_1 = generator.lu();
        init_X_boundaries( 0., x_1);
        thrust::host_vector<double> x_vec(sizeX); 
        for(unsigned i=0; i<sizeX; i++) x_vec[i] = this->abscissasX()[i];
        generator( psi, psiX, psiY, laplacePsi, x_vec, y_vec, r_, z_, xr_, xz_, yr_, yz_);
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
    dg::conformal::RingGrid3d<container> g_assoc_;

};

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid2d : public dg::refined::Grid2d
{
    typedef dg::ConformalCylindricalTag metric_category;
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    RingGrid2d( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, 
            double psi_0, double psi_1, double x0, double y0, 
            unsigned multiple_x, unsigned multiple_y, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx):
        dg::refined::Grid2d( multiple_x, multiple_y, 0, 1., 0., 2*M_PI, n,n_old,Nx,Ny, bcx, dg::PER),
        g_assoc_( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0, n_old, Nx, Ny, bcx) 
    {
        dg::refined::conformal::RingGrid3d<container> g( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0, multiple_x, multiple_y, n,n_old,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx();
        vol2d_=g.perpVol();
    }

    RingGrid2d( unsigned multiple_x, unsigned multiple_y, const solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx): 
        dg::refined::Grid2d( multiple_x, multiple_y, 0, 1., 0., 2*M_PI, n,n_old,Nx,Ny, bcx, dg::PER),
        g_assoc_( gp, psi_0, psi_1, n_old, Nx, Ny, bcx) 
    {
        dg::refined::conformal::RingGrid3d<container> g( multiple_x, multiple_y, gp, psi_0, psi_1, n,n_old,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx();
        vol2d_=g.perpVol();
    }
    RingGrid2d( const RingGrid3d<container>& g):
        dg::refined::Grid2d( g ), g_assoc_(g.associated())
    {
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s),
        g_xx_.resize( s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        {r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }

    const dg::conformal::RingGrid2d<container>& associated()const{return g_assoc_;}

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
    dg::conformal::RingGrid2d<container> g_assoc_;
};

/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
}//namespace conformal
}//namespace refined
}//namespace dg
