#pragma once

#include "dg/geometry/refined_grid.h"
#include "orthogonal.h"



namespace dg
{
namespace refined
{
namespace orthogonal
{

template< class container>
struct RingGrid2d; 

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid3d : public dg::refined::Grid3d
{
    typedef dg::OrthogonalCylindricalTag metric_category;
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
    RingGrid3d( unsigned multiple_x, unsigned multiple_y, solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, int firstline = 0): 
        dg::refined::Grid3d( multiple_x, multiple_y, 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, n_old, Nx, Ny, Nz, bcx, dg::PER, dg::PER),
        g_assoc_( gp, psi_0, psi_1, n_old, Nx, Ny, Nz, bcx, firstline)
    { 
        solovev::Psip psip(gp); 
        solovev::PsipR psipR(gp); solovev::PsipZ psipZ(gp);
        solovev::LaplacePsip lapPsip(gp); 
        construct( psip, psipR, psipZ, lapPsip, psi_0, psi_1, gp.R_0, 0, multiple_x, multiple_y, n, n_old, Nx, Ny, firstline);

    }
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    RingGrid3d( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, 
            double psi_0, double psi_1, double x0, double y0, unsigned multiple_x, unsigned multiple_y, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, int firstline = 0):
        dg::refined::Grid3d( multiple_x, multiple_y, 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, n_old, Nx, Ny, Nz, bcx, dg::PER, dg::PER),
        g_assoc_( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0, n_old, Nx, Ny, Nz, bcx, firstline)
    { 
        construct( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0, multiple_x, multiple_y, n, n_old, Nx, Ny, firstline);
    }


    perpendicular_grid perp_grid() const { return orthogonal::refined::RingGrid2d<container>(*this);}
    const orthogonal::RingGrid3d<container>& associated() const{ return g_assoc_;}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const thrust::host_vector<double>& lapx()const{return lapx_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    private:
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    void construct( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi,
            double psi_0, double psi_1, 
            double x0, double y0, unsigned multiple_x, unsigned multipl_y, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, int firstline)
    {
        assert( psi_1 != psi_0);

        //compute innermost flux surface
        orthogonal::detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi, psiX, psiY, x0, y0, firstline);
        unsigned sizeY = this->n()*this->Ny();
        unsigned sizeX = this->n()*this->Nx();
        thrust::host_vector<double> y_vec(sizeY);
        for(unsigned i=0; i<sizeY; i++) y_vec[i] = this->abscissasY()[i*sizeX];
        thrust::host_vector<double> r_init(sizeY), z_init(sizeY);
        double R0, Z0, f0;
        thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
        f0 = fpsi.construct_f( psi_0, R0, Z0);
        if( psi_1 < psi_0) f0*=-1;
        detail::compute_rzy( psiX, psiY, psi_0, y_vec, r_init, z_init, R0, Z0, f0, firstline);

        //now construct grid in x
        double x_1 = fabs( f0*(psi_1-psi_0));
        init_X_boundaries( 0., x_1);

        thrust::host_vector<double> x_vec(sizeX); 
        for(unsigned i=0; i<sizeX; i++) x_vec[i] = this->abscissasX()[i];
        detail::Nemov<PsiX, PsiY, LaplacePsi> nemov(psiX, psiY, laplacePsi, f0, firstline);
        thrust::host_vector<double> h;
        detail::construct_rz(nemov, x_vec, r_init, z_init, r_, z_, h);
        r_.resize(size()), z_.resize(size());
        xr_.resize(size()), xz_.resize(size()), 
        yr_.resize(size()), yz_.resize(size());
        lapx_.resize(size());
        for( unsigned idx=0; idx<r_.size(); idx++)
        {
            double psipR = psiX(r_[idx], z_[idx]);
            double psipZ = psiY(r_[idx], z_[idx]);
            xr_[idx] = f0*psipR;
            xz_[idx] = f0*psipZ;
            yr_[idx] = -h[idx]*psipZ;
            yz_[idx] = +h[idx]*psipR;
            lapx_[idx] = f0*(laplacePsi( r_[idx], z_[idx]));
        }
        lift3d( ); //lift to 3D grid
        construct_metric();
    }
    void lift3d( )
    {
        //lift to 3D grid
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
                lapx_[k*Nx*Ny+i] = lapx_[(k-1)*Nx*Ny+i];
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
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_, lapx_; 
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
    orthogonal::RingGrid3d<container> g_assoc_;

};

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid2d : public dg::refined::Grid2d
{
    typedef dg::OrthogonalCylindricalTag metric_category;
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    RingGrid2d( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, 
            double psi_0, double psi_1, double x0, double y0, 
            unsigned multiple_x, unsigned multiple_y, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx, int firstline = 0):
        dg::refined::Grid2d( multiple_x, multiple_y, 0, 1., 0., 2*M_PI, n,n_old,Nx,Ny, bcx, dg::PER),
        g_assoc_( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0, n_old, Nx, Ny, bcx, firstline) 
    {
        orthogonal::refined::RingGrid3d<container> g( psi, psiX, psiY, laplacePsi, psi_0, psi_1, x0, y0, multiple_x, multiple_y, n,n_old,Nx,Ny,1,bcx, firstline);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }

    RingGrid2d( unsigned multiple_x, unsigned multiple_y, const solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx, int firstline =0): 
        dg::refined::Grid2d( multiple_x, multiple_y, 0, 1., 0., 2*M_PI, n,n_old,Nx,Ny, bcx, dg::PER),
        g_assoc_( gp, psi_0, psi_1, n_old, Nx, Ny, bcx, firstline) 
    {
        orthogonal::refined::RingGrid3d<container> g( multiple_x, multiple_y, gp, psi_0, psi_1, n,n_old,Nx,Ny,1,bcx, firstline);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    RingGrid2d( const RingGrid3d<container>& g):
        dg::refined::Grid2d( g ), g_assoc_(g.associated())
    {
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s), lapx_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        {r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i], lapx_[i] = g.lapx()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.g_xy().begin(), g.g_xy().begin()+s, g_xy_.begin());
        thrust::copy( g.g_yy().begin(), g.g_yy().begin()+s, g_yy_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }

    const orthogonal::RingGrid2d<container>& associated()const{return g_assoc_;}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const thrust::host_vector<double>& lapx()const{return lapx_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_, lapx_;
    container g_xx_, g_xy_, g_yy_, vol2d_;
    orthogonal::RingGrid2d<container> g_assoc_;
};

/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
}//namespace orthogonal
}//namespace refined
}//namespace dg
