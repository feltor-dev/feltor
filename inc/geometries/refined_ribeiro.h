#pragma once


#include "dg/geometry/refined_grid.h"
#include "ribeiro.h"


namespace dg
{
namespace refined 
{
namespace ribeiro
{

///@cond
template< class container>
struct RingGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on "almost-ribeiro" coordinates by Ribeiro and Scott 2010
 *
 * @tparam container Vector class that holds metric coefficients
 */
template< class container>
struct RingGrid3d : public dg::refined::Grid3d
{
    typedef dg::CurvilinearCylindricalTag metric_category; //!< metric tag
    typedef RingGrid2d<container> perpendicular_grid; //!< the two-dimensional grid type

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
        assert( bcx == dg::PER|| bcx == dg::DIR);
        solovev::Psip psip(gp);
        solovev::PsipR psipR(gp);
        solovev::PsipZ psipZ(gp);
        ribeiro::detail::Fpsi<solovev::Psip, solovev::PsipR, solovev::PsipZ> fpsi( psip, psipR, psipZ, gp.R_0, 0);
        double x_1 = fpsi.find_x1( psi_0, psi_1);
        if( x_1 > 0)
            init_X_boundaries( 0., x_1);
        else
        {
            init_X_boundaries( x_1, 0.);
            std::swap( psi_0, psi_1);
        }
        //compute psi(x) for a grid on x and call construct_rzy for all psi
        ribeiro::detail::FieldFinv<solovev::Psip, solovev::PsipR, solovev::PsipZ> fpsiMinv_( psip, psipR, psipZ, gp.R_0, 0, 500);
        thrust::host_vector<double> x_vec(this->n()*this->Nx()); 
        for(unsigned i=0; i<x_vec.size(); i++) x_vec[i] = this->abscissasX()[i];
        thrust::host_vector<double> psi_x;
        dg::detail::construct_psi_values( fpsiMinv_, gp, psi_0, psi_1, this->x0(), x_vec, this->x1(), psi_x, f_x_);

        construct_rz( gp, psi_x);
        construct_metric();
    }
    const thrust::host_vector<double>& f_x()const{return f_x_;}
    thrust::host_vector<double> x()const{
        dg::Grid1d<double> gx( x0(), x1(), n(), Nx());
        return dg::create::abscissas(gx);}

    const thrust::host_vector<double>& f()const{return f_;}
    perpendicular_grid perp_grid() const { return ribeiro::refined::RingGrid2d<container>(*this);}
    const ribeiro::RingGrid3d<container>& associated() const{ return g_assoc_;}

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
    //call the construct_rzy function for all psi_x and lift to 3d grid
    //construct r,z,xr,xz,yr,yz,f_x
    void construct_rz( const solovev::GeomParameters& gp, thrust::host_vector<double>& psi_x)
    {
        //std::cout << "In grid function:\n";
        solovev::Psip psip(gp);
        solovev::PsipR psipR(gp);
        solovev::PsipZ psipZ(gp);
        solovev::PsipRR psipRR(gp);
        solovev::PsipRZ psipRZ(gp);
        solovev::PsipZZ psipZZ(gp);
        ribeiro::detail::Fpsi<solovev::Psip, solovev::PsipR, solovev::PsipZ> fpsi( psip, psipR, psipZ, gp.R_0, 0);
        solovev::ribeiro::FieldRZYRYZY<solovev::PsipR, solovev::PsipZ, solovev::PsipRR, solovev::PsipRZ, solovev::PsipZZ> fieldRZYRYZY(psipR, psipZ, psipRR, psipRZ, psipZZ);
        r_.resize(size()), z_.resize(size()), f_.resize(size());
        yr_ = r_, yz_ = z_, xr_ = r_, xz_ = r_ ;
        //r_x0.resize( psi_x.size()), z_x0.resize( psi_x.size());
        thrust::host_vector<double> f_p(f_x_);
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        thrust::host_vector<double> y_vec(this->n()*this->Ny());
        for(unsigned i=0; i<y_vec.size(); i++) y_vec[i] = this->abscissasY()[i*psi_x.size()];
        for( unsigned i=0; i<Nx; i++)
        {
            thrust::host_vector<double> ry, zy;
            thrust::host_vector<double> yr, yz, xr, xz;
            double R0, Z0;
            dg::detail::compute_rzy( fpsi, fieldRZYRYZY, psi_x[i], y_vec, ry, zy, yr, yz, xr, xz, R0, Z0, f_x_[i], f_p[i]);
            for( unsigned j=0; j<Ny; j++)
            {
                r_[j*Nx+i]  = ry[j], z_[j*Nx+i]  = zy[j], f_[j*Nx+i] = f_x_[i]; 
                yr_[j*Nx+i] = yr[j], yz_[j*Nx+i] = yz[j];
                xr_[j*Nx+i] = xr[j], xz_[j*Nx+i] = xz[j];
            }
        }
        //r_x1 = r_x0, z_x1 = z_x0; //periodic boundaries
        //now lift to 3D grid
        thrust::host_vector<double> wx = this->weightsX();
        thrust::host_vector<double> wy = this->weightsY();
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                f_[k*Nx*Ny+i] = f_[(k-1)*Nx*Ny+i];
                r_[k*Nx*Ny+i] = r_[(k-1)*Nx*Ny+i];
                z_[k*Nx*Ny+i] = z_[(k-1)*Nx*Ny+i];
                yr_[k*Nx*Ny+i] = yr_[(k-1)*Nx*Ny+i]*wy[k*Nx*Ny+i];
                yz_[k*Nx*Ny+i] = yz_[(k-1)*Nx*Ny+i]*wy[k*Nx*Ny+i];
                xr_[k*Nx*Ny+i] = xr_[(k-1)*Nx*Ny+i]*wx[k*Nx*Ny+i];
                xz_[k*Nx*Ny+i] = xz_[(k-1)*Nx*Ny+i]*wx[k*Nx*Ny+i];
            }
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric()
    {
        thrust::host_vector<double> tempxx( r_), tempxy(r_), tempyy(r_), tempvol(r_);
        for( unsigned idx=0; idx<this->size(); idx++)
        {
            tempxx[idx] = (xr_[idx]*xr_[idx]+xz_[idx]*xz_[idx]);
            tempxy[idx] = (yr_[idx]*xr_[idx]+yz_[idx]*xz_[idx]);
            tempyy[idx] = (yr_[idx]*yr_[idx]+yz_[idx]*yz_[idx]);
            //tempvol[idx] = r_[idx]/(f_[idx]*f_[idx] + tempxx[idx]);
            tempvol[idx] = r_[idx]/sqrt( tempxx[idx]*tempyy[idx] - tempxy[idx]*tempxy[idx] );
        }
        g_xx_=tempxx, g_xy_=tempxy, g_yy_=tempyy, vol_=tempvol;
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        vol2d_ = tempvol;
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    thrust::host_vector<double> f_x_; //1d vector
    thrust::host_vector<double> f_, r_, z_, xr_, xz_, yr_, yz_; //3d vector
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
    ribeiro::RingGrid3d<container> g_assoc_;
    
};

/**
 * @brief A two-dimensional grid based on "almost-ribeiro" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid2d : public dg::refined::Grid2d
{
    typedef dg::CurvilinearCylindricalTag metric_category;
    RingGrid2d( unsigned multiple_x, unsigned multiple_y, const solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx): 
        dg::refined::Grid2d( multiple_x, multiple_y, 0, 1., 0., 2*M_PI, n,n_old,Nx,Ny, bcx, dg::PER),
        g_assoc_( gp, psi_0, psi_1, n_old, Nx, Ny, bcx) 
    {
        ribeiro::refined::RingGrid3d<container> g( multiple_x, multiple_y, gp, psi_0, psi_1, n,n_old,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        f_x_ = g.f_x();
        f_ = g.f(), r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    RingGrid2d( const RingGrid3d<container>& g):
        dg::refined::Grid2d( g), g_assoc_( g.associated())
    {
        f_x_ = g.f_x();
        unsigned s = this->size();
        f_.resize(s), r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        {f_[i] = g.f()[i], r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.g_xy().begin(), g.g_xy().begin()+s, g_xy_.begin());
        thrust::copy( g.g_yy().begin(), g.g_yy().begin()+s, g_yy_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }

    const ribeiro::RingGrid2d<container>& associated()const{return g_assoc_;}
    const thrust::host_vector<double>& f_x()const{return f_x_;}
    thrust::host_vector<double> x()const{
        dg::Grid1d<double> gx( x0(), x1(), n(), Nx());
        return dg::create::abscissas(gx);}

    const thrust::host_vector<double>& f()const{return f_;}

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
    thrust::host_vector<double> f_x_; //1d vector
    thrust::host_vector<double> f_, r_, z_, xr_, xz_, yr_, yz_; //2d vector
    container g_xx_, g_xy_, g_yy_, vol2d_;
    ribeiro::RingGrid2d<container> g_assoc_;
};

}//namespace ribeiro
}//namespace refined
}//namespace dg

