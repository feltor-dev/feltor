#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/functions.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/operator.h"
#include "dg/backend/derivatives.h"
#include "dg/functors.h"
#include "dg/runge_kutta.h"
#include "dg/nullstelle.h"
#include "dg/geometry.h"
#include "fields.h"
#include "utilities.h"



namespace dg
{
namespace ribeiro
{

///@cond
namespace detail
{

//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as 
//good as it can, i.e. until machine precision is reached
template< class Psi, class PsiX, class PsiY>
struct Fpsi
{
    Fpsi( Psi psi, PsiX psiX, PsiY psiY, double x0, double y0): 
        psip_(psi), fieldRZYT_(psiX, psiY, x0, y0), fieldRZtau_(psiX, psiY)
    {
        R_init = x0; Z_init = y0;
        while( fabs( psiX(R_init, Z_init)) <= 1e-10 && fabs( psiY( R_init, Z_init)) <= 1e-10)
            R_init = x0 + 1.; Z_init = y0;
    }
    //finds the starting points for the integration in y direction
    void find_initial( double psi, double& R_0, double& Z_0) 
    {
        unsigned N = 50;
        thrust::host_vector<double> begin2d( 2, 0), end2d( begin2d), end2d_old(begin2d); 
        begin2d[0] = end2d[0] = end2d_old[0] = R_init;
        begin2d[1] = end2d[1] = end2d_old[1] = Z_init;
        //std::cout << "In init function\n";
        double eps = 1e10, eps_old = 2e10;
        while( (eps < eps_old || eps > 1e-7) && eps > 1e-14)
        {
            eps_old = eps; end2d_old = end2d;
            N*=2; dg::stepperRK17( fieldRZtau_, begin2d, end2d, psip_(R_init, Z_init), psi, N);
            eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) + (end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
        }
        R_init = R_0 = end2d_old[0], Z_init = Z_0 = end2d_old[1];
    }

    //compute f for a given psi between psi0 and psi1
    double construct_f( double psi, double& R_0, double& Z_0) 
    {
        find_initial( psi, R_0, Z_0);
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        begin[0] = R_0, begin[1] = Z_0;
        //std::cout << begin[0]<<" "<<begin[1]<<" "<<begin[2]<<"\n";
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 50;
        //double y_eps = 1;
        while( (eps < eps_old || eps > 1e-7)&& N < 1e6)
        {
            eps_old = eps, end_old = end;
            N*=2; dg::stepperRK17( fieldRZYT_, begin, end, 0., 2*M_PI, N);
            eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
        }
        //std::cout << "\t error "<<eps<<" with "<<N<<" steps\t";
        //std::cout <<end_old[2] << " "<<end[2] << "error in y is "<<y_eps<<"\n";
        double f_psi = 2.*M_PI/end_old[2];
        return f_psi;
    }
    double operator()( double psi)
    {
        double R_0, Z_0; 
        return construct_f( psi, R_0, Z_0);
    }

    /**
     * @brief This function computes the integral x_1 = -\int_{\psi_0}^{\psi_1} f(\psi) d\psi to machine precision
     *
     * @param psi_0 lower boundary 
     * @param psi_1 upper boundary 
     *
     * @return x1
     */
    double find_x1( double psi_0, double psi_1 ) 
    {
        unsigned P=8;
        double x1 = 0, x1_old = 0;
        double eps=1e10, eps_old=2e10;
        //std::cout << "In x1 function\n";
        while(eps < eps_old && P < 20 && eps > 1e-15)
        {
            eps_old = eps; 
            x1_old = x1;

            P+=1;
            dg::Grid1d<double> grid( psi_0, psi_1, P, 1);
            thrust::host_vector<double> psi_vec = dg::evaluate( dg::coo1, grid);
            thrust::host_vector<double> f_vec(grid.size(), 0);
            thrust::host_vector<double> w1d = dg::create::weights(grid);
            for( unsigned i=0; i<psi_vec.size(); i++)
            {
                f_vec[i] = this->operator()( psi_vec[i]);
            }
            x1 = dg::blas1::dot( f_vec, w1d);

            eps = fabs((x1 - x1_old)/x1);
            //std::cout << "X1 = "<<-x1<<" rel. error "<<eps<<" with "<<P<<" polynomials\n";
        }
        return -x1_old;
    }

    double f_prime( double psi) 
    {
        //compute fprime
        double deltaPsi = fabs(psi)/100.;
        double fofpsi[4];
        fofpsi[1] = operator()(psi-deltaPsi);
        fofpsi[2] = operator()(psi+deltaPsi);
        double fprime = (-0.5*fofpsi[1]+0.5*fofpsi[2])/deltaPsi, fprime_old;
        double eps = 1e10, eps_old=2e10;
        while( eps < eps_old)
        {
            deltaPsi /=2.;
            fprime_old = fprime;
            eps_old = eps;
            fofpsi[0] = fofpsi[1], fofpsi[3] = fofpsi[2];
            fofpsi[1] = operator()(psi-deltaPsi);
            fofpsi[2] = operator()(psi+deltaPsi);
            //reuse previously computed fpsi for current fprime
            fprime  = (+ 1./12.*fofpsi[0] 
                       - 2./3. *fofpsi[1]
                       + 2./3. *fofpsi[2]
                       - 1./12.*fofpsi[3]
                     )/deltaPsi;
            eps = fabs((fprime - fprime_old)/fprime);
            //std::cout << "fprime "<<fprime<<" rel error fprime is "<<eps<<" delta psi "<<deltaPsi<<"\n";
        }
        return fprime_old;
    }

    private:
    double R_init, Z_init;
    Psi psip_;
    solovev::ribeiro::FieldRZYT<PsiX, PsiY> fieldRZYT_;
    solovev::FieldRZtau<PsiX, PsiY> fieldRZtau_;
};

//This struct computes -2pi/f with a fixed number of steps for all psi
template<class Psi, class PsiR, class PsiZ>
struct FieldFinv
{
    FieldFinv( Psi psi, PsiR psiR, PsiZ psiZ, double x0, double y0, unsigned N_steps = 500):
        fpsi_(psi, psiR, psiZ, x0, y0), fieldRZYT_(psiR, psiZ, x0, y0), N_steps(N_steps) { }
    void operator()(const thrust::host_vector<double>& psi, thrust::host_vector<double>& fpsiM) 
    { 
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        fpsi_.find_initial( psi[0], begin[0], begin[1]);
        //std::cout << begin[0]<<" "<<begin[1]<<" "<<begin[2]<<"\n";
        dg::stepperRK17( fieldRZYT_, begin, end, 0., 2*M_PI, N_steps);
        //eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
        fpsiM[0] = - end[2]/2./M_PI;
        //std::cout <<"fpsiMinverse is "<<fpsiM[0]<<" "<<-1./fpsi_(psi[0])<<" "<<eps<<"\n";
    }
    private:
    Fpsi<Psi, PsiR, PsiZ> fpsi_;
    solovev::ribeiro::FieldRZYT<PsiR, PsiZ> fieldRZYT_;
    unsigned N_steps;
};
} //namespace detail

template< class container>
struct RingGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on "almost-ribeiro" coordinates by Ribeiro and Scott 2010
 *
 * @tparam container Vector class that holds metric coefficients
 */
template< class container>
struct RingGrid3d : public dg::Grid3d<double>
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
    RingGrid3d( solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx): 
        dg::Grid3d<double>( 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    { 
        solovev::Psip psip(gp); 
        solovev::PsipR psipR(gp); solovev::PsipZ psipZ(gp);
        solovev::PsipRR psipRR(gp); solovev::PsipZZ psipZZ(gp); solovev::PsipRZ psipRZ(gp);
        solovev::LaplacePsip lapPsip(gp); 
        construct( psip, psipR, psipZ, psipRR, psipRZ, psipZZ, psi_0, psi_1, gp.R_0, 0, n, Nx, Ny);
    }
    template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
    RingGrid3d( Psi psi, PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY,
            double psi_0, double psi_1, double x0, double y0, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx):
        dg::Grid3d<double>( 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    { 
        construct( psi, psiX, psiY, psiXX, psiXY, psiYY, psi_0, psi_1, x0, y0, n, Nx, Ny);
    }

    template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
    void construct( Psi psi, PsiX psiX, PsiY psiY, 
            PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, 
            double psi_0, double psi_1, 
            double x0, double y0, unsigned n, unsigned Nx, unsigned Ny)
    {
        assert( psi_1 != psi_0);
        assert( this->bcx() == dg::PER|| this->bcx() == dg::DIR);
        ribeiro::detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi, psiX, psiY, x0, y0);
        double x_1 = fpsi.find_x1( psi_0, psi_1);
        if( x_1 > 0)
            init_X_boundaries( 0., x_1);
        else
        {
            init_X_boundaries( x_1, 0.);
            std::swap( psi_0, psi_1);
        }
        //compute psi(x) for a grid on x and call construct_rzy for all psi
        detail::FieldFinv<Psi, PsiX, PsiY> fpsiMinv_(psi, psiX, psiY, x0,y0, 500);
        dg::Grid1d<double> g1d_( this->x0(), this->x1(), n, Nx, this->bcx());
        thrust::host_vector<double> x_vec = dg::evaluate( dg::coo1, g1d_);
        thrust::host_vector<double> psi_x;
        dg::detail::construct_psi_values( fpsiMinv_, psi_0, psi_1, this->x0(), x_vec, this->x1(), psi_x, f_x_);

        construct_rz( psi, psiX, psiY, psiXX, psiXY, psiYY, x0, y0, psi_x);
        construct_metric();
    }
    const thrust::host_vector<double>& f_x()const{return f_x_;}
    thrust::host_vector<double> x()const{
        dg::Grid1d<double> gx( x0(), x1(), n(), Nx());
        return dg::create::abscissas(gx);}

    const thrust::host_vector<double>& f()const{return f_;}
    perpendicular_grid perp_grid() const { return ribeiro::RingGrid2d<container>(*this);}

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
    template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
    void construct_rz( Psi psi, PsiX psiX, PsiY psiY, 
            PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, double x0, double y0, thrust::host_vector<double>& psi_x)
    {
        //std::cout << "In grid function:\n";
        detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi, psiX, psiY, x0, y0);
        solovev::ribeiro::FieldRZYRYZY<PsiX, PsiY, PsiXX, PsiXY, PsiYY> fieldRZYRYZY(psiX, psiY, psiXX, psiXY, psiYY);
        r_.resize(size()), z_.resize(size()), f_.resize(size());
        yr_ = r_, yz_ = z_, xr_ = r_, xz_ = r_ ;
        //r_x0.resize( psi_x.size()), z_x0.resize( psi_x.size());
        thrust::host_vector<double> f_p(f_x_);
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        dg::Grid1d<double> g1d( 0., 2.*M_PI, this->n(), this->Ny());
        const thrust::host_vector<double> y_vec = dg::create::abscissas( g1d);
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
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                f_[k*Nx*Ny+i] = f_[(k-1)*Nx*Ny+i];
                r_[k*Nx*Ny+i] = r_[(k-1)*Nx*Ny+i];
                z_[k*Nx*Ny+i] = z_[(k-1)*Nx*Ny+i];
                yr_[k*Nx*Ny+i] = yr_[(k-1)*Nx*Ny+i];
                yz_[k*Nx*Ny+i] = yz_[(k-1)*Nx*Ny+i];
                xr_[k*Nx*Ny+i] = xr_[(k-1)*Nx*Ny+i];
                xz_[k*Nx*Ny+i] = xz_[(k-1)*Nx*Ny+i];
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
    
    //The following points might also be useful for external grid generation
    //thrust::host_vector<double> r_0y, r_1y, z_0y, z_1y; //boundary points in x
    //thrust::host_vector<double> r_x0, r_x1, z_x0, z_x1; //boundary points in y

};

/**
 * @brief A two-dimensional grid based on "almost-ribeiro" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid2d : public dg::Grid2d<double>
{
    typedef dg::CurvilinearCylindricalTag metric_category;
    RingGrid2d( const solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx): 
        dg::Grid2d<double>( 0, 1., 0., 2*M_PI, n,Nx,Ny, bcx, dg::PER)
    {
        ribeiro::RingGrid3d<container> g( gp, psi_0, psi_1, n,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        f_x_ = g.f_x();
        f_ = g.f(), r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    RingGrid2d( const RingGrid3d<container>& g):
        dg::Grid2d<double>( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
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
};

/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
struct Field
{
    Field( solovev::GeomParameters gp,const thrust::host_vector<double>& x, const thrust::host_vector<double>& f_x):
        gp_(gp),
        psipR_(gp), psipZ_(gp),
        ipol_(gp), invB_(gp), last_idx(0), x_(x), fx_(f_x)
    { }

    /**
     * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
     \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
     \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I}  \hat{R}_0}  \f]
     */ 
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        //x,y,s,R,Z
        double psipR = psipR_(y[3],y[4]), psipZ = psipZ_(y[3],y[4]), ipol = ipol_( y[3],y[4]);
        double fx = find_fx( y[0]);
        yp[0] = 0;
        yp[1] = fx*y[3]*(0.0+1.00*(psipR*psipR+psipZ*psipZ))/ipol;
        yp[2] =  y[3]*y[3]/invB_(y[3],y[4])/ipol/gp_.R_0; //ds/dphi =  R^2 B/I/R_0_hat
        yp[3] =  y[3]*psipZ/ipol;              //dR/dphi =  R/I Psip_Z
        yp[4] = -y[3]*psipR/ipol;             //dZ/dphi = -R/I Psip_R

    }
    /**
     * @brief \f[   \frac{1}{\hat{B}} = 
      \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
     */ 
    double operator()( double R, double Z) const { return invB_(R,Z); }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const { return invB_(R,Z,phi); }
    double error( const dg::HVec& x0, const dg::HVec& x1)
    {
        //compute error in x,y,s
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end){ 
        if ( isnan(end[1]) || isnan(end[2]) || isnan(end[3])||isnan( end[4]) ) 
        {
            return false;
        }
        if( (end[3] < 1e-5) || end[3]*end[3] > 1e10 ||end[1]*end[1] > 1e10 ||end[2]*end[2] > 1e10 ||(end[4]*end[4] > 1e10) )
        {
            return false;
        }
        return true;
    }
    
    private:
    double find_fx(double x) 
    {
        if( fabs(x-x_[last_idx]) < 1e-12)
            return fx_[last_idx];
        for( unsigned i=0; i<x_.size(); i++)
            if( fabs(x-x_[i]) < 1e-12)
            {
                last_idx = (int)i;
                return fx_[i];
            }
        std::cerr << "x not found!!\n";
        return 0;
    }
    
    solovev::GeomParameters gp_;
    solovev::PsipR  psipR_;
    solovev::PsipZ  psipZ_;
    solovev::Ipol   ipol_;
    solovev::InvB   invB_;
    int last_idx;
    thrust::host_vector<double> x_;
    thrust::host_vector<double> fx_;
   
};

}//namespace ribeiro
} //namespace dg
