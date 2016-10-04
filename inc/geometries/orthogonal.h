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



namespace dg
{
namespace orthogonal
{

///@cond
namespace detail
{

//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as 
//good as it can, i.e. until machine precision is reached
template< class Psi, class PsiX, class PsiY>
struct Fpsi
{
    
    //firstline = 0 -> conformal, firstline = 1 -> equalarc
    Fpsi( Psi psi, PsiX psiX, PsiY psiY, double x0, double y0, int firstline): 
        psip_(psi), fieldRZYTconf_(psiX, psiY, x0, y0),fieldRZYTequl_(psiX, psiY, x0, y0), fieldRZtau_(psiX, psiY)
    {
        X_init = x0, Y_init = y0;
        while( fabs( psiX(X_init, Y_init)) <= 1e-10 && fabs( psiY( X_init, Y_init)) <= 1e-10)
            X_init +=  1.; 
        firstline_ = firstline;
    }
    //finds the starting points for the integration in y direction
    void find_initial( double psi, double& R_0, double& Z_0) 
    {
        unsigned N = 50;
        thrust::host_vector<double> begin2d( 2, 0), end2d( begin2d), end2d_old(begin2d); 
        begin2d[0] = end2d[0] = end2d_old[0] = X_init;
        begin2d[1] = end2d[1] = end2d_old[1] = Y_init;
        double eps = 1e10, eps_old = 2e10;
        while( (eps < eps_old || eps > 1e-7) && eps > 1e-14)
        {
            eps_old = eps; end2d_old = end2d;
            N*=2; dg::stepperRK17( fieldRZtau_, begin2d, end2d, psip_(X_init, Y_init), psi, N);
            eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) + (end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
        }
        X_init = R_0 = end2d_old[0], Y_init = Z_0 = end2d_old[1];
        //std::cout << "In init function error: psi(R,Z)-psi0: "<<psip_(X_init, Y_init)-psi<<"\n";
    }

    //compute f for a given psi between psi0 and psi1
    double construct_f( double psi, double& R_0, double& Z_0) 
    {
        find_initial( psi, R_0, Z_0);
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        begin[0] = R_0, begin[1] = Z_0;
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 50;
        while( (eps < eps_old || eps > 1e-7)&& eps > 1e-14)
        {
            eps_old = eps, end_old = end; N*=2; 
            if( firstline_ == 0)
                dg::stepperRK17( fieldRZYTconf_, begin, end, 0., 2*M_PI, N);
            if( firstline_ == 1)
                dg::stepperRK17( fieldRZYTequl_, begin, end, 0., 2*M_PI, N);
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

    private:
    int firstline_;
    double X_init, Y_init;
    Psi psip_;
    solovev::ribeiro::FieldRZYT<PsiX, PsiY> fieldRZYTconf_;
    solovev::equalarc::FieldRZYT<PsiX, PsiY> fieldRZYTequl_;
    solovev::FieldRZtau<PsiX, PsiY> fieldRZtau_;

};

//compute the vector of r and z - values that form one psi surface
//assumes y_0 = 0
template <class PsiX, class PsiY>
void compute_rzy( PsiX psiX, PsiY psiY, const thrust::host_vector<double>& y_vec,
        thrust::host_vector<double>& r, 
        thrust::host_vector<double>& z, 
        double R_0, double Z_0, double f_psi, int mode ) 
{

    thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old);
    thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size());
    thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
    begin[0] = R_0, begin[1] = Z_0;
    //std::cout <<f_psi<<" "<<" "<< begin[0] << " "<<begin[1]<<"\t";
    solovev::ribeiro::FieldRZY<PsiX, PsiY> fieldRZYconf(psiX, psiY);
    solovev::equalarc::FieldRZY<PsiX, PsiY> fieldRZYequi(psiX, psiY);
    fieldRZYconf.set_f(f_psi);
    fieldRZYequi.set_f(f_psi);
    unsigned steps = 1;
    double eps = 1e10, eps_old=2e10;
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-14)
    {
        //begin is left const
        eps_old = eps, r_old = r, z_old = z;
        if(mode==0)dg::stepperRK17( fieldRZYconf, begin, end, 0, y_vec[0], steps);
        if(mode==1)dg::stepperRK17( fieldRZYequi, begin, end, 0, y_vec[0], steps);
        r[0] = end[0], z[0] = end[1];
        for( unsigned i=1; i<y_vec.size(); i++)
        {
            temp = end; 
            if(mode==0)dg::stepperRK17( fieldRZYconf, temp, end, y_vec[i-1], y_vec[i], steps);
            if(mode==1)dg::stepperRK17( fieldRZYequi, temp, end, y_vec[i-1], y_vec[i], steps);
            r[i] = end[0], z[i] = end[1];
        }
        //compute error in R,Z only
        dg::blas1::axpby( 1., r, -1., r_old, r_diff);
        dg::blas1::axpby( 1., z, -1., z_old, z_diff);
        double er = dg::blas1::dot( r_diff, r_diff);
        double ez = dg::blas1::dot( z_diff, z_diff);
        double ar = dg::blas1::dot( r, r);
        double az = dg::blas1::dot( z, z);
        eps =  sqrt( er + ez)/sqrt(ar+az);
        //std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        //temp = end; 
        //if(mode==0)dg::stepperRK17( fieldRZYconf, temp, end, y_vec[y_vec.size()-1], 2.*M_PI, steps);
        //if(mode==1)dg::stepperRK17( fieldRZYequi, temp, end, y_vec[y_vec.size()-1], 2.*M_PI, steps);
        //std::cout << "abs. error is "<<sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]))<<"\n";
        steps*=2;
    }
    r = r_old, z = z_old;

}

//This struct computes -2pi/f with a fixed number of steps for all psi
//and provides the Nemov algorithm for orthogonal grid
//template< class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY, class LaplacePsiX, class LaplacePsiY>
template< class PsiX, class PsiY, class LaplacePsi>
struct Nemov
{
    Nemov( PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double f0, int mode):
        f0_(f0), mode_(mode),
        psipR_(psiX), psipZ_(psiY),
        laplacePsip_( laplacePsi)
            { }
    void initialize( 
        const thrust::host_vector<double>& r_init, //1d intial values
        const thrust::host_vector<double>& z_init, //1d intial values
        thrust::host_vector<double>& h_init) //,
    //    thrust::host_vector<double>& hr_init,
    //    thrust::host_vector<double>& hz_init)
    {
        unsigned size = r_init.size(); 
        h_init.resize( size);//, hr_init.resize( size), hz_init.resize( size);
        for( unsigned i=0; i<size; i++)
        {
            if(mode_ == 0)
                h_init[i] = f0_;
            if(mode_ == 1)
            {
                double psipR = psipR_(r_init[i], z_init[i]), 
                       psipZ = psipZ_(r_init[i], z_init[i]);
                double psip2 = (psipR*psipR+psipZ*psipZ);
                h_init[i]  = f0_/sqrt(psip2); //equalarc
            }
            //double laplace = psipRR_(r_init[i], z_init[i]) + 
                             //psipZZ_(r_init[i], z_init[i]);
            //hr_init[i] = -f0_*laplace/psip2*psipR;
            //hz_init[i] = -f0_*laplace/psip2*psipZ;
        }
    }

    void operator()(const std::vector<thrust::host_vector<double> >& y, std::vector<thrust::host_vector<double> >& yp) 
    { 
        //y[0] = R, y[1] = Z, y[2] = h, y[3] = hr, y[4] = hz
        unsigned size = y[0].size();
        double psipR, psipZ, psip2;
        for( unsigned i=0; i<size; i++)
        {
            psipR = psipR_(y[0][i], y[1][i]), psipZ = psipZ_(y[0][i], y[1][i]);
            //psipRR = psipRR_(y[0][i], y[1][i]), psipRZ = psipRZ_(y[0][i], y[1][i]), psipZZ = psipZZ_(y[0][i], y[1][i]);
            psip2 = f0_*(psipR*psipR+psipZ*psipZ);
            yp[0][i] = psipR/psip2;
            yp[1][i] = psipZ/psip2;
            yp[2][i] = y[2][i]*( -laplacePsip_(y[0][i], y[1][i]) )/psip2;
            //yp[3][i] = ( -(2.*psipRR+psipZZ)*y[3][i] - psipRZ*y[4][i] - laplacePsipR_(y[0][i], y[1][i])*y[2][i])/psip2;
            //yp[4][i] = ( -psipRZ*y[3][i] - (2.*psipZZ+psipRR)*y[4][i] - laplacePsipZ_(y[0][i], y[1][i])*y[2][i])/psip2;
        }
    }
    private:
    double f0_;
    int mode_;
    PsiX psipR_;
    PsiY psipZ_;
    LaplacePsi laplacePsip_;
};

template<class Nemov>
void construct_rz( Nemov nemov, 
        double x_0, //the x value that corresponds to the first psi surface
        const thrust::host_vector<double>& x_vec,  //1d x values
        const thrust::host_vector<double>& r_init, //1d intial values of the first psi surface
        const thrust::host_vector<double>& z_init, //1d intial values of the first psi surface
        thrust::host_vector<double>& r, 
        thrust::host_vector<double>& z, 
        thrust::host_vector<double>& h//,
        //thrust::host_vector<double>& hr,
        //thrust::host_vector<double>& hz
    )
{
    unsigned N = 1;
    double eps = 1e10, eps_old=2e10;
    std::vector<thrust::host_vector<double> > begin(3); //begin(5);
    thrust::host_vector<double> h_init( r_init.size(), 0.);
    //thrust::host_vector<double> h_init, hr_init, hz_init;
    nemov.initialize( r_init, z_init, h_init);//, hr_init, hz_init);
    begin[0] = r_init, begin[1] = z_init, 
    begin[2] = h_init; //begin[3] = hr_init, begin[4] = hz_init;
    //now we have the starting values 
    std::vector<thrust::host_vector<double> > end(begin), temp(begin);
    unsigned sizeX = x_vec.size(), sizeY = r_init.size();
    unsigned size2d = x_vec.size()*r_init.size();
    r.resize(size2d), z.resize(size2d), h.resize(size2d); //hr.resize(size2d), hz.resize(size2d);
    double x0=x_0, x1 = x_vec[0];
    thrust::host_vector<double> r_old(r), r_diff( r), z_old(z), z_diff(z);
    while( (eps < eps_old || eps > 1e-6) && eps > 1e-13)
    {
        r_old = r, z_old = z; eps_old = eps; 
        temp = begin;
        //////////////////////////////////////////////////
        for( unsigned i=0; i<sizeX; i++)
        {
            x0 = i==0?x_0:x_vec[i-1], x1 = x_vec[i];
            //////////////////////////////////////////////////
            dg::stepperRK17( nemov, temp, end, x0, x1, N);
            for( unsigned j=0; j<sizeY; j++)
            {
                unsigned idx = j*sizeX+i;
                 r[idx] = end[0][j],  z[idx] = end[1][j];
                //hr[idx] = end[3][j], hz[idx] = end[4][j];
                 h[idx] = end[2][j]; 
            }
            //////////////////////////////////////////////////
            temp = end;
        }
        dg::blas1::axpby( 1., r, -1., r_old, r_diff);
        dg::blas1::axpby( 1., z, -1., z_old, z_diff);
        dg::blas1::pointwiseDot( r_diff, r_diff, r_diff);
        dg::blas1::pointwiseDot( 1., z_diff, z_diff, 1., r_diff);
        eps = sqrt( dg::blas1::dot( r_diff, r_diff)/sizeX/sizeY); //should be relative to the interpoint distances
        std::cout << "Effective Absolute diff error is "<<eps<<" with "<<N<<" steps\n"; 
        N*=2;
    }

}

} //namespace detail

}//namespace orthogonal

template< class Psi, class PsiX, class PsiY, class LaplacePsi>
struct SimpleOrthogonal
{
    SimpleOrthogonal( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double psi_0, double psi_1, double x0, double y0, int firstline =0):
        psiX_(psiX), psiY_(psiY), laplacePsi_(laplacePsi)
    {
        assert( psi_1 != psi_0);
        firstline_ = firstline;
        orthogonal::detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi, psiX, psiY, x0, y0, firstline);
        f0_ = fabs( fpsi.construct_f( psi_0, R0_, Z0_));
        if( psi_1 < psi_0) f0_*=-1;
        lz_ =  f0_*(psi_1-psi_0);
    }
    double f0() const{return f0_;}
    double width() const{return lz_;}
    double height() const{return 2.*M_PI;}
    bool isOrthogonal() const{return true;}
    bool isConformal()  const{return false;}
    void operator()( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) 
    {
        thrust::host_vector<double> r_init, z_init;
        orthogonal::detail::compute_rzy( psiX_, psiY_, eta1d, r_init, z_init, R0_, Z0_, f0_, firstline_);
        orthogonal::detail::Nemov<PsiX, PsiY, LaplacePsi> nemov(psiX_, psiY_, laplacePsi_, f0_, firstline_);
        thrust::host_vector<double> h;
        orthogonal::detail::construct_rz(nemov, 0., zeta1d, r_init, z_init, x, y, h);
        unsigned size = x.size();
        zetaX.resize(size), zetaY.resize(size), 
        etaX.resize(size), etaY.resize(size);
        for( unsigned idx=0; idx<size; idx++)
        {
            double psipR = psiX_(x[idx], y[idx]);
            double psipZ = psiY_(x[idx], y[idx]);
            zetaX[idx] = f0_*psipR;
            zetaY[idx] = f0_*psipZ;
            etaX[idx] = -h[idx]*psipZ;
            etaY[idx] = +h[idx]*psipR;
        }
    }
    private:
    PsiX psiX_;
    PsiY psiY_;
    LaplacePsi laplacePsi_;
    double f0_, lz_, R0_, Z0_;
    int firstline_;
};

namespace orthogonal
{

template< class container>
struct RingGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on orthogonal coordinates
 */
template< class container>
struct RingGrid3d : public dg::Grid3d<double>
{
    typedef dg::OrthogonalTag metric_category;
    typedef RingGrid2d<container> perpendicular_grid;

    template< class Generator>
    RingGrid3d( Generator generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx):
        dg::Grid3d<double>( 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    { 
        construct( generator, n, Nx, Ny);
    }

    perpendicular_grid perp_grid() const { return orthogonal::RingGrid2d<container>(*this);}
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
        assert( generator.isOrthogonal());
        dg::Grid1d<double> gY1d( 0, 2*M_PI, n, Ny, dg::PER);
        dg::Grid1d<double> gX1d( 0., generator.width(), n, Nx);
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
 */
template< class container>
struct RingGrid2d : public dg::Grid2d<double>
{
    typedef dg::OrthogonalTag metric_category;
    template< class Generator>
    RingGrid2d( Generator generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx):
        dg::Grid2d<double>( 0, 1, 0., 2.*M_PI, n, Nx, Ny, bcx, dg::PER)
    {
        orthogonal::RingGrid3d<container> g( generator, n,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();

    }
    RingGrid2d( const RingGrid3d<container>& g):
        dg::Grid2d<double>( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
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

/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
struct Field
{
    Field( solovev::GeomParameters gp, const dg::Grid2d<double>& gXY, const thrust::host_vector<double>& f2):
        gp_(gp),
        psipR_(gp), psipZ_(gp),
        ipol_(gp), invB_(gp), gXY_(gXY), g_(dg::create::forward_transform(f2, gXY)) 
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
        double xs = y[0],ys=y[1];
        gXY_.shift_topologic( y[0], M_PI, xs,ys);
        double g = dg::interpolate( xs,  ys, g_, gXY_);
        yp[0] = 0;
        yp[1] = y[3]*g*(psipR*psipR+psipZ*psipZ)/ipol;
        //yp[1] = g/ipol;
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
    solovev::GeomParameters gp_;
    solovev::PsipR  psipR_;
    solovev::PsipZ  psipZ_;
    solovev::Ipol   ipol_;
    solovev::InvB   invB_;
    const dg::Grid2d<double> gXY_;
    thrust::host_vector<double> g_;
   
};

}//namespace orthogonal
}//namespace dg
