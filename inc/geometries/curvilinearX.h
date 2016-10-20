#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/gridX.h"
#include "dg/backend/evaluationX.cuh"
#include "dg/backend/weightsX.cuh"
#include "dg/runge_kutta.h"
#include "utilitiesX.h"
#include "ribeiro.h"

//! ATTENTION: algorithm might be flawed since f(psi) might make a jump at
//! ATTENTION: separatrix


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
struct FpsiX
{
    FpsiX( Psi psi, PsiX psiX, PsiY psiY, double xX, double yX, double x0, double y0): 
        initX_(psi, psiX, psiY, xX, yX), fieldRZYT_(psiX, psiY, x0, y0), fieldRZYZ_(psiX, psiY)
    { }
    //for a given psi finds the four starting points for the integration in y direction on the perpendicular lines through the X-point
    void find_initial( double psi, double* R_0, double* Z_0) 
    {
        initX_.find_initial(psi, R_0, Z_0);
    }

    //compute f for a given psi between psi0 and psi1
    double construct_f( double psi, double* R_i, double* Z_i) 
    {
        find_initial( psi, R_i, Z_i);
        //std::cout << "Begin error "<<eps_old<<" with "<<N<<" steps\n";
        //std::cout << "In Stepper function:\n";
        //double y_old=0;
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        begin[0] = R_i[0], begin[1] = Z_i[0];
        //std::cout << begin[0]<<" "<<begin[1]<<" "<<begin[2]<<"\n";
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 32; 
        //double y_eps;
        while( (eps < eps_old || eps > 1e-7) && N < 1e6)
        {
            //remember old values
            eps_old = eps, end_old = end; //y_old = end[2];
            //compute new values
            N*=2;
            if( psi < 0)
            {
                dg::stepperRK17( fieldRZYT_, begin, end, 0., 2.*M_PI, N);
                //std::cout << "result is "<<end[0]<<" "<<end[1]<<" "<<end[2]<<"\n";
                eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
            }
            else
            {
                dg::stepperRK17( fieldRZYZ_, begin, end, begin[1], 0., N);
                thrust::host_vector<double> temp(end);
                dg::stepperRK17( fieldRZYT_, temp, end, 0., M_PI, N);
                temp = end; //temp[1] should be 0 now
                dg::stepperRK17( fieldRZYZ_, temp, end, temp[1], Z_i[1], N);
                eps = sqrt( (end[0]-R_i[1])*(end[0]-R_i[1]) + (end[1]-Z_i[1])*(end[1]-Z_i[1]));
            }
            if( isnan(eps)) { eps = eps_old/2.; end = end_old; 
                //std::cerr << "\t nan! error "<<eps<<"\n";
            } //near X-point integration can go wrong
            //y_eps = sqrt( (y_old - end[2])*(y_old-end[2]));
            //std::cout << "error "<<eps<<" with "<<N<<" steps| psip "<<psi_(end[0], end[1])<<"\n";
            //std::cout <<"error in y is "<<y_eps<<"\n";
        }
        double f_psi = 2.*M_PI/end_old[2];
        //std::cout << "f_psi "<<f_psi<<"\n";
        return f_psi;
        //return 1./f_psi;
    }
    double operator()( double psi)
    {
        double R_0[2], Z_0[2]; 
        return construct_f( psi, R_0, Z_0);
    }

    /**
     * @brief This function computes the integral x_0 = \int_{\psi}^{0} f(\psi) d\psi to machine precision
     *
     * @return x0
     */
    double find_x( double psi ) 
    {
        unsigned P=6;
        double x0 = 0, x0_old = 0;
        double eps=1e10, eps_old=2e10;
        //std::cout << "In x1 function\n";
        while( (eps < eps_old||eps>1e-7) && P < 20 )
        {
            eps_old = eps; x0_old = x0;
            P+=2;
            dg::Grid1d grid( 0, 1, P, 1);
            if( psi>0)
            {
                dg::Grid1d grid1( 0, psi, P, 1);
                grid = grid1;
            }
            else 
            {
                dg::Grid1d grid2( psi, 0, P, 1);
                grid = grid2;
            }
            thrust::host_vector<double> psi_vec = dg::evaluate( dg::cooX1d, grid);
            thrust::host_vector<double> f_vec(grid.size(), 0);
            thrust::host_vector<double> w1d = dg::create::weights(grid);
            for( unsigned i=0; i<psi_vec.size(); i++)
            {
                f_vec[i] = this->operator()( psi_vec[i]);
                //std::cout << " "<<f_vec[i]<<"\n";
            }
            if( psi < 0)
                x0 = dg::blas1::dot( f_vec, w1d);
            else
                x0 = -dg::blas1::dot( f_vec, w1d);

            eps = fabs((x0 - x0_old)/x0);
            if( isnan(eps)) { std::cerr << "Attention!!\n"; eps = eps_old -1e-15; x0 = x0_old;} //near X-point integration can go wrong
            std::cout << "X = "<<-x0<<" rel. error "<<eps<<" with "<<P<<" polynomials\n";
        }
        return -x0_old;

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
        while( eps < eps_old || eps > 1e-7)
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
        }
        //std::cout << "\t fprime "<<fprime<<" rel error fprime is "<<eps<<" delta psi "<<deltaPsi<<"\n";
        return fprime_old;
    }
    private:
    dg::orthogonal::detail::InitialX<Psi, PsiX, PsiY> initX_;
    const solovev::ribeiro::FieldRZYT<PsiX, PsiY> fieldRZYT_;
    const solovev::ribeiro::FieldRZYZ<PsiX, PsiY> fieldRZYZ_;

};

//This struct computes -2pi/f with a fixed number of steps for all psi
template<class Psi, class PsiX, class PsiY>
struct XFieldFinv
{
    XFieldFinv( Psi psi, PsiX psiX, PsiY psiY, double xX, double yX, double x0, double y0, unsigned N_steps = 500): 
        fpsi_(psi, psiX, psiY, xX, yX, x0, y0), fieldRZYT_(psiX, psiY, x0, y0), fieldRZYZ_(psiX, psiY) , N_steps(N_steps)
            { xAtOne_ = fpsi_.find_x(0.1); }
    void operator()(const thrust::host_vector<double>& psi, thrust::host_vector<double>& fpsiM) 
    { 
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        double R_i[2], Z_i[2];
        dg::Timer t;
        t.tic();
        fpsi_.find_initial( psi[0], R_i, Z_i);
        t.toc();
        //std::cout << "find_initial took "<<t.diff()<< "s\n";
        t.tic();
        begin[0] = R_i[0], begin[1] = Z_i[0];
        unsigned N = N_steps;
        if( psi[0] < -1. && psi[0] > -2.) N*=2;
        if( psi[0] < 0 && psi[0] > -1.) N*=10;
        if( psi[0] <0  )
            dg::stepperRK17( fieldRZYT_, begin, end, 0., 2.*M_PI, N);
        else
        {
            dg::stepperRK17( fieldRZYZ_, begin, end, begin[1], 0., N);
            thrust::host_vector<double> temp(end);
            dg::stepperRK17( fieldRZYT_, temp, end, 0., M_PI, N/2);
            temp = end; //temp[1] should be 0 now
            dg::stepperRK17( fieldRZYZ_, temp, end, temp[1], Z_i[1], N);
        }
        //eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
        fpsiM[0] = end[2]/2./M_PI;
        //fpsiM[0] = - 2.*M_PI/end[2];
        t.toc();
        //std::cout << "Finding f took "<<t.diff()<<"s\n";
        //std::cout <<"fpsiMinverse is "<<fpsiM[0]<<" "<<-1./fpsi_(psi[0])<<" "<<eps<<"\n";
    }
    double find_psi( double x)
    {
        assert( x > 0);
        //integrate from x0 to x, with psi(x0) = 0.1;
        double x0 = xAtOne_; 
        thrust::host_vector<double> begin( 1, 0.1), end(begin), end_old(begin);
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 1;
        while( eps < eps_old && N < 1e6 &&  eps > 1e-9)
        {
            eps_old = eps, end_old = end; 
            N*=2; dg::stepperRK17( *this, begin, end, x0, x, N);
            eps = fabs( end[0]- end_old[0]);
            //std::cout << "\t error "<<eps<<" with "<<N<<" steps\n";
        }
        return end_old[0];
    }

    private:
    FpsiX<Psi, PsiX, PsiY> fpsi_;
    solovev::ribeiro::FieldRZYT<PsiX, PsiY> fieldRZYT_;
    solovev::ribeiro::FieldRZYZ<PsiX, PsiY> fieldRZYZ_;
    thrust::host_vector<double> fpsi_neg_inv;
    unsigned N_steps;
    double xAtOne_;
};
} //namespace detail
///@endcond
}//namespace ribeiro

/**
 * @brief A two-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 * @ingroup generators
 * @tparam Psi All the template parameters must model a Binary-operator i.e. the bracket operator() must be callable with two arguments and return a double. 
 */
template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
struct RibeiroX
{
    RibeiroX( Psi psi, PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, double psi_0, double fx, 
            double xX, double yX, double x0, double y0):
        psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY), fpsi_(psi, psiX, psiY, xX, yX, x0,y0), fpsiMinv_(psi, psiX, psiY, xX, yX, x0,y0, 500)
    {
        assert( psi_0 < 0 );
        zeta0_ = fpsi_.find_x( psi_0);
        f0_=zeta0_/psi_0;
        zeta1_= -fx/(1.-fx)*zeta0_;
        x0_=x0, y0_=y0, psi0_=psi_0;
    }
    bool isConformal()const{return false;}
    bool isOrthogonal()const{return false;}
    double f0() const{return f0_;}
    /**
     * @brief The length of the zeta-domain
     *
     * Call before discretizing the zeta domain
     * @return length of zeta-domain (f0*(psi_1-psi_0))
     * @note the length is always positive
     */
    double width() const{return lx_;}
    /**
     * @brief 2pi (length of the eta domain)
     *
     * Always returns 2pi
     * @return 2pi 
     */
    double height() const{return 2.*M_PI;}
    /**
     * @brief The vector f(x)
     *
     * @return f(x)
     */
    thrust::host_vector<double> fx() const{ return fx_;}
    double psi1() const {return psi_1_numerical_;}
    /**
     * @brief Generate the points and the elements of the Jacobian
     *
     * Call the width() and height() function before calling this function!
     * @param zeta1d one-dimensional list of points inside the zeta-domain (0<zeta<width())
     * @param eta1d one-dimensional list of points inside the eta-domain (0<eta<height())
     * @param x  = x(zeta,eta)
     * @param y  = y(zeta,eta)
     * @param zetaX = zeta_x(zeta,eta)
     * @param zetaY = zeta_y(zeta,eta)
     * @param etaX = eta_x(zeta,eta)
     * @param etaY = eta_y(zeta,eta)
     * @note All the resulting vectors are write-only and get properly resized
     */
    void operator()( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         const unsigned nodeX0, const unsigned nodeX1, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) 
    {
        //compute psi(x) for a grid on x and call construct_rzy for all psi
        unsigned inside=0;
        for(unsigned i=0; i<zeta1d.size(); i++)
            if( zeta1d[i]< 0) inside++;//how many points are inside
        thrust::host_vector<double> psi_x;
        psi_1_numerical_ = dg::detail::construct_psi_values( fpsiMinv_, psi0_, zeta0_, zeta1d, zeta1_, inside, psi_x);

        //std::cout << "In grid function:\n";
        solovev::ribeiro::FieldRZYRYZY<PsiX, PsiY, PsiXX, PsiXY, PsiYY> fieldRZYRYZYribeiro(psiX_, psiY_, psiXX_, psiXY_, psiYY_);
        unsigned size = zeta1d.size()*eta1d.size();
        x.resize(size), y.resize(size);
        zetaX = zetaY = etaX = etaY =x ;
        unsigned Nx = zeta1d.size(), Ny = eta1d.size();
        fx_.resize(Nx);
        for( unsigned i=0; i<zeta1d.size(); i++)
        {
            thrust::host_vector<double> ry, zy;
            thrust::host_vector<double> yr, yz, xr, xz;
            double R0[2], Z0[2];
            dg::detail::computeX_rzy( fpsi_, fieldRZYRYZYribeiro, psi_x[i], eta1d, nodeX0, nodeX1, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i]);
            for( unsigned j=0; j<Ny; j++)
            {
                x[j*Nx+i]  = ry[j], y[j*Nx+i]  = zy[j];
                etaX[j*Nx+i] = yr[j], etaY[j*Nx+i] = yz[j];
                zetaX[j*Nx+i] = xr[j], zetaY[j*Nx+i] = xz[j];
            }
        }
    }
    private:
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
    dg::ribeiro::detail::XFieldFinv<Psi, PsiX, PsiY> fpsiMinv_; 
    dg::ribeiro::detail::FpsiX<Psi, PsiX, PsiY> fpsi_;
    double f0_, psi_1_numerical_;
    thrust::host_vector<double> fx_;
    double zeta0_, zeta1_;
    double lx_, x0_, y0_, psi0_, psi1_;
    int mode_; //0 = ribeiro, 1 = equalarc
};

///@cond
template< class container>
struct CurvilinearGridX2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 * @ingroup grid
 */
template< class container>
struct CurvilinearGridX3d : public dg::GridX3d
{
    typedef dg::CurvilinearCylindricalTag metric_category;
    typedef CurvilinearGridX2d<container> perpendicular_grid;

    template< class Generator>
    CurvilinearGridX3d( Generator generator, double psi_0, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, dg::bc bcy):
        dg::GridX3d( 0,1, -2.*M_PI*fy/(1.-2.*fy), 2.*M_PI*(1.+fy/(1.-2.*fy)), 0., 2*M_PI, fx, fy, n, Nx, Ny, Nz, bcx, bcy, dg::PER),
        r_(this->size()), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_)
    {
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
    perpendicular_grid perp_grid() const { return CurvilinearGridX2d<container>(*this);}
    private:
    template<class Generator>
    void construct( Generator generator, double psi_0, double fx, unsigned n, unsigned Nx, unsigned Ny )
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
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 * @ingroup grid
 */
template< class container>
struct CurvilinearGridX2d : public dg::GridX2d
{
    typedef dg::CurvilinearTag metric_category;
    template<class Generator>
    CurvilinearGridX2d(Generator generator, double psi_0, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy):
        dg::GridX2d( 0, 1,-fy*2.*M_PI/(1.-2.*fy), 2*M_PI+fy*2.*M_PI/(1.-2.*fy), fx, fy, n, Nx, Ny, bcx, bcy)
    {
        CurvilinearGridX3d<container> g( generator, psi_0, fx,fy, n,Nx,Ny,1,bcx,bcy);
        init_X_boundaries( g.x0(),g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    CurvilinearGridX2d( const CurvilinearGridX3d<container>& g):
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
