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
#include "ribeiro.h"


namespace dg
{
namespace geo
{

///@cond
namespace flux
{
namespace detail
{

//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as 
//good as it can, i.e. until machine precision is reached
template< class Psi, class PsiX, class PsiY, class Ipol>
struct Fpsi
{
    
    //firstline = 0 -> conformal, firstline = 1 -> equalarc
    Fpsi( Psi psi, PsiX psiX, PsiY psiY, Ipol ipol, double x0, double y0): 
        psip_(psi), fieldRZYT_(psiX, psiY, ipol, x0, y0), fieldRZtau_(psiX, psiY)
    {
        X_init = x0, Y_init = y0;
        while( fabs( psiX(X_init, Y_init)) <= 1e-10 && fabs( psiY( X_init, Y_init)) <= 1e-10)
            X_init +=  1.; 
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
            dg::stepperRK17( fieldRZYT_, begin, end, 0., 2*M_PI, N);
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
    double X_init, Y_init;
    Psi psip_;
    dg::geo::flux::FieldRZYT<PsiX, PsiY, Ipol> fieldRZYT_;
    dg::geo::FieldRZtau<PsiX, PsiY> fieldRZtau_;

};

} //namespace detail
}//namespace flux
///@endcond

/**
 * @brief A symmetry flux generator
 * @ingroup generators
 * @tparam Psi All the template parameters must model aBinaryOperator i.e. the bracket operator() must be callable with two arguments and return a double. 
 * @tparam PsiX models aBinaryOperator 
 * @tparam PsiY models aBinaryOperator 
 * @tparam PsiXX models aBinaryOperator 
 * @tparam PsiXY models aBinaryOperator 
 * @tparam PsiYY models aBinaryOperator 
 * @tparam Ipol models aBinaryOperator 
 * @tparam IpolX models aBinaryOperator 
 * @tparam IpolY models aBinaryOperator 
 */
template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY, class Ipol, class IpolX, class IpolY>
struct FluxGenerator
{
    /**
     * @brief Construct a symmetry flux grid generator
     *
     * @param psi \f$ \psi(x,y)\f$ the flux function in Cartesian coordinates (x,y)
     @param psiX \f$ \psi_x\f$ its derivative in x
     @param psiY \f$ \psi_y\f$ its derivative in y
     @param psiXX \f$ \psi_{xx}\f$ second derivative
     @param psiXY \f$ \psi_{xy}\f$ second derivative
     @param psiYY \f$ \psi_{yy}\f$ second derivative
     * @param ipol \f$ I(x,y)\f$ the current function in Cartesian coordinates (x,y)
     * @param ipolX \f$ I_x(x,y)\f$ its derivative in x
     * @param ipolY \f$ I_y(x,y)\f$ its derivative in x
     * @param psi_0 first boundary 
     * @param psi_1 second boundary
     * @param x0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param y0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param mode This parameter indicates the adaption type used to create the grid: 0 is no adaption, 1 is an equalarc adaption
     * @note If mode == 1 then this class does the same as the RibeiroFluxGenerator
     */
    FluxGenerator( Psi psi, PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, Ipol ipol, IpolX ipolX, IpolY ipolY, double psi_0, double psi_1, double x0, double y0, int mode=0):
        psi_(psi), psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY), ipol_(ipol), ipolR_(ipolX), ipolZ_(ipolY), mode_(mode)
    {
        psi0_ = psi_0, psi1_ = psi_1;
        assert( psi_1 != psi_0);
        if( mode==0)
        {
            flux::detail::Fpsi<Psi, PsiX, PsiY, Ipol> fpsi(psi, psiX, psiY, ipol, x0, y0);
            f0_ = fabs( fpsi.construct_f( psi_0, x0_, y0_));
        }
        else
        {
            ribeiro::detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi, psiX, psiY, x0, y0, mode);
            f0_ = fabs( fpsi.construct_f( psi_0, x0_, y0_));
        }
        if( psi_1 < psi_0) f0_*=-1;
        lx_ =  f0_*(psi_1-psi_0);
        x0_=x0, y0_=y0, psi0_=psi_0, psi1_=psi_1;
        //std::cout << "lx_ = "<<lx_<<"\n";
    }

    bool isOrthogonal()const{return false;}
    bool isConformal()const{return false;}
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
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) 
    {
        //compute psi(x) for a grid on x and call construct_rzy for all psi
        thrust::host_vector<double> psi_x(zeta1d);
        for( unsigned i=0; i<psi_x.size(); i++)
            psi_x[i] = zeta1d[i]/f0_ +psi0_;

        //std::cout << "In grid function:\n";
        flux::detail::Fpsi<Psi, PsiX, PsiY, Ipol> fpsi(psi_, psiX_, psiY_, ipol_, x0_, y0_);
        dg::geo::flux::FieldRZYRYZY<PsiX, PsiY, PsiXX, PsiXY, PsiYY, Ipol, IpolX, IpolY> fieldRZYRYZY(psiX_, psiY_, psiXX_, psiXY_, psiYY_, ipol_, ipolR_, ipolZ_);
        ribeiro::detail::Fpsi<Psi, PsiX, PsiY> fpsiRibeiro(psi_, psiX_, psiY_, x0_, y0_, mode_);
        dg::geo::equalarc::FieldRZYRYZY<PsiX, PsiY, PsiXX, PsiXY, PsiYY> fieldRZYRYZYequalarc(psiX_, psiY_, psiXX_, psiXY_, psiYY_);
        unsigned size = zeta1d.size()*eta1d.size();
        x.resize(size), y.resize(size);
        zetaX = zetaY = etaX = etaY =x ;
        fx_.resize( zeta1d.size());
        thrust::host_vector<double> f_p(fx_);
        unsigned Nx = zeta1d.size(), Ny = eta1d.size();
        for( unsigned i=0; i<zeta1d.size(); i++)
        {
            thrust::host_vector<double> ry, zy;
            thrust::host_vector<double> yr, yz, xr, xz;
            double R0, Z0;
            if(mode_==0)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZY, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i]);
            if(mode_==1)dg::geo::detail::compute_rzy( fpsiRibeiro, fieldRZYRYZYequalarc, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i]);
            for( unsigned j=0; j<Ny; j++)
            {
                x[j*Nx+i]  = ry[j], y[j*Nx+i]  = zy[j];
                etaX[j*Nx+i] = yr[j], etaY[j*Nx+i] = yz[j];
                zetaX[j*Nx+i] = xr[j]/fx_[i]*f0_, zetaY[j*Nx+i] = xz[j]/fx_[i]*f0_;
            }
        }
    }
    private:
    Psi psi_;
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
    Ipol ipol_;
    IpolX ipolR_;
    IpolY ipolZ_;
    thrust::host_vector<double> fx_;
    double f0_, lx_, x0_, y0_, psi0_, psi1_;
    int mode_;
};

/**
 * @brief Same as the Ribeiro class just but uses psi as a flux label directly
 * @ingroup generators
 * @tparam Psi All the template parameters must model aBinaryOperator i.e. the bracket operator() must be callable with two arguments and return a double. 
     * @tparam PsiX models aBinaryOperator 
     * @tparam PsiY models aBinaryOperator 
     * @tparam PsiXX models aBinaryOperator 
     * @tparam PsiXY models aBinaryOperator 
     * @tparam PsiYY models aBinaryOperator 
 */
template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
struct RibeiroFluxGenerator
{
    /**
     * @brief Construct a flux aligned grid generator
     *
     * @param psi \f$ \psi(x,y)\f$ the flux function in Cartesian coordinates (x,y)
     @param psiX \f$ \psi_x\f$ its derivative in x
     @param psiY \f$ \psi_y\f$ its derivative in y
     @param psiXX \f$ \psi_{xx}\f$ second derivative
     @param psiXY \f$ \psi_{xy}\f$ second derivative
     @param psiYY \f$ \psi_{yy}\f$ second derivative
     * @param psi_0 first boundary 
     * @param psi_1 second boundary
     * @param x0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param y0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param mode This parameter indicates the adaption type used to create the grid: 0 is no adaption, 1 is an equalarc adaption
     */
    RibeiroFluxGenerator( Psi psi, PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, double psi_0, double psi_1, double x0, double y0, int mode=0):
        psi_(psi), psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY), mode_(mode)
    {
        psi0_ = psi_0, psi1_ = psi_1;
        assert( psi_1 != psi_0);
        ribeiro::detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi, psiX, psiY, x0, y0, mode);
        f0_ = fabs( fpsi.construct_f( psi_0, x0_, y0_));
        if( psi_1 < psi_0) f0_*=-1;
        lx_ =  f0_*(psi_1-psi_0);
        x0_=x0, y0_=y0, psi0_=psi_0, psi1_=psi_1;
        //std::cout << "lx_ = "<<lx_<<"\n";
    }
    bool isOrthogonal()const{return false;}
    bool isConformal()const{return false;}
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
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) 
    {
        //compute psi(x) for a grid on x and call construct_rzy for all psi
        thrust::host_vector<double> psi_x(zeta1d);
        for( unsigned i=0; i<psi_x.size(); i++)
            psi_x[i] = zeta1d[i]/f0_ +psi0_;

        ribeiro::detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi_, psiX_, psiY_, x0_, y0_, mode_);
        dg::geo::ribeiro::FieldRZYRYZY<PsiX, PsiY, PsiXX, PsiXY, PsiYY> fieldRZYRYZYribeiro(psiX_, psiY_, psiXX_, psiXY_, psiYY_);
        dg::geo::equalarc::FieldRZYRYZY<PsiX, PsiY, PsiXX, PsiXY, PsiYY> fieldRZYRYZYequalarc(psiX_, psiY_, psiXX_, psiXY_, psiYY_);
        unsigned size = zeta1d.size()*eta1d.size();
        x.resize(size), y.resize(size);
        zetaX = zetaY = etaX = etaY =x ;
        fx_.resize( zeta1d.size());
        thrust::host_vector<double> f_p(fx_);
        unsigned Nx = zeta1d.size(), Ny = eta1d.size();
        for( unsigned i=0; i<zeta1d.size(); i++)
        {
            thrust::host_vector<double> ry, zy;
            thrust::host_vector<double> yr, yz, xr, xz;
            double R0, Z0;
            if(mode_==0)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZYribeiro, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i]);
            if(mode_==1)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZYequalarc, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i]);
            for( unsigned j=0; j<Ny; j++)
            {
                x[j*Nx+i]  = ry[j], y[j*Nx+i]  = zy[j];
                etaX[j*Nx+i] = yr[j], etaY[j*Nx+i] = yz[j];
                zetaX[j*Nx+i] = xr[j]/fx_[i]*f0_, zetaY[j*Nx+i] = xz[j]/fx_[i]*f0_;
            }
        }
    }
    private:
    Psi psi_;
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
    thrust::host_vector<double> fx_;
    double f0_, lx_, x0_, y0_, psi0_, psi1_;
    int mode_;
};
}//namespace geo
}//namespace dg
