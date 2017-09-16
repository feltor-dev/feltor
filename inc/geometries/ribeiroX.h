#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/gridX.h"
#include "dg/backend/evaluationX.cuh"
#include "dg/backend/weightsX.cuh"
#include "dg/runge_kutta.h"
#include "dg/geometry/generatorX.h"
#include "utilitiesX.h"
#include "ribeiro.h"

//! ATTENTION: algorithm might be flawed since f(psi) might make a jump at
//! ATTENTION: separatrix

namespace dg
{
namespace geo
{
///@cond
namespace ribeiro
{
namespace detail
{
//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as 
//good as it can, i.e. until machine precision is reached
struct FpsiX
{
    FpsiX( const BinaryFunctorsLvl1& psi, double xX, double yX, double x0, double y0): 
        initX_(psi, xX, yX), fieldRZYT_(psi, x0, y0), fieldRZYZ_(psi)
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
            if( std::isnan(eps)) { eps = eps_old/2.; end = end_old; 
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
            if( std::isnan(eps)) { std::cerr << "Attention!!\n"; eps = eps_old -1e-15; x0 = x0_old;} //near X-point integration can go wrong
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
    dg::geo::orthogonal::detail::InitialX initX_;
    const dg::geo::ribeiro::FieldRZYT fieldRZYT_;
    const dg::geo::ribeiro::FieldRZYZ fieldRZYZ_;

};

//This struct computes -2pi/f with a fixed number of steps for all psi
struct XFieldFinv
{
    XFieldFinv( const BinaryFunctorsLvl1& psi, double xX, double yX, double x0, double y0, unsigned N_steps = 500): 
        fpsi_(psi, xX, yX, x0, y0), fieldRZYT_(psi, x0, y0), fieldRZYZ_(psi) , N_steps(N_steps)
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
    FpsiX fpsi_;
    dg::geo::ribeiro::FieldRZYT fieldRZYT_;
    dg::geo::ribeiro::FieldRZYZ fieldRZYZ_;
    thrust::host_vector<double> fpsi_neg_inv;
    unsigned N_steps;
    double xAtOne_;
};
} //namespace detail
}//namespace ribeiro
///@endcond

/**
 * @brief A two-dimensional grid based on "almost-conformal" coordinates by %Ribeiro and Scott 2010 
 * @ingroup generators_geo
 * @tparam Psi All the template parameters must model aBinaryOperator i.e. the bracket operator() must be callable with two arguments and return a double. 
 */
struct RibeiroX : public aGeneratorX2d
{
    RibeiroX( const BinaryFunctorsLvl2& psi, double psi_0, double fx, 
            double xX, double yX, double x0, double y0):
        psi_(psi), fpsi_(psi, xX, yX, x0,y0), fpsiMinv_(psi, xX, yX, x0,y0, 500)
    {
        assert( psi_0 < 0 );
        zeta0_ = fpsi_.find_x( psi_0);
        f0_=zeta0_/psi_0;
        zeta1_= -fx/(1.-fx)*zeta0_;
        x0_=x0, y0_=y0, psi0_=psi_0;
    }
    private:
    bool isConformal()const{return false;}
    bool do_isOrthogonal()const{return false;}
    double f0() const{return f0_;}
    /**
     * @brief The vector f(x)
     *
     * @return f(x)
     */
    thrust::host_vector<double> fx() const{ return fx_;}
    double psi1() const {return psi_1_numerical_;}
    void do_generate( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         unsigned nodeX0, unsigned nodeX1, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) const
    {
        //compute psi(x) for a grid on x and call construct_rzy for all psi
        unsigned inside=0;
        for(unsigned i=0; i<zeta1d.size(); i++)
            if( zeta1d[i]< 0) inside++;//how many points are inside
        thrust::host_vector<double> psi_x;
        psi_1_numerical_ = dg::geo::detail::construct_psi_values( fpsiMinv_, psi0_, zeta0_, zeta1d, zeta1_, inside, psi_x);

        //std::cout << "In grid function:\n";
        dg::geo::ribeiro::FieldRZYRYZY fieldRZYRYZYribeiro(psi_);
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
            dg::geo::detail::computeX_rzy( fpsi_, fieldRZYRYZYribeiro, psi_x[i], eta1d, nodeX0, nodeX1, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i]);
            for( unsigned j=0; j<Ny; j++)
            {
                x[j*Nx+i]  = ry[j], y[j*Nx+i]  = zy[j];
                etaX[j*Nx+i] = yr[j], etaY[j*Nx+i] = yz[j];
                zetaX[j*Nx+i] = xr[j], zetaY[j*Nx+i] = xz[j];
            }
        }
    }

    virtual double do_zeta0(double fx) const { return zeta0_; }
    virtual double do_zeta1(double fx) const { return zeta1_;}
    virtual double do_eta0(double fy) const { return -2.*M_PI*fy/(1.-2.*fy); }
    virtual double do_eta1(double fy) const { return 2.*M_PI*(1.+fy/(1.-2.*fy));}
    private:
    BinaryFunctorsLvl2 psi_;
    dg::geo::ribeiro::detail::XFieldFinv fpsiMinv_; 
    dg::geo::ribeiro::detail::FpsiX fpsi_;
    double f0_, psi_1_numerical_;
    thrust::host_vector<double> fx_;
    double zeta0_, zeta1_;
    double lx_, x0_, y0_, psi0_, psi1_;
    int mode_; //0 = ribeiro, 1 = equalarc
};


}//namespace geo
}//namespace dg
