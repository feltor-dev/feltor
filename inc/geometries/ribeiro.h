#pragma once

#include "dg/algorithm.h"
#include "generator.h"
#include "utilities.h"



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
struct Fpsi
{
    Fpsi( const CylindricalFunctorsLvl1& psi, double x0, double y0, int mode, bool verbose = false):
        psip_(psi), fieldRZYTribeiro_(psi,x0, y0),fieldRZYTequalarc_(psi, x0, y0), fieldRZtau_(psi), mode_(mode), m_verbose(verbose)
    {
        R_init = x0; Z_init = y0;
        while( fabs( psi.dfx()(R_init, Z_init)) <= 1e-10 && fabs( psi.dfy()( R_init, Z_init)) <= 1e-10)
        {
            R_init = x0 + 1.;
            Z_init = y0;
        }
    }
    //finds the starting points for the integration in y direction
    void find_initial( double psi, double& R_0, double& Z_0)
    {
        unsigned N = 50;
        std::array<double, 2> begin2d{ {0,0} }, end2d(begin2d), end2d_old(begin2d);
        begin2d[0] = end2d[0] = end2d_old[0] = R_init;
        begin2d[1] = end2d[1] = end2d_old[1] = Z_init;
        if(m_verbose)std::cout << "In init function\n";
        double eps = 1e10, eps_old = 2e10;
        using Vec = std::array<double,2>;
        dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                    begin2d), fieldRZtau_);
        while( (eps < eps_old || eps > 1e-7) && eps > 1e-14)
        {
            eps_old = eps; end2d_old = end2d;
            N*=2; odeint.integrate_steps( psip_.f()(R_init, Z_init), begin2d,
                    psi, end2d, N);
            eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) +
                    (end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
        }
        R_init = R_0 = end2d_old[0], Z_init = Z_0 = end2d_old[1];
        if(m_verbose)std::cout << "In init function error: psi(R,Z)-psi0: "<<psip_.f()(R_init, Z_init)-psi<<"\n";
    }

    //compute f for a given psi between psi0 and psi1
    double construct_f( double psi, double& R_0, double& Z_0)
    {
        find_initial( psi, R_0, Z_0);
        std::array<double, 3> begin{ {0,0,0} }, end(begin), end_old(begin);
        begin[0] = R_0, begin[1] = Z_0;
        if(m_verbose)std::cout << begin[0]<<" "<<begin[1]<<" "<<begin[2]<<"\n";
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 50;
        //double y_eps = 1;
        using Vec = std::array<double,3>;
        dg::SinglestepTimeloop<Vec> odeint;
        if( mode_ == 0)
            odeint.construct( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                    begin), fieldRZYTribeiro_);
        if( mode_ == 1)
            odeint.construct( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                    begin), fieldRZYTequalarc_);
        while( (eps < eps_old || eps > 1e-7)&& N < 1e6)
        {
            eps_old = eps, end_old = end; N*=2;
            odeint.integrate_steps(  0., begin, 2*M_PI, end, N);
            eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) +
                    (end[1]-begin[1])*(end[1]-begin[1]));
        }
        if(m_verbose)std::cout << "\t error "<<eps<<" with "<<N<<" steps\t";
        if(m_verbose)std::cout <<end_old[2] << " "<<end[2] <<"\n";
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
        if(m_verbose)std::cout << "In x1 function\n";
        while(eps < eps_old && P < 20 && eps > 1e-15)
        {
            eps_old = eps;
            x1_old = x1;

            P+=1;
            if( psi_1 < psi_0) std::swap(psi_0, psi_1);
            dg::Grid1d grid( psi_0, psi_1, P, 1);
            thrust::host_vector<double> psi_vec = dg::evaluate( dg::cooX1d, grid);
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
        double fprime = (-0.5*fofpsi[1]+0.5*fofpsi[2])/deltaPsi, fprime_old = fprime;
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
    CylindricalFunctorsLvl1 psip_;
    dg::geo::ribeiro::FieldRZYT fieldRZYTribeiro_;
    dg::geo::equalarc::FieldRZYT fieldRZYTequalarc_;
    dg::geo::FieldRZtau fieldRZtau_;
    int mode_;
    bool m_verbose;
};

//This struct computes -2pi/f with a fixed number of steps for all psi
struct FieldFinv
{
    FieldFinv( const CylindricalFunctorsLvl1& psi, double x0, double y0, unsigned N_steps, int mode):
        fpsi_(psi, x0, y0, mode), fieldRZYTribeiro_(psi, x0, y0), fieldRZYTequalarc_(psi, x0, y0), N_steps(N_steps), mode_(mode) { }
    void operator()(double, double psi, double& fpsiM)
    {
        std::array<double, 3> begin{ {0,0,0} }, end(begin);
        fpsi_.find_initial( psi, begin[0], begin[1]);
        using Vec = std::array<double,3>;
        dg::SinglestepTimeloop<Vec> odeint;
        if( mode_ == 0)
            odeint.construct( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                    begin), fieldRZYTribeiro_);
        if( mode_ == 1)
            odeint.construct( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                    begin), fieldRZYTequalarc_);
        odeint.integrate_steps(  0., begin, 2*M_PI, end, N_steps);
        fpsiM = end[2]/2./M_PI;
        //std::cout <<"fpsiMinverse is "<<fpsiM[0]<<" "<<-1./fpsi_(psi[0])<<" "<<eps<<"\n";
    }
    private:
    Fpsi fpsi_;
    dg::geo::ribeiro::FieldRZYT fieldRZYTribeiro_;
    dg::geo::equalarc::FieldRZYT fieldRZYTequalarc_;
    unsigned N_steps;
    int mode_;
};
} //namespace detail
}//namespace ribeiro
///@endcond

/**
 * @brief A two-dimensional grid based on "almost-conformal" coordinates by %Ribeiro and Scott 2010
 *
 * The radial coordinate is given by \f$ \zeta = -\int_{\psi_0}^{\psi} f(\psi') d\psi'\f$
 *
 * The poloidal coordinate lines are given by
 * \f{align}{
 * \frac{d R}{d \eta} &=  \frac{ 1}{f(\psi_p)(\nabla\psi_p)^2} \frac{\partial \psi_p}{\partial Z} \\
 * \frac{d Z}{d \eta} &= -\frac{ 1}{f(\psi_p)(\nabla\psi_p)^2} \frac{\partial \psi_p}{\partial R} \f},
 * where \f$ f(\psi_p)\f$ is the normalisation constant
 * that makes the poloidal \f$ \eta\f$ coordinate go from 0 to \f$2\pi\f$.
 *
 * When an "equalarc" adaption is chosen then the integration is changed to
 * \f{align}{
 * \frac{d R}{d \eta} &=  \frac{ 1}{f(\psi_p)|\nabla\psi_p|} \frac{\partial \psi_p}{\partial Z} \\
 * \frac{d Z}{d \eta} &= -\frac{ 1}{f(\psi_p)|\nabla\psi_p|} \frac{\partial \psi_p}{\partial R} \f},
 * @ingroup generators_geo
 * @snippet flux_b.cpp doxygen
 */
struct Ribeiro : public aGenerator2d
{
    /**
     * @brief Construct a near-conformal grid generator
     *
     * @param psi \f$ \psi(x,y)\f$ the flux function and its derivatives in Cartesian coordinates (x,y)
     * @param psi_0 first boundary
     * @param psi_1 second boundary
     * @param x0 a point in the inside of the domain bounded by \c psi_0 (shouldn't be the O-point)
     * @param y0 a point in the inside of the domain bounded by \c psi_0 (shouldn't be the O-point)
     * @param mode This parameter indicates the adaption type used to create the grid: 0 is no adaption, 1 is an equalarc adaption
     * @param verbose if true the integrators will write additional information to \c std::cout
     */
    Ribeiro( const CylindricalFunctorsLvl2& psi, double psi_0, double psi_1, double x0, double y0, int mode = 0, bool verbose = false):
        psi_(psi), mode_(mode), m_verbose(verbose)
    {
        assert( psi_1 != psi_0);
        ribeiro::detail::Fpsi fpsi(psi, x0, y0, mode);
        lx_ = fabs(fpsi.find_x1( psi_0, psi_1));
        x0_=x0, y0_=y0, psi0_=psi_0, psi1_=psi_1;
        if(m_verbose)std::cout << "lx = "<<lx_<<"\n";
    }
    virtual Ribeiro* clone() const override final{return new Ribeiro(*this);}

    private:
    /**
     * @brief The length of the zeta-domain
     *
     * Call before discretizing the zeta domain
     * @return length of zeta-domain (f0*(psi_1-psi_0))
     * @note the length is always positive
     */
    virtual double do_width() const override final{return lx_;}
    /**
     * @brief 2pi (length of the eta domain)
     *
     * Always returns 2pi
     * @return 2pi
     */
    virtual double do_height() const override final{return 2.*M_PI;}
    virtual void do_generate(
         const thrust::host_vector<double>& zeta1d,
         const thrust::host_vector<double>& eta1d,
         thrust::host_vector<double>& x,
         thrust::host_vector<double>& y,
         thrust::host_vector<double>& zetaX,
         thrust::host_vector<double>& zetaY,
         thrust::host_vector<double>& etaX,
         thrust::host_vector<double>& etaY) const override final
    {
        //compute psi(x) for a grid on x and call construct_rzy for all psi
        ribeiro::detail::FieldFinv fpsiMinv_(psi_, x0_,y0_, 500, mode_);
        thrust::host_vector<double> psi_x, fx_;
        dg::geo::detail::construct_psi_values( fpsiMinv_, psi0_, psi1_, 0., zeta1d, lx_, psi_x, fx_);

        if(m_verbose)std::cout << "In grid function:\n";
        ribeiro::detail::Fpsi fpsi(psi_, x0_, y0_, mode_, m_verbose);
        dg::geo::ribeiro::FieldRZYRYZY fieldRZYRYZYribeiro(psi_);
        dg::geo::equalarc::FieldRZYRYZY fieldRZYRYZYequalarc(psi_);
        thrust::host_vector<double> f_p(fx_);
        unsigned Nx = zeta1d.size(), Ny = eta1d.size();
        for( unsigned i=0; i<zeta1d.size(); i++)
        {
            thrust::host_vector<double> ry, zy;
            thrust::host_vector<double> yr, yz, xr, xz;
            double R0, Z0;
            if(mode_==0)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZYribeiro, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i], m_verbose);
            if(mode_==1)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZYequalarc, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i], m_verbose);
            for( unsigned j=0; j<Ny; j++)
            {
                x[j*Nx+i]  = ry[j], y[j*Nx+i]  = zy[j];
                etaX[j*Nx+i] = yr[j], etaY[j*Nx+i] = yz[j];
                zetaX[j*Nx+i] = xr[j], zetaY[j*Nx+i] = xz[j];
            }
        }
    }
    CylindricalFunctorsLvl2 psi_;
    double lx_, x0_, y0_, psi0_, psi1_;
    int mode_; //0 = ribeiro, 1 = equalarc
    bool m_verbose;
};

} //namespace geo
} //namespace dg
