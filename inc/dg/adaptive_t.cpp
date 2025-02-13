#include <iostream>
#include <iomanip>
#include <functional>
#include <array>

#include "backend/typedefs.h"
#include "topology/evaluation.h"
#include "topology/grid.h"
#include "arakawa.h"
#include "runge_kutta.h"
#include "adaptive.h"

#include "catch2/catch_all.hpp"

//![function]
struct RHS
{
    RHS( double damping, double omega_0, double omega_drive) :
        m_d( damping), m_w0( omega_0), m_wd( omega_drive){}
    void operator()( double t, const std::array<double,2>& y,
            std::array<double,2>& yp)
    {
        //damped driven harmonic oscillator
        // x -> y[0] , v -> y[1]
        yp[0] = y[1];
        yp[1] = -2.*m_d*m_w0*y[1] - m_w0*m_w0*y[0] + sin(m_wd*t);
    }
    void operator()( double alpha, double t, std::array<double,2>& y,
            const std::array<double,2>& yp)
    {
        // y - alpha RHS( t, y) = rho
        // can be solved analytically
        y[1] = ( yp[1] + alpha*sin(m_wd*t) - alpha* m_w0*m_w0*yp[0])/
               (1.+2.*alpha*m_d*m_w0+alpha*alpha*m_w0*m_w0);
        y[0] = yp[0] + alpha*y[1];
    }
    private:
    double m_d, m_w0, m_wd;
};
//![function]

inline std::array<double, 2> solution( double t, double damping, double omega_0,
        double omega_drive)
{
    double tmp1 = (2.*omega_0*damping);
    double tmp2 = (omega_0*omega_0 - omega_drive*omega_drive)/omega_drive;
    double amp = 1./sqrt( tmp1*tmp1 + tmp2*tmp2);
    double phi = atan( 2.*omega_drive*omega_0*damping/
            (omega_drive*omega_drive-omega_0*omega_0));

    double x = amp*sin(omega_drive*t+phi)/omega_drive;
    double v = amp*cos(omega_drive*t+phi);
    return {x,v};
}

TEST_CASE( "Adaptive")
{
    INFO("Test correct implementation of adaptive methods "
        <<"at the example of the damped driven harmonic oscillator");
    std::cout << std::scientific;
    //![doxygen]
    //... in main
    //set start and end time
    double t_start = 0., t_end = 1.;
    //set physical parameters and initial condition
    const double damping = 0.2, omega_0 = 1.0, omega_drive = 0.9;
    std::array<double,2> u_start = solution(t_start, damping, omega_0,
            omega_drive), u_end(u_start);
    //construct a rhs with the right interface
    RHS rhs( damping, omega_0, omega_drive);
    //integration
    using Vec = std::array<double,2>;
    SECTION( "Dormand-Prince-7-4-5")
    {
        dg::Adaptive<dg::ERKStep<Vec>> adapt( "Dormand-Prince-7-4-5", u_start);
        dg::AdaptiveTimeloop<Vec>( adapt, rhs, dg::pid_control, dg::fast_l2norm,
                1e-6, 1e-10).integrate( t_start, u_start, t_end, u_end);
        //now compute error
        dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1.,
                u_end);
        double err = dg::fast_l2norm(u_end);
        INFO("With "<<adapt.nsteps()<<"\t Dormand Prince steps norm of error is "
                  << err);
        CHECK( err < 1e-6);
    }
    //![doxygen]
    SECTION("Explicit Methods")
    {
        auto name = GENERATE( as<std::string>{},
            "Heun-Euler-2-1-2",
            "Cavaglieri-3-1-2 (explicit)",
            "Fehlberg-3-2-3",
            "Bogacki-Shampine-4-2-3",
            "Cavaglieri-4-2-3 (explicit)",
            "ARK-4-2-3 (explicit)",
            "Zonneveld-5-3-4",
            "ARK-6-3-4 (explicit)",
            "Sayfy-Aburub-6-3-4",
            "Cash-Karp-6-4-5",
            "Fehlberg-6-4-5",
            "Dormand-Prince-7-4-5",
            "Tsitouras09-7-4-5",
            "Tsitouras11-7-4-5",
            "ARK-8-4-5 (explicit)",
            "Verner-9-5-6",
            "Verner-10-6-7",
            "Fehlberg-13-7-8",
            "Dormand-Prince-13-7-8",
            "Feagin-17-8-10"
        );
        SECTION( "Integrate")
        {
            u_start = solution(t_start, damping, omega_0, omega_drive);
            dg::Adaptive<dg::ERKStep<Vec>> adapt( name, u_start);
            dg::AdaptiveTimeloop<Vec>( adapt, rhs, dg::pid_control, dg::fast_l2norm,
                    1e-6, 1e-10).integrate( t_start, u_start, t_end, u_end);

            std::array<double, 2> sol = solution(t_end, damping, omega_0, omega_drive);
            dg::blas1::axpby( 1.,sol  , -1., u_end);
            double err = dg::fast_l2norm( u_end);

            INFO( "With "<<std::setw(6)<<adapt.nsteps()
                      <<" steps norm of error in "
                      <<std::setw(24)<<name<<"\t"<<err);
            CHECK( err < 1e-5);
        }
        SECTION( "Domain restriction")
        {
            double dt = 0;
            double t_start = 0;
            double t_end = 10;
            double u_start = 1.0, u_end;
            auto rhs = [](double t, double y, double& yp){
                    yp = y;
            };
            dg::Adaptive<dg::ERKStep<double>> adapt( name,u_start);
            dg::AdaptiveTimeloop<double> odeint( adapt,
                    rhs, dg::pid_control, dg::fast_l2norm, 1e-6, 1e-10);
            odeint.integrate_in_domain( t_start, u_start, t_end, u_end, dt,
                    dg::Grid1d( 0., 100., 1,1), 1e-6  );
            double analytic = log( 100.);
            double err = fabs( t_end - analytic);
            INFO( "With "<<std::setw(6)<<adapt.nsteps()
                      <<" steps norm of error in "
                      <<std::setw(24)<<name<<"\t"<<err);
            CHECK( err < 1e-4);
        }
    }
    ///-------------------------------Implicit Methods----------------------//
    SECTION( "Implicit Methods")
    {
        auto name = GENERATE( as<std::string>{},
            "SDIRK-2-1-2",
            "Cavaglieri-3-1-2 (implicit)",
            "SDIRK-4-2-3",
            "Kvaerno-4-2-3",
            "Cavaglieri-4-2-3 (implicit)",
            "ARK-4-2-3 (implicit)",
            "Cash-5-2-4",
            "Cash-5-3-4",
            "SDIRK-5-3-4",
            "ARK-6-3-4 (implicit)",
            "Kvaerno-7-4-5",
            "ARK-8-4-5 (implicit)"
        );
        u_start = solution(t_start, damping, omega_0, omega_drive);
        dg::Adaptive< dg::DIRKStep< std::array<double,2> > >
                adapt( name, u_start);

        dg::AdaptiveTimeloop<std::array<double,2>> odeint(
                    adapt, std::tie(rhs,rhs), dg::pid_control, dg::fast_l2norm,
                    1e-6, 1e-10);
        // Test integrate at least
        dg::blas1::copy( u_start, u_end);
        unsigned maxout = 10; //force it to make smaller steps than it wants
        double deltaT = (t_end-t_start)/(double)maxout;
        double time = t_start;
        for( unsigned u=1; u<=maxout; u++)
        {
            odeint.integrate( time, u_end, t_start + u*deltaT, u_end,
                u<maxout ? dg::to::at_least :  dg::to::exact);
        }
        //odeint.integrate( t_start, u_start, t_end, u_end);

        std::array<double, 2> sol = solution(t_end, damping, omega_0, omega_drive);
        dg::blas1::axpby( 1., sol, -1., u_end);
        double err = dg::fast_l2norm( u_end);
        INFO( "With "<<std::setw(6)<<adapt.nsteps()
                  <<" steps norm of error in "
                  <<std::setw(24)<<name<<"\t"<<err);
        CHECK( err < 1e-5);
    }
}
