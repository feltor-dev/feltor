#include <iostream>
#include <iomanip>
#include <functional>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "backend/typedefs.h"
#include "topology/evaluation.h"
#include "runge_kutta.h"

#include "catch2/catch_all.hpp"



inline std::array<double, 2> solution( double t, double damping, double omega_0, double omega_drive)
{
    double tmp1 = (2.*omega_0*damping);
    double tmp2 = (omega_0*omega_0 - omega_drive*omega_drive)/omega_drive;
    double amp = 1./sqrt( tmp1*tmp1 + tmp2*tmp2);
    double phi = atan( 2.*omega_drive*omega_0*damping/(omega_drive*omega_drive-omega_0*omega_0));

    double x = amp*sin(omega_drive*t+phi)/omega_drive;
    double v = amp*cos(omega_drive*t+phi);
    return {x,v};
}

TEST_CASE( "is_same and is_divisable")
{
    CHECK(  dg::is_same ( 3.3, 3.3+1e-16) );
    CHECK( !dg::is_same ( 3.3, 4.0) );
    CHECK(  dg::is_same ( (float)3, (float)(3+1e-7)) );
    CHECK( !dg::is_same ( (float)3, (float)4) );
    CHECK(  dg::is_same ( 3, 3) );
    CHECK(  dg::is_divisable ( (float)12, (float)3) );
    CHECK( !dg::is_divisable ( (float)12, (float)5) );
    CHECK(  dg::is_divisable ( (double)12, (double)3) );
    CHECK( !dg::is_divisable ( (double)12, (double)3.1) );
}

TEST_CASE( "Runge Kutta")
{
    INFO( "Test correct implementation of Runge Kutta methods"
        <<" at the example of the damped driven harmonic oscillator");
    //![doxygen]
    //... in main
    //set start and end time, number of steps and timestep
    const double t_start = 0., t_end = 1.;
    const unsigned N = 20;
    const double dt = (t_end - t_start)/(double)N;
    //set physical parameters and initial condition
    const double damping = 0.2, omega_0 = 1.0, omega_drive = 0.9;
    std::array<double,2> u = solution(t_start, damping, omega_0, omega_drive);
    //construct a functor with the right interface
    //![function]
    auto rhs = [=]( double t, const std::array<double,2>& y,
            std::array<double,2>& yp)
    {
        //damped driven harmonic oscillator
        // x -> y[0] , v -> y[1]
        yp[0] = y[1];
        yp[1] = -2.*damping*omega_0*y[1] - omega_0*omega_0*y[0]
                + sin(omega_drive*t);
    };
    auto solve = [=]( double alpha, double t, std::array<double,2>& y,
            const std::array<double,2>& yp)
    {
        // y - alpha RHS( t, y) = rho
        // can be solved analytically
        y[1] = ( yp[1] + alpha*sin(omega_drive*t) - alpha*omega_0*omega_0*yp[0])/
               (1.+2.*alpha*damping*omega_0+alpha*alpha*omega_0*omega_0);
        y[0] = yp[0] + alpha*y[1];
    };
    SECTION( "Runge-Kutta-4-4")
    {
        //![function]
        //construct Runge Kutta class
        dg::RungeKutta<std::array<double,2> >  rk( "Runge-Kutta-4-4", u);
        //integration loop
        double t=t_start;
        for( unsigned i=0; i<N; i++)
            rk.step( rhs, t, u, t, u, dt); //step inplace
        //now compute error
        dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u);
        double err = sqrt( dg::blas1::dot( u,u));
        INFO( "Norm of error is "<<err);
        CHECK( err < 1e-7);
    }
    //![doxygen]
    SECTION( "Explicit Methods ")
    {
        auto name = GENERATE( as<std::string>{},
            // in order to test embedded method temporarily return bt in ButcherTableau
            "Euler",
            "Midpoint-2-2",
            "Kutta-3-3",
            "Runge-Kutta-4-4",
            "SSPRK-2-2",
            "SSPRK-3-2",
            "SSPRK-3-3",
            "SSPRK-5-3",
            "SSPRK-5-4",
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
        auto b = dg::create::tableau<double>(name);
        std::vector<unsigned> NTs = {4,8};
        if( b.order() == 1)
            NTs = {40, 80};
        if( b.order() > 5)
            NTs = {2,4};
        std::vector<double> err(NTs.size());
        for( unsigned k = 0; k<NTs.size(); k++)
        {
            unsigned N = NTs[k];
            u = solution(t_start, damping, omega_0, omega_drive);
            std::array<double, 2> u1(u), sol = solution(t_end, damping, omega_0,
                    omega_drive);
            dg::SinglestepTimeloop<std::array<double,2>>(
                    dg::RungeKutta<std::array<double,2>>( name, u), rhs
                    ).integrate_steps( t_start, u, t_end, u1, N);
            dg::blas1::axpby( 1., sol , -1., u1);
            err[k] = sqrt(dg::blas1::dot( u1, u1));
        }
        double order = log( err[0]/err[1])/log( (double)NTs[1]/(double)NTs[0]);
        INFO("Norm of error in "<<std::setw(24) <<name<<"\t"<<err[1]<<" order "
            <<order<<" expected "<<b.order()<<" "<<(b.isFsal()?" (fsal)" : ""));
        // The order is allowed to be larger than expected...
        // So that Feagin-17-8-10, Verner-9-5-6 and Tsitouras11-7-4-5 pass ...
        CHECK( (b.order()  - order)/double(b.order()) < 0.014);
    }
    SECTION( "Shu-Osher Methods")
    {
        auto name = GENERATE( as<std::string>{},
            "SSPRK-2-2",
            "SSPRK-3-2",
            "SSPRK-3-3",
            "SSPRK-5-3",
            "SSPRK-5-4"
        );
        auto b = dg::create::tableau<double>(name);
        std::vector<unsigned> NTs = {4,8};
        if( b.order() > 5)
            NTs = {2,4};
        std::vector<double> err(NTs.size());
        for( unsigned k = 0; k<NTs.size(); k++)
        {
            unsigned N = NTs[k];
            u = solution(t_start, damping, omega_0, omega_drive);
            std::array<double, 2> u1(u), sol = solution(t_end, damping, omega_0, omega_drive);
            dg::ShuOsher<std::array<double,2>> rk( name, u);
            dg::IdentityFilter id;
            const double dt = (t_end-t_start)/(double)N;
            dg::SinglestepTimeloop<std::array<double,2>>( rk, std::tie(rhs,id),
                    dt).integrate( t_start, u, t_end, u1);
            dg::blas1::axpby( 1., sol , -1., u1);
            err[k] = sqrt(dg::blas1::dot( u1, u1));
        }
        double order = log( err[0]/err[1])/log( (double)NTs[1]/(double)NTs[0]);
        INFO("Norm of error in "<<std::setw(24) <<name<<"\t"<<err[1]<<" order "
            <<order<<" expected "<<b.order());
        CHECK( fabs( (order  - b.order())/double(b.order())) < 0.012);
    }
    SECTION("Implicit Methods")
    {
    ///-------------------------------Implicit Methods----------------------//
        auto name = GENERATE( as<std::string>{},
            "Euler (implicit)",
            "Midpoint (implicit)",
            "Trapezoidal-2-2",
            "SDIRK-2-1-2",
            "Cavaglieri-3-1-2 (implicit)",
            "Sanchez-3-3",
            "Kvaerno-4-2-3",
            "SDIRK-4-2-3",
            "Cavaglieri-4-2-3 (implicit)",
            "ARK-4-2-3 (implicit)",
            "Sanchez-3-4",
            "Cash-5-2-4",
            "Cash-5-3-4",
            "SDIRK-5-3-4",
            "ARK-6-3-4 (implicit)",
            "Sanchez-6-5",
            "Kvaerno-7-4-5",
            "ARK-8-4-5 (implicit)",
            "Sanchez-7-6"
        );
        auto b = dg::create::tableau<double>(name);
        std::vector<unsigned> NTs = {15,20};
        if( b.order() == 1)
            NTs = {40, 80};
        if( b.order() > 5)
            NTs = {4,5};
        std::vector<double> err(NTs.size());
        for( unsigned k = 0; k<NTs.size(); k++)
        {
            const double dt_im = (t_end - t_start)/(double)NTs[k];
            u = solution(t_start, damping, omega_0, omega_drive);
            std::array<double, 2> u1(u), sol = solution(t_end, damping, omega_0,
                    omega_drive);
            dg::ImplicitRungeKutta<std::array<double,2>> irk( name, u);
            dg::SinglestepTimeloop<std::array<double,2>> odeint( irk,
                    std::tie(rhs,solve), dt_im);
            // Test integrate at least
            unsigned maxout = 3;
            double deltaT = (t_end-t_start)/(double)maxout;
            double time = t_start;
            for( unsigned u=1; u<=maxout; u++)
            {
                odeint.integrate( time, u1, t_start + u*deltaT, u1,
                    u<maxout ? dg::to::at_least :  dg::to::exact);
            }
            dg::blas1::axpby( 1., sol , -1., u1);
            err[k] = sqrt(dg::blas1::dot( u1, u1));
        }
        double order = log( err[0]/err[1])/log( (double)NTs[1]/(double)NTs[0]);
        INFO("Norm of error in "<<std::setw(24) <<name<<"\t"<<err[1]<<" order "
            <<order<<" expected "<<b.order());
        CHECK( fabs( (order  - b.order())/double(b.order())) < 0.012);
    }
}
