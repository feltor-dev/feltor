#include <iostream>
#include <iomanip>
#include <functional>

#include "multistep.h"
#include "adaptive.h"

#include "catch2/catch_all.hpp"

//method of manufactured solution
inline std::array<double,2> solution( double t, double nu) {
    return {exp( -nu*t) + cos(t), exp( -nu*t) + sin(t)};
}

//![function]
struct Explicit
{
    Explicit( double nu): m_nu( nu) {}
    void operator()( double t, const std::array<double,2>& T,
            std::array<double,2>& Tp)
    {
        Tp[0] = m_nu*cos(t) - sin(t);
        Tp[1] = m_nu*sin(t) + cos(t);
    }
    private:
    double m_nu;
};

//the implicit part contains  Tp = -nu T
struct Implicit
{
    Implicit( double nu): m_nu(nu) { }

    void operator()( double t, const std::array<double,2>& T,
            std::array<double,2>& Tp)
    {
        Tp[0] = -m_nu*T[0];
        Tp[1] = -m_nu*T[1];
    }
    void operator()( double alpha, double t, std::array<double,2>& y, const
            std::array<double,2>& rhs)
    {
        // solve y - alpha I(t,y)  = rhs
        y[0] = rhs[0]/(1+alpha*m_nu);
        y[1] = rhs[1]/(1+alpha*m_nu);
    }
  private:
    double m_nu;
};
//![function]

struct FullImplicit
{
    FullImplicit( double nu): m_nu(nu){}
    void operator()( double t, const std::array<double,2>& T,
            std::array<double,2>& Tp)
    {
        Tp[0] = m_nu*cos(t) - sin(t) - m_nu*T[0];
        Tp[1] = m_nu*sin(t) + cos(t) - m_nu*T[1];
    }
    void operator()( double alpha, double t, std::array<double,2>& y, const
            std::array<double,2>& rhs)
    {
        y[0] = (rhs[0]+alpha*(m_nu*cos(t) - sin(t)))/(1+alpha*m_nu);
        y[1] = (rhs[1]+alpha*(m_nu*sin(t) + cos(t)))/(1+alpha*m_nu);
    }
  private:
    double m_nu;
};


//const unsigned NT = (unsigned)(nu*T*n*n*N*N/0.01/lx/lx);

TEST_CASE( "Multistep")
{
    INFO("Multistep and Semi-Implicit methods on a manufactured PDE");
    const double T = 1;
    const double nu = 0.01;
    //evaluate the initial condition
    const std::array<double,2> init( solution(0.,nu));
    std::array<double,2> y0(init);

    const std::array<double,2> sol = solution(T,nu);
    const double norm_sol = dg::blas1::dot( sol, sol);
    double time = 0.;
    Explicit ex( nu);
    Implicit im( nu);
    FullImplicit full( nu);
    SECTION( "Explicit Multistep methods")
    {
        auto name = GENERATE( as<std::string>{},
        "AB-1-1", "AB-2-2", "AB-3-3", "AB-4-4", "AB-5-5",
        "eBDF-1-1", "eBDF-2-2", "eBDF-3-3", "eBDF-4-4", "eBDF-5-5", "eBDF-6-6",
        "TVB-1-1", "TVB-2-2", "TVB-3-3", "TVB-4-4", "TVB-5-5", "TVB-6-6",
        "SSP-1-1", "SSP-2-2", "SSP-3-2", "SSP-4-2", "SSP-5-3", "SSP-6-3"
        );
        auto b = dg::create::lmstableau<double>(name);
        std::vector<unsigned> NTs = {30,40};
        if( b.order() > 5)
            NTs = {39,40};
        std::vector<double> err(NTs.size());
        for( unsigned k = 0; k<NTs.size(); k++)
        {
            unsigned N = NTs[k];
            double dt = T/ (double)N;
            time = 0., y0 = init;
            dg::ExplicitMultistep< std::array<double,2> > ab( name, y0);
            dg::MultistepTimeloop<std::array<double,2>>( ab, full, time, init,
                    dt).integrate( 0, init, T, y0);
            dg::blas1::axpby( -1., sol, 1., y0);
            err[k] = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        }
        double order = log( err[0]/err[1])/log( (double)NTs[1]/(double)NTs[0]);
        INFO( "Number of steps "<<NTs[0]<<" "<<NTs[1]);
        INFO("Norm of error in "<<std::setw(20) <<name<<"\t"<<err[1]<<" order "
            <<order<<" expected "<<b.order());
        CHECK( fabs( (order  - b.order())/double(b.order())) < 0.05);
    }
    SECTION( "Implicit multistep methods")
    {
        auto name = GENERATE( as<std::string>{},
            "Euler", "ImEx-Koto-2-2", "ImEx-Adams-2-2",
            "ImEx-BDF-2-2", "ImEx-BDF-3-3", "ImEx-BDF-4-4", "ImEx-BDF-5-5",
            "ImEx-BDF-6-6", "ImEx-TVB-3-3", "ImEx-TVB-4-4", "ImEx-TVB-5-5"
        );
        auto b = dg::create::lmstableau<double>(name);
        //std::vector<unsigned> NTs = {4,8};
        //if( b.order() == 1)
        //    NTs = {20, 40};
        //if( b.order() > 5)
        //    NTs = {2,4};
        std::vector<unsigned> NTs = {30,40};
        if( b.order() > 5)
            NTs = {39,40};
        std::vector<double> err(NTs.size());
        SECTION( "Implicit")
        {
            for( unsigned k = 0; k<NTs.size(); k++)
            {
                time = 0., y0 = init;
                unsigned NT = NTs[k];
                double dt = T/ (double)NT;
                dg::ImplicitMultistep< std::array<double,2>> bdf( name, y0);
                dg::MultistepTimeloop<std::array<double,2>> odeint( bdf,
                    std::tie( full, full), time, y0, dt);
                // Test integrate at least
                unsigned maxout = 3;
                double deltaT = T/(double)maxout;
                double time = 0;
                for( unsigned u=1; u<=maxout; u++)
                {
                    odeint.integrate( time, y0, 0 + u*deltaT, y0,
                        u<maxout ? dg::to::at_least :  dg::to::exact);
                }
                dg::blas1::axpby( -1., sol, 1., y0);
                err[k] = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
            }
        }
        SECTION( "ImEx multistep methods")
        {
            for( unsigned u = 0; u<NTs.size(); u++)
            {
                //![karniadakis]
                //construct time stepper
                dg::ImExMultistep< std::array<double,2> > imex( name, y0);
                time = 0., y0 = init; //y0 and init are of type std::array<double,2> and contain the initial condition
                //initialize the timestepper (ex and im are objects of type Explicit and Implicit defined above)
                unsigned NT = NTs[u];
                double dt = T/ (double)NT;
                imex.init( std::tie(ex, im, im), time, y0, dt);
                //main time loop (NT = 20)
                for( unsigned k=0; k<NT; k++)
                    imex.step( std::tie(ex, im, im), time, y0); //inplace step
                //![karniadakis]
                dg::blas1::axpby( -1., sol, 1., y0);
                err[u] = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
            }
        }
        double order = log( err[0]/err[1])/log( (double)NTs[1]/(double)NTs[0]);
        INFO("Norm of error in "<<std::setw(20) <<name<<"\t"<<err[1]
            <<" order " <<order<<" expected "<<b.order());
        CHECK( fabs( (order  - b.order())/double(b.order())) < 0.05);
    }
}

TEST_CASE( "ARK methods")
{
    const double T = 1;
    const double nu = 0.01;
    //evaluate the initial condition
    const std::array<double,2> init( solution(0.,nu));
    std::array<double,2> y0(init);

    const std::array<double,2> sol = solution(T,nu);
    const double norm_sol = dg::blas1::dot( sol, sol);
    double time = 0.;
    Explicit ex( nu);
    Implicit im( nu);
    FullImplicit full( nu);
    SECTION( "Semi-implicit ARK methods")
    {
        auto name = GENERATE( as<std::string>{},
            "Cavaglieri-3-1-2", "Cavaglieri-4-2-3", "ARK-4-2-3", "ARK-6-3-4",
            "ARK-8-4-5");
        auto b = dg::create::tableau<double>(name + " (explicit)");
        std::vector<unsigned> NTs = {4,8};
        if( b.order() == 1)
            NTs = {20, 40};
        if( b.order() > 5)
            NTs = {2,4};
        std::vector<double> err(NTs.size());
        for( unsigned u = 0; u<NTs.size(); u++)
        {
            dg::ARKStep< std::array<double,2> > imex( name, y0);
            std::array<double,2> delta{0,0};
            unsigned NT = NTs[u];
            double dt = T/ (double)NT;
            time = 0., y0 = init;
            //main time loop (NT = 20)
            for( unsigned k=0; k<NT; k++)
                imex.step( std::tie(ex, im, im), time, y0, time, y0, dt, delta ); //inplace step
            dg::blas1::axpby( -1., sol, 1., y0);
            err[u] = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        }
        double order = log( err[0]/err[1])/log( (double)NTs[1]/(double)NTs[0]);
        INFO("Norm of error in "<<std::setw(20) <<name<<"\t"<<err[1]
            <<" order " <<order<<" expected "<<b.order());
        // Allow order to be larger
        CHECK( (b.order() - order)/double(b.order()) < 0.012);
    }

    SECTION( "Adaptive semi-implicit ARK methods")
    {
        const double rtol = 1e-7, atol = 1e-10;
        auto name = GENERATE( as<std::string>{},
            "Cavaglieri-3-1-2", "Cavaglieri-4-2-3", "ARK-4-2-3", "ARK-6-3-4",
            "ARK-8-4-5");
        //![adaptive]
        time = 0., y0 = init;
        dg::Adaptive<dg::ARKStep<std::array<double,2>>> adapt( name, y0);
        double time = 0;
        double dt = 1e-6;
        int counter=0;
        while( time < T )
        {
            if( time + dt > T)
                dt = T-time;
            adapt.step( std::tie(ex, im, im), time, y0, time, y0, dt,
                    dg::imex_control, dg::l2norm, rtol, atol);
            counter ++;
        }
        //![adaptive]
        dg::blas1::axpby( -1., sol, 1., y0);
        double err = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        INFO("# steps "<< counter);
        INFO( "Relative error "<<name<<" is "<< err);
        CHECK( err < 1e-6);
    }
}
TEST_CASE( "Operator splitting")
{
    const double T = 1;
    const double nu = 0.01;
    //evaluate the initial condition
    const std::array<double,2> init( solution(0.,nu));
    std::array<double,2> y0(init);

    const std::array<double,2> sol = solution(T,nu);
    const double norm_sol = dg::blas1::dot( sol, sol);
    double time = 0.;
    Explicit ex( nu);
    Implicit im( nu);
    SECTION( "Strang operator splitting")
    {
        const double rtol = 1e-7, atol = 1e-10;
        auto name = GENERATE( as<std::string>{},
            "Heun-Euler-2-1-2",
            "Bogacki-Shampine-4-2-3",
            "ARK-4-2-3 (explicit)",
            "Zonneveld-5-3-4",
            "ARK-6-3-4 (explicit)",
            "Sayfy-Aburub-6-3-4",
            "Cash-Karp-6-4-5",
            "Fehlberg-6-4-5",
            "Dormand-Prince-7-4-5",
            "ARK-8-4-5 (explicit)"
        );
        time = 0., y0 = init;
        dg::Adaptive<dg::ERKStep<std::array<double,2>>> adapt( name, y0);
        dg::ImplicitRungeKutta<std::array<double,2>> dirk( "Trapezoidal-2-2", y0 );
        double dt = 1e-6;
        int counter=0;
        adapt.stepper().ignore_fsal();
        while( time < T )
        {
            if( time + dt > T)
                dt = T-time;
            double dt_old = dt;
            dirk.step( std::tie(im, im), time, y0, time, y0, dt_old/2.);
            adapt.step( ex, time-dt_old/2., y0, time, y0, dt, dg::pid_control,
                dg::l2norm, rtol, atol);
            dirk.step( std::tie(im, im), time-dt_old/2., y0, time, y0, dt_old/2.);
            counter ++;
        }
        dg::blas1::axpby( -1., sol, 1., y0);
        double err = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        INFO( "# steps "<<counter);
        INFO( "Relative error: "<<std::setw(24) <<name<<" "<<err);
        // Very lax condition here, not sure if there is a better test
        CHECK( err < 1e-3);
    }
}
