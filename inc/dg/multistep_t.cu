#include <iostream>
#include <iomanip>
#include <functional>

#undef DG_DEBUG
#include "multistep.h"
#include "adaptive.h"
#include "elliptic.h"

//![function]
//method of manufactured solution
std::array<double,2> solution( double t, double nu) {
    return {exp( -nu*t) + cos(t), exp( -nu*t) + sin(t)};
}

//the explicit part contains the source Tp = S(x,y,t)
struct Explicit
{
    Explicit( double nu): m_nu( nu) {}
    void operator()( double t, const std::array<double,2>& T, std::array<double,2>& Tp)
    {
        Tp[0] = m_nu*cos(t) - sin(t);
        Tp[1] = m_nu*sin(t) + cos(t);
    }
    private:
    const double m_nu;
};

//the implicit part contains  Tp = -nu T
struct Implicit
{
    Implicit( double nu): m_nu(nu) { }

    void operator()( double t, const std::array<double,2>& T, std::array<double,2>& Tp)
    {
        Tp[0] = -m_nu*T[0];
        Tp[1] = -m_nu*T[1];
    }
  private:
    double m_nu;
};
struct ImplicitSolver
{
    ImplicitSolver( double nu): m_nu(nu) { }
    std::array<double,2> copyable() const {return {0,0};}
    template<class Implicit>
    void solve( double alpha, Implicit& im, double t, std::array<double,2>& y, const std::array<double,2>& rhs)
    {
        y[0] = rhs[0]/(1-alpha*m_nu);
        y[1] = rhs[1]/(1-alpha*m_nu);
    }
  private:
    double m_nu;
};
struct FullSolver
{
    FullSolver( double nu): m_nu(nu) { }
    std::array<double,2> copyable() const {return {0,0};}
    template<class Implicit>
    void solve( double alpha, Implicit& im, double t, std::array<double,2>& y, const std::array<double,2>& rhs)
    {
        y[0] = (rhs[0]-alpha*(m_nu*cos(t) - sin(t)))/(1-alpha*m_nu);
        y[1] = (rhs[1]-alpha*(m_nu*sin(t) + cos(t)))/(1-alpha*m_nu);
    }
  private:
    double m_nu;
};

//![function]

struct Full
{
    Full( double nu):
        m_exp( nu), m_imp( nu) { }
    void operator()( double t, const std::array<double,2>& y, std::array<double,2>& yp) {
        m_exp( t, y, yp);
        m_imp( t, y, m_temp);
        dg::blas1::axpby( 1., m_temp, 1., yp);
    }
  private:
    Explicit m_exp;
    Implicit m_imp;
    std::array<double,2> m_temp;
};


const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

//const unsigned NT = (unsigned)(nu*T*n*n*N*N/0.01/lx/lx);

int main()
{
    std::cout << "Program tests Multistep and Semi-Implicit methods on a manufactured PDE\n";
    const double T = 1;
    const double NT= 40;
    const double dt = (T/NT);
    const double nu = 0.01;
    Full full( nu);
    //evaluate the initial condition
    const std::array<double,2> init( solution(0.,nu));
    std::array<double,2> y0(init);

    const std::array<double,2> sol = solution(T,nu);
    const double norm_sol = dg::blas1::dot( sol, sol);
    double time = 0.;
    std::array<double,2> error( sol);
    exblas::udouble res;
    std::cout << "### Test Adams Bashforth methods with "<<NT<<" steps\n";
    for( unsigned s=1; s<6; s++)
    {
        time = 0., y0 = init;
        dg::AdamsBashforth< std::array<double,2> > ab( s, y0);
        ab.init( full, time, y0, dt);
        //main time loop
        for( unsigned k=0; k<NT; k++)
            ab.step( full, time, y0);
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        std::cout << "Relative error AB "<<s<<"        is "<< res.d<<"\t"<<res.i<<std::endl;
    }
    std::cout << "### Test Explicit Multistep methods with "<<NT<<" steps\n";
    for( unsigned s=1; s<7; s++)
    {
        time = 0., y0 = init;
        dg::ExplicitMultistep< std::array<double,2> > ab( "eBDF", s, y0);
        ab.init( full, time, y0, dt);
        //main time loop
        for( unsigned k=0; k<NT; k++)
            ab.step( full, time, y0);
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        std::cout << "Relative error eBDF "<<s<<"        is "<< res.d<<"\t"<<res.i<<std::endl;
    }
    for( unsigned s=1; s<6; s++)
    {
        time = 0., y0 = init;
        dg::ExplicitMultistep< std::array<double,2> > ab( "TVB", s, y0);
        ab.init( full, time, y0, dt);
        //main time loop
        for( unsigned k=0; k<NT; k++)
            ab.step( full, time, y0);
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        std::cout << "Relative error TVB  "<<s<<"        is "<< res.d<<"\t"<<res.i<<std::endl;
    }
    std::cout << "### Test implicit multistep methods with "<<NT<<" steps\n";
    for( unsigned s=1; s<7; s++)
    {
        time = 0., y0 = init;
        dg::BDF< std::array<double,2>, FullSolver > bdf( s, nu);
        bdf.init( full, time, y0, dt);
        //main time loop
        for( unsigned k=0; k<NT; k++)
            bdf.step( full, time, y0);
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        std::cout << "Relative error BDF "<<s<<"        is "<< res.d<<"\t"<<res.i<<std::endl;
    }
    Explicit ex( nu);
    Implicit im( nu);
    std::cout << "### Test semi-implicit Karniadakis methods with "<<NT<<" steps\n";
    //![karniadakis]
    //construct time stepper
    dg::Karniadakis< std::array<double,2>, ImplicitSolver > karniadakis( nu);
    time = 0., y0 = init; //y0 and init are of type std::array<double,2> and contain the initial condition
    //initialize the timestepper (ex and im are objects of type Explicit and Implicit defined above)
    karniadakis.init( ex, im, time, y0, dt);
    //main time loop (NT = 20)
    for( unsigned i=0; i<NT; i++)
        karniadakis.step( ex, im, time, y0); //inplace step
    //![karniadakis]
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
    std::cout << "Relative error Karniadakis 3 is "<< res.d<<"\t"<<res.i<<std::endl;
    for( unsigned s=2; s>0; s--)
    {
        time = 0., y0 = init;
        karniadakis.set_order(s);
        karniadakis.init( ex, im, time, y0, dt);
        for( unsigned i=0; i<NT; i++)
            karniadakis.step( ex, im, time, y0);
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        std::cout << "Relative error Karniadakis "<<s<<" is "<< res.d<<"\t"<<res.i<<std::endl;
    }


    std::cout << "### Test semi-implicit ARK methods\n";
    std::vector<std::string> names{"ARK-4-2-3", "ARK-6-3-4", "ARK-8-4-5"};
    double rtol = 1e-7, atol = 1e-10;
    for( auto name : names)
    {
        //![adaptive]
        time = 0., y0 = init;
        dg::Adaptive<dg::ARKStep<std::array<double,2>, ImplicitSolver>> adapt( name, nu);
        double time = 0;
        double dt = adapt.guess_stepsize( ex, time, y0, dg::forward, dg::l2norm, rtol, atol);
        int counter=0;
        while( time < T )
        {
            if( time + dt > T)
                dt = T-time;
            adapt.step( ex, im, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, rtol, atol);
            counter ++;
        }
        //![adaptive]
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        std::cout << counter <<" steps! ";
        std::cout << "Relative error "<<name<<" is "<< res.d<<"\t"<<res.i<<std::endl;
    }
    std::cout << "### Test Strang operator splitting\n";

    std::vector<std::string> ex_names{
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
    };
    for( auto name : ex_names)
    {
        time = 0., y0 = init;
        dg::Adaptive<dg::ERKStep<std::array<double,2>>> adapt( name, y0);
        dg::ImplicitRungeKutta<std::array<double,2>, ImplicitSolver> dirk( "Trapezoidal-2-2", nu );
        double time = 0;
        double dt = adapt.guess_stepsize( ex, time, y0, dg::forward, dg::l2norm, rtol, atol);
        int counter=0;
        adapt.stepper().ignore_fsal();
        while( time < T )
        {
            if( time + dt > T)
                dt = T-time;
            double dt_old = dt;
            dirk.step( im, time, y0, time, y0, dt_old/2.);
            adapt.step( ex, time-dt_old/2., y0, time, y0, dt, dg::pid_control,
                dg::l2norm, rtol, atol);
            dirk.step( im, time-dt_old/2., y0, time, y0, dt_old/2.);
            counter ++;
        }
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas1::dot( y0, y0)/norm_sol);
        std::cout << std::setw(4)<<counter <<" steps! ";
        std::cout << "Relative error "<<std::setw(24) <<name<<"\t"<<res.d<<"\n";
    }
    return 0;
}
