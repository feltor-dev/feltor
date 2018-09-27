#include <iostream>
#include <functional>

#undef DG_DEBUG
#include "multistep.h"
#include "runge_kutta.h"
#include "elliptic.h"

//![function]
//method of manufactured solution
struct Solution{
    Solution(double t, double nu):t(t), nu(nu){}
DG_DEVICE
    double operator()(double x, double y) const{
        return sin(t)*exp( -2.*nu*t)*sin(x)*sin(y);
    }
    private:
    double t, nu;
};

struct Source{
    Source(double t, double nu):t(t), nu(nu){}
DG_DEVICE
    double operator()(double x, double y) const{
        return sin(x)*sin(y)*cos(t)*exp(-2*t*nu)*(1-sin(t));
    }
    private:
    double t, nu;
};

//the explicit part contains the source Tp = S(x,y,t)
template<class container>
struct Explicit
{
    Explicit( const dg::Grid2d& g, double nu):
        m_nu( nu),
        m_x ( dg::evaluate(dg::cooX2d, g)),//x-coordinate
        m_y ( dg::evaluate(dg::cooY2d, g)) //y-coordinate
    {}
    void operator()( double t, const container& T, container& Tp) {
        dg::blas1::evaluate( Tp, dg::equals(), Source(t,m_nu), m_x, m_y);
    }
    private:
    const double m_nu;
    const container m_x, m_y;

};

//the implicit part contains  Tp = nu Delta T(x,y,t) + cos(t) T(x,y,t)
template< class Matrix, class container>
struct Implicit
{
    Implicit( const dg::Grid2d& g, double nu):
        m_nu(nu),
        m_w2d( dg::create::weights(g)),
        m_v2d( dg::create::inv_weights(g)),
        m_LaplacianM( g, dg::normed)
        { }

    void operator()( double t, const container& T, container& Tp)
    {
        dg::blas2::gemv( m_LaplacianM, T, Tp);
        dg::blas1::axpby( cos(t), T, -m_nu, Tp);
    }
    //required by inversion in semi-implicit schemes
    const container& inv_weights(){return m_v2d;}
    const container& weights(){return m_w2d;}
    const container& precond(){return m_v2d;}
  private:
    double m_nu;
    const container m_w2d, m_v2d, m_x, m_y;
    dg::Elliptic<dg::CartesianGrid2d, Matrix, container> m_LaplacianM;
};

//![function]

template< class Matrix, class container>
struct Full
{
    Full( const dg::Grid2d& g, double nu):
        m_exp( g, nu), m_imp( g, nu), m_temp( dg::evaluate( dg::one, g))

    { }
    void operator()( double t, const container& y, container& yp) {
        m_exp( t, y, yp);
        m_imp( t, y, m_temp);
        dg::blas1::axpby( 1., m_temp, 1., yp);
    }
  private:
    Explicit<container> m_exp;
    Implicit<Matrix, container> m_imp;
    container m_temp;
};


const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

//const unsigned NT = (unsigned)(nu*T*n*n*N*N/0.01/lx/lx);

int main()
{
    unsigned n = 3, Nx = 50 , Ny = 50;
    std::cout << "Program tests Multistep and Semi-Implicit methods on a manufactured PDE\n";
    const double T = 0.1;
    const double NT= 40, eps = 1e-8;
    const double dt = (T/NT);
    const double nu = 0.01;
    //construct the grid and the explicit and implicit parts
    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    Explicit<dg::DVec> exp( grid, nu);
    Implicit<dg::DMatrix, dg::DVec> imp( grid, nu);

    Full<dg::DMatrix, dg::DVec> full( grid, nu);
    //evaluate the initial condition
    const dg::DVec init( dg::evaluate(Solution(0.,nu), grid));
    dg::DVec y0(init);

    const dg::DVec sol = dg::evaluate( Solution(T,nu), grid);
    const dg::DVec w2d = dg::create::weights( grid);
    const double norm_sol = dg::blas2::dot( w2d, sol);
    double time = 0.;
    dg::DVec error( sol);

    dg::AB< 1, dg::DVec > ab1( y0);
    dg::AB< 2, dg::DVec > ab2( y0);
    dg::AB< 3, dg::DVec > ab3( y0);
    dg::AB< 4, dg::DVec > ab4( y0);
    dg::AB< 5, dg::DVec > ab5( y0);
    ab1.init( full, time, y0, dt);
    ab2.init( full, time, y0, dt);
    ab3.init( full, time, y0, dt);
    ab4.init( full, time, y0, dt);
    ab5.init( full, time, y0, dt);

    exblas::udouble res;

    //main time loop
    time = 0., y0 =  init;
    for( unsigned i=0; i<NT; i++)
        ab1.step( full, time, y0);
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error AB 1        is "<< res.d<<"\t"<<res.i<<std::endl;
    //main time loop
    time = 0., y0 =  init;
    for( unsigned i=0; i<NT; i++)
        ab2.step( full, time, y0);
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error AB 2        is "<< res.d<<"\t"<<res.i<<std::endl;
    //main time loop
    time = 0., y0 =  init;
    for( unsigned i=0; i<NT; i++)
        ab3.step( full, time, y0);
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error AB 3        is "<< res.d<<"\t"<<res.i<<std::endl;
    //main time loop
    time = 0., y0 =  init;
    for( unsigned i=0; i<NT; i++)
        ab4.step( full, time, y0);
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error AB 4        is "<< res.d<<"\t"<<res.i<<std::endl;
    //main time loop
    time = 0., y0 =  init;
    for( unsigned i=0; i<NT; i++)
        ab5.step( full, time, y0);
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error AB 5        is "<< res.d<<"\t"<<res.i<<std::endl;
    //![sirk]
    //construct time stepper (eps = 1e-8)
    dg::SIRK< dg::DVec > sirk( y0, y0.size(), eps);
    time = 0., y0 = init; //y0 and init are of type dg::DVec and contain the initial condition
    //main time loop (NT = 20, exp and imp are objects of type Explicit and Implicit defined above)
    for( unsigned i=0; i<NT; i++)
        sirk.step( exp, imp, time, y0, time, y0, dt); //inplace step
    //![sirk]
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error SIRK        is "<< res.d<<"\t"<<res.i<<std::endl;
    //![karniadakis]
    //construct time stepper
    dg::Karniadakis< dg::DVec > karniadakis( y0, y0.size(), eps);
    time = 0., y0 = init; //y0 and init are of type dg::DVec and contain the initial condition
    //initialize the timestepper (exp and imp are objects of type Explicit and Implicit defined above)
    karniadakis.init( exp, imp, time, y0, dt);
    //main time loop (NT = 20)
    for( unsigned i=0; i<NT; i++)
        karniadakis.step( exp, imp, time, y0); //inplace step
    //![karniadakis]
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error Karniadakis is "<< res.d<<"\t"<<res.i<<std::endl;
    //main time loop
    std::cout << "\nAdaptive SIRK Timer \n";
    time = 0., y0 =  init;
    double adapt = dt;
    while( time < T-adapt){
        sirk.adaptive_step( exp, imp, time, y0, time, y0, adapt, 1e-8, true);
    }
    adapt = T - time;
    sirk.adaptive_step( exp, imp, time, y0, time, y0, adapt, 1e-8, true);
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error adaptive sirk: "<< res.d<<"\t"<<res.i<<std::endl;
    return 0;
}
