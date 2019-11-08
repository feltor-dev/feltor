#include <iostream>
#include <iomanip>
#include <functional>

#undef DG_DEBUG
#include "multistep.h"
#include "adaptive.h"
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
    const container& weights(){return m_imp.weights();}
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
    //std::cout << "Type n (3), Nx (50), Ny (50)\n";
    //std::cin >> n >> Nx >> Ny;
    std::cout << "Computing on "<<n<<" x "<<Nx<<" x "<<Ny<<"\n";
    const double T = 0.1;
    const double NT= 40, eps = 1e-6;
    const double dt = (T/NT);
    const double nu = 0.01;
    //construct the grid and the explicit and implicit parts
    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);

    Full<dg::DMatrix, dg::DVec> full( grid, nu);
    //evaluate the initial condition
    const dg::DVec init( dg::evaluate(Solution(0.,nu), grid));
    dg::DVec y0(init);

    const dg::DVec sol = dg::evaluate( Solution(T,nu), grid);
    const dg::DVec w2d = dg::create::weights( grid);
    const double norm_sol = dg::blas2::dot( w2d, sol);
    double time = 0.;
    dg::DVec error( sol);
    exblas::udouble res;
    std::cout << "### Test explicit multistep methods with "<<NT<<" steps\n";
    for( unsigned s=1; s<6; s++)
    {
        time = 0., y0 = init;
        dg::AdamsBashforth< dg::DVec > ab( s, y0);
        ab.init( full, time, y0, dt);
        //main time loop
        for( unsigned k=0; k<NT; k++)
            ab.step( full, time, y0);
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
        std::cout << "Relative error AB "<<s<<"        is "<< res.d<<"\t"<<res.i<<std::endl;
    }
    std::cout << "### Test implicit multistep methods with "<<NT<<" steps\n";
    for( unsigned s=1; s<7; s++)
    {
        time = 0., y0 = init;
        dg::BDF< dg::DVec, dg::AndersonSolver<dg::DVec> > bdf( s, y0, 0, 1e-10, 100, 1, 1);
        //dg::BDF< dg::DVec, dg::FixedPointSolver<dg::DVec> > bdf( s, y0, 10, 1e-10);
        bdf.init( full, time, y0, dt);
        //main time loop
        for( unsigned k=0; k<NT; k++)
            bdf.step( full, time, y0);
        dg::blas1::axpby( -1., sol, 1., y0);
        res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
        std::cout << "Relative error BDF "<<s<<"        is "<< res.d<<"\t"<<res.i<<std::endl;
    }
    Explicit<dg::DVec> ex( grid, nu);
    Implicit<dg::DMatrix, dg::DVec> im( grid, nu);
    std::cout << "### Test semi-implicit Karniadakis methods with "<<NT<<"steps\n";
    //![karniadakis]
    //construct time stepper
    dg::Karniadakis< dg::DVec > karniadakis( y0, y0.size(), eps);
    time = 0., y0 = init; //y0 and init are of type dg::DVec and contain the initial condition
    //initialize the timestepper (ex and im are objects of type Explicit and Implicit defined above)
    karniadakis.init( ex, im, time, y0, dt);
    //main time loop (NT = 20)
    for( unsigned i=0; i<NT; i++)
        karniadakis.step( ex, im, time, y0); //inplace step
    //![karniadakis]
    dg::blas1::axpby( -1., sol, 1., y0);
    res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
    std::cout << "Relative error Karniadakis is "<< res.d<<"\t"<<res.i<<std::endl;


    std::cout << "### Test semi-implicit ARK methods\n";
    std::vector<std::string> names{"ARK-4-2-3", "ARK-6-3-4", "ARK-8-4-5"};
    double rtol = 1e-7, atol = 1e-10;
    for( auto name : names)
    {
        //![adaptive]
        time = 0., y0 = init;
        dg::Adaptive<dg::ARKStep<dg::DVec>> adapt( name, y0, y0.size(), eps);
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
        res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
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
        dg::Adaptive<dg::ERKStep<dg::DVec>> adapt( name, y0);
        dg::ImplicitRungeKutta<dg::DVec> dirk( "Trapezoidal-2-2", y0, y0.size(), eps );
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
        res.d = sqrt(dg::blas2::dot( w2d, y0)/norm_sol);
        std::cout << std::setw(4)<<counter <<" steps! ";
        std::cout << "Relative error "<<std::setw(24) <<name<<"\t"<<res.d<<"\n";
    }
    return 0;
}
