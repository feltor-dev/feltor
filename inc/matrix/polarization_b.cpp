#include <iostream>

#include "dg/file/file.h"

#include "polarization.h"
#include "functors.h"

const double tau = 1.;
const double alpha = -tau;
const double beta = -tau/2.;
const double m = 4.;
const double n = 4.;

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;


dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

// df
double phi_ana_df( double x,double y){ return sin(m*x)*sin(n*y);}
double rho_ana_df( double x,double y){ return (m*m+n*n)/(-1.+(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}

// full_f  TODO: calculate boundary consistent analytical rho
double amp = 0.1;
// double chi_ana( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
double chi_ana( double, double y) {return 1. + amp*sin(y); } //must be strictly positive
// double rho_ana_FF( double x, double y) { return (amp*cos(2.*x)*(1./sqrt(1.-4.*alpha) - 2.*cos(2.*y)/sqrt(1.-8.*alpha)) + amp* cos(2.*y)/sqrt(1.-4.*alpha)-(4.*sin(x)*sin(y))/sqrt(1.-2.*alpha))/2.; }
// double phi_ana_FF(double x, double y)  { return (sin(x)*sin(y))*sqrt(1.-(1.+1.)*alpha);}
// double rho_ana_FF( double x, double y) { return (
//     amp*cos(2.*x)*(1./sqrt(1.-4.*alpha)  - 2.*cos(2.*y)/sqrt(1.-8.*alpha))
//     + amp* cos(2.*y)/sqrt(1.-4.*alpha)
//     -(4.*sin(x)*sin(y))/sqrt(1.-2.*alpha)
//         )/2./sqrt(1.-2.*alpha); }
double rho_ana_FF( double x, double y) { return  (sin(x)*((3.0*amp*(1.0-2.0*alpha)*cos(2.0*y))/sqrt(1.0-5.0*alpha)-             (amp-2.0*amp*alpha+4.0*sqrt(1.0-2.0*alpha)*sqrt(1.0-alpha)*sin(y))/sqrt(1.0-alpha)))/(2.0*sqrt((1.0-2.0*alpha)*(1.0-2.0*alpha)*(1.0-2.0*alpha))); }


double phi_ana_FF(double x, double y)  { return (sin(x)*sin(y));}


// double rho_ana_FFO4( double x, double y) { return ((amp * cos(2.* y))/(1. - 4.* beta) +
//  amp* cos(2.* x)* (1./(1. - 4.* beta) + (10.* cos(2.* y))/(-1. + 8.* beta)) + (
//  12.* sin(x) *sin(y))/(-1. + 2.* beta ))/(-2. + 4.* beta);}
double rho_ana_FFO4( double x, double y) { return  (sin(x)*(amp*(2.0 + 1.0/(-1.0 + beta)) + (11.0*amp*(1.0 - 2.0*beta)*cos(2.0*y))/(-1.0 + 5.0*beta) +
   12.0*sin(y)))/(2.0*(1.0 - 2.0*beta)*(1.0 - 2.0*beta));}

// //Full f cold
// double rho_ana_FFO2( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1.)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
// double phi_ana_FFO2(double x, double y)  { return sin( x)*sin(y);}


using Matrix = dg::DMatrix;
using Container = dg::DVec;

int main()
{
    dg::Timer t;

    unsigned n, Nx, Ny;
    std::cout << "# Type n, Nx and Ny! \n";
    std::cin >> n>> Nx >> Ny;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<std::endl;
    double eps_pol = 1e-6;
    double eps_gamma = 1e-7;
    std::cout << "# Type in eps_pol and eps_gamma (eps_gamma < eps_pol)\n";
    std::cin >> eps_pol >> eps_gamma;
    std::cout <<"# You typed\n"
              <<"eps_pol: "<<eps_pol<<"\n"
              <<"eps_gamma: "<<eps_gamma<<std::endl;
    std::vector<double> eps_pol_vec = {eps_pol, eps_pol, eps_pol};
    std::vector<double> eps_gamma_vec = {eps_gamma, eps_gamma, eps_gamma};

    dg::CartesianGrid2d grid2d( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);

    const Container w2d = dg::create::weights( grid2d);
    const Container one = dg::evaluate( dg::one, grid2d);
    const Container rho = dg::evaluate( rho_ana_df, grid2d);
    const Container sol = dg::evaluate( phi_ana_df, grid2d);
    const Container rho_FF = dg::evaluate( rho_ana_FF, grid2d);
    const Container sol_FF = dg::evaluate( phi_ana_FF, grid2d);
    Container x(rho.size(), 0.), temp(rho), error(rho), x_gamma(x);
    const Container chi =  dg::evaluate( chi_ana, grid2d);
    const Container rho_FFO4 =    dg::evaluate( rho_ana_FFO4, grid2d);

    dg::exblas::udouble res;

    dg::PCG <Container> pcg( x,  grid2d.size()*grid2d.size());

    const unsigned stages = 3;
    dg::MultigridCG2d<dg::CartesianGrid2d, Matrix, Container > multigrid( grid2d, stages);

    //df polarization charge with nested inversions
    {
        std::cout << "\n#####df polarization charge with nested inversion (commute = false)\n";
        std::cout << "df-nested-nocommute:" << std::endl;
        dg::mat::PolCharge< dg::CartesianGrid2d, Matrix, Container > pol_df;
        pol_df.construct(alpha, eps_gamma_vec, grid2d, grid2d.bcx(),
                grid2d.bcy(), dg::centered, 1., "df", true);
        pol_df.set_commute(false);
        dg::blas1::scal(x, 0.0);

        t.tic();
        unsigned number = pcg.solve( pol_df, x, rho, pol_df.precond(),
                pol_df.weights(), eps_pol);
        dg::blas1::scal(x, -1.0);
        t.toc();

        dg::blas1::axpby( 1., sol, -1., x, error);
        res.d = sqrt( dg::blas2::dot( w2d, error));
        std::cout << "    time: "<<t.diff() << "s \n";
        std::cout << "    iter:  "<<number<<std::endl;
        std::cout << "    error_abs: " << res.d<<" \n";
        std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;

        std::cout << "\n#####df polarization charge with nested inversion (commute = true)\n";
        std::cout << "df-nested-commute:" << std::endl;
        pol_df.set_commute(true);
        dg::blas1::scal(x,0.0);

        t.tic();
        number = pcg.solve( pol_df, x, rho, pol_df.precond(),
                pol_df.weights(), eps_pol);
        dg::blas1::scal(x, -1.0);
        t.toc();

        dg::blas1::axpby( 1., sol, -1., x, error);
        res.d = sqrt( dg::blas2::dot( w2d, error));
        std::cout << "    time: "<<t.diff() << "s \n";
        std::cout << "    iter:  "<<number<<std::endl;
        std::cout << "    error_abs: " << res.d<<std::endl;
        std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
    }

    //df polarization charge without nested inversions
    {
        std::cout << "#####df polarization charge without nested inversion (commute = false)\n";
        std::cout << "df-notnested-nocommute:" << std::endl;
        dg::Elliptic< dg::CartesianGrid2d, Matrix, Container > lapperp(grid2d, grid2d.bcx(), grid2d.bcy(), dg::centered);
        dg::Helmholtz< dg::CartesianGrid2d, Matrix, Container > gamma0inv(  alpha, {grid2d,grid2d.bcx(),grid2d.bcy(), dg::centered});

        dg::blas1::scal(x,0.0);

        t.tic();
        dg::blas1::copy( rho, temp);
        dg::blas1::scal(temp,-1.0);
        unsigned number = pcg.solve( lapperp, x, temp, 1., w2d, eps_pol);
        dg::blas2::symv(gamma0inv, x, temp);
        dg::blas1::copy( temp, x);
        t.toc();

        dg::blas1::axpby( 1., sol, -1., x, error);
        res.d = sqrt( dg::blas2::dot( w2d, error));
        std::cout << "    time: "<<t.diff() << "s \n";
        std::cout << "    iter:  "<<number<<std::endl;
        std::cout << "    error_abs: " << res.d<<std::endl;
        std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;


        std::cout << "#####df polarization charge without nested inversion (commute = true)\n";
        std::cout << "df-notnested-commute:" << std::endl;
        dg::blas1::scal(x, 0.0);

        t.tic();
        dg::blas2::symv(gamma0inv, rho, temp);
        dg::blas1::scal(temp,-1.0);
        number = pcg.solve( lapperp, x, temp, 1., w2d, eps_pol);
        t.toc();

        dg::blas1::axpby( 1., sol, -1., x, error);
        res.d = sqrt( dg::blas2::dot( w2d, error));
        std::cout << "    time: "<<t.diff() << "s \n";
        std::cout << "    iter:  "<<number<<std::endl;
        std::cout << "    error_abs: " << res.d<<std::endl;
        std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
    }

    //ff polarization charge of order 2 with nested inversions //TODO not converging
    {
        std::cout << "#####ff polarization charge with nested inversion (commute = false)\n";
        std::cout << "ffO2-nested-nocommute:" << std::endl;

        dg::mat::PolCharge< dg::CartesianGrid2d, Matrix, Container > pol_ff;
        pol_ff.construct(alpha, eps_gamma_vec, grid2d,
                dg::centered, 1., "ff", false);
        pol_ff.set_commute(false);
        pol_ff.set_chi(chi);
        dg::blas1::copy(0., x);

        t.tic();
        unsigned number = pcg.solve( pol_ff, x, rho_FF, pol_ff.precond(),
                pol_ff.weights(), eps_pol);
        dg::blas1::scal(x,-1.0);
        t.toc();

        dg::blas1::axpby( 1., sol_FF, -1., x, error);
        res.d = sqrt( dg::blas2::dot( w2d, error));
        std::cout << "    time: "<<t.diff() << "s \n";
        std::cout << "    iter:  "<<number<<std::endl;
        std::cout << "    error_abs: " << res.d<<std::endl;
        std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol_FF))<<std::endl;

        //test application of operator //TODO not converging but relatively close to sol
        dg::blas2::symv( pol_ff,  sol_FF, x);
        dg::blas1::scal(x,-1.0);

        dg::blas1::axpby( 1., rho_FF, -1., x, error);
        std::cout << "#   error_rel in Operator application " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, rho_FF))<<std::endl;
    }
    //ff polarization charge of order 2 without nested inversions
    {
        std::cout << "#####ff polarization charge without nested inversion (commute = false)\n";
        std::cout << "ffO2-notnested-nocommute:" << std::endl;
        dg::Elliptic< dg::CartesianGrid2d, Matrix, Container > lapperp(grid2d, grid2d.bcx(), grid2d.bcy(), dg::centered);
        lapperp.set_chi( chi);
        dg::Helmholtz< dg::CartesianGrid2d, Matrix, Container > gamma0inv(  alpha,{grid2d, grid2d.bcx(), grid2d.bcy(), dg::centered, 1.0});
        dg::blas1::scal(x_gamma, 0.0);
        dg::blas1::scal(temp, 0.0);
        dg::blas1::scal(x, 0.0);
        t.tic();
        unsigned numberCG = 0;
        //auto func = dg::mat::make_FuncEigen_Te1([](double x){return sqrt(x);});
        dg::mat::MatrixSqrt<Container> sqrtsolve( gamma0inv, +1, w2d,
                eps_gamma);

        dg::apply( sqrtsolve, rho_FF, temp);
        dg::blas1::scal(temp,-1.0);
        numberCG = pcg.solve( lapperp, x_gamma, temp, 1., w2d, eps_pol);
        dg::apply( sqrtsolve, x_gamma, x);

        t.toc();

        dg::blas1::axpby( 1., sol_FF, -1., x, error);
        res.d = sqrt( dg::blas2::dot( w2d, error));
        std::cout << "    time: "<<t.diff() << "s \n";
        std::cout << "    iter:  "<<numberCG<<std::endl;
        std::cout << "    error_abs: " << res.d<<std::endl;
        std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol_FF))<<std::endl;
    }
//
//   ff polarization charge of order 4 with nested inversions (Note: converges for eps_gamma <=  machine precision and if BC fo outer helmholtz are changed to PER, )
//     {
//         std::cout << "#####ff polarization charge of order 4 with nested inversions  (commute = false)\n";
//         std::cout << "ffO4-nested-nocommute:" << std::endl;
//         dg::PolCharge<dg::CartesianGrid2d, Matrix, Container> pol_ffO4;
//         eps_gamma_vec = {eps_gamma, 0.1*eps_gamma, 0.1*eps_gamma};
// 
//         pol_ffO4.construct(beta, eps_gamma_vec, grid2d, grid2d.bcx(), grid2d.bcy(), dg::centered, 1., false, "ffO4");
//         pol_ffO4.set_chi(chi);
//         pol_ffO4.set_iota(chi);
//
//         dg::blas1::scal(x, 0.0);
//
//         t.tic();
//         unsigned number = pcg.solve(pol_ffO4, x, rho_FF04, eps_pol);
//         t.toc();
//
//         dg::blas1::axpby( 1., sol_FF, -1., x, error);
//         res.d = sqrt( dg::blas2::dot( w2d, error));
//         std::cout << "    time: "<<t.diff() << "s \n";
//         std::cout << "    iter:  "<<number<<std::endl;
//         std::cout << "    error_abs: " << res.d<<std::endl;
//         std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol_FF))<<std::endl;
//
//
//     }
//
// //  ff polarization charge of order 4 without nested inversions
//     {
//         std::cout << "#####ff polarization charge of order 4 without nested inversion (commute = false)\n";
//         std::cout << "ffO4-notnested-nocommute:" << std::endl;
//         const std::vector<Container> multi_chi = multigrid.project( chi);
//
//         std::vector<dg::TensorElliptic<dg::CartesianGrid2d, Matrix, Container> > multi_tensorelliptic( stages);
//         for(unsigned u=0; u<stages; u++)
//         {
//             multi_tensorelliptic[u].construct( multigrid.grid(u), dg::centered, 1.);
//             multi_tensorelliptic[u].set_chi( multi_chi[u]);
//             multi_tensorelliptic[u].set_iota( multi_chi[u]);
//         }
//         dg::Helmholtz< dg::CartesianGrid2d,  Matrix, Container > gamma1inv(beta, {grid2d, grid2d.bcx(),grid2d.bcy(), dg::centered, 1.});
//         dg::blas1::scal(x, 0.0);
//         eps_pol_vec = {eps_pol, 0.1*eps_pol, 0.1*eps_pol};
//
//         t.tic();
//         dg::blas2::symv(gamma1inv, rho_FFO4, temp); //fullfills no DIR bc conditions/only PER on [0,2 pi]!
//         std::vector<unsigned> number = multigrid.direct_solve(multi_tensorelliptic, x, temp, eps_pol_vec);
//         dg::blas2::symv(gamma1inv, x, temp);
//         dg::blas1::copy( temp, x);
//         t.toc();
//
//         dg::blas1::axpby( 1., sol_FF, -1., x, error);
//         res.d = sqrt( dg::blas2::dot( w2d, error));
//         std::cout << "    time: "<<t.diff() << "s \n";
//         for( unsigned u=0; u<number.size(); u++)
//             std::cout << " # iter["<<u<<"] "<< number.size()-1-u << " " << number[number.size()-1-u] << " \n";
//         std::cout << "    iter:  "<<number[0]+number[1]+number[2]<<std::endl;
//         std::cout << "    error_abs: " << res.d<<std::endl;
//         std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol_FF))<<std::endl;
//     }
//

    return 0;
}
