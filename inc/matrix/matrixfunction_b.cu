#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"
#include "lanczos.h"
#include "mcg.h"
#include "matrixfunction.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double m=4.;
const double n=4.;
const double alpha = 0.5;
const double ell_fac = (m*m+n*n);
//const double helm_fac = 1.+ ell_fac;

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}

using Matrix = dg::DMatrix;
using Container = dg::DVec;

int main()
{
    dg::Timer t;

    unsigned n, Nx, Ny;
    std::cout << "# Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<std::endl;
    unsigned iter = 0;

    unsigned max_iter = 1;
    std::cout << "# Type max_iter of tridiagonalization (500) ?\n";
    std::cin >> max_iter ;
    std::cout << "# Type in eps of tridiagonalization (1e-7)\n";
    double eps = 1e-7; //# of pcg iter increases very much if
    std::cin >> eps;
    std::cout <<"# You typed\n"
              <<"max_iter: "<<max_iter<<"\n"
              <<"eps: "<<eps<<std::endl;

    double erel = 0;

    dg::Grid2d g( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
    const Container w2d = dg::create::weights( g);
    double max_weights = dg::blas1::reduce(w2d, 0., dg::AbsMax<double>() );
    double min_weights = dg::blas1::reduce(w2d, max_weights, dg::AbsMin<double>() );
    std::cout << "#   min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
    const double kappa = sqrt(max_weights/min_weights); //condition number
    //dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( alpha, {g, dg::centered});
    dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( {g, dg::centered, 1.0});
    dg::mat::UniversalLanczos<Container> lanczos( A.weights(), 20);
    auto T = lanczos.tridiag( A, A.weights(), A.weights());
    auto extremeEVs = dg::mat::compute_extreme_EV( T);
    double EVmin = extremeEVs[0];
    double EVmax = extremeEVs[1];

    std::vector< std::function<double (double)>> funcs{
	//Elliptic
        [](double x) { return sqrt(alpha*x);},
        [](double x) { return 1./sqrt(alpha*x);},
        [](double x) { return exp(-alpha*x);},
        [](double x) { return boost::math::cyl_bessel_j(0, alpha*sqrt(x));},
        [](double x) { return exp(-alpha*x)*boost::math::cyl_bessel_i(0, -alpha*x);},
        [](double x) { return 1./(alpha*x);},
	//Helmholtz
        [](double x) { return sqrt(1.+alpha*x);},
        [](double x) { return 1./sqrt(1.+alpha*x);},
        [](double x) { return 1./(1.+alpha*x);}
    };
    std::vector<std::string> outs = {
	    "Sqrt(alpha A)", 
	    "Inv(Sqrt(alpha A))", 
	    "Exp(-alpha A)",
        "BesselJ0(alpha Sqrt( A))",
	    "Exp(-alpha A) BesselI0(-alpha A)", 
	    "Inv(alpha A)",
        "Sqrt(1+alpha A)", 
	    "Inv(Sqrt(1+alpha A))", 
	    "Inv(1+alpha A)"};
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        dg::blas1::scal(x_exac, funcs[u](ell_fac));

        double res_fac = kappa*funcs[u](EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
        std::cout << outs[u] << ":\n";
        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);
        t.tic();
        auto func = dg::mat::make_FuncEigen_Te1( funcs[u]);
        iter = krylovfunceigen.solve(x, func, A, b, w2d, eps, 1.,
                "residual", res_fac);
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    residual-time: "<<time<<"s \n";
        std::cout << "    residual-error: "<<erel  << "\n";
        std::cout << "    residual-iter: "<<std::setw(3)<<iter << "\n";

        dg::mat::MCGFuncEigen<Container> mcgfunceigen( x, max_iter);
        t.tic();
        iter = mcgfunceigen(x, funcs[u], A, b, w2d, eps, 1.,
                res_fac);
        t.toc();
        time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    mcg-time: "<<time<<"s \n";
        std::cout << "    mcg-error: "<<erel  << "\n";
        std::cout << "    mcg-iter: "<<std::setw(3)<<iter << "\n";

        t.tic();
        iter = krylovfunceigen.solve(x, func, A, b, w2d, eps, 1.,
                "universal");
        t.toc();
        time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
        std::cout << "    universal-iter: "<<std::setw(3)<<iter << "\n";
    }
    return 0;
}
