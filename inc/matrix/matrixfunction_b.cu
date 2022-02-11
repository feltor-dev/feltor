// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"
#include "matrixfunction.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double m=1.;
const double n=1.;
const double alpha = -1.;
const double ell_fac = -alpha*(m*m+n*n);
const double helm_fac = 1.+ ell_fac;

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}

using Matrix = dg::DMatrix;
using Container = dg::DVec;

int main(int argc, char * argv[])
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
    double max_weights =   dg::blas1::reduce(w2d, 0., dg::AbsMax<double>() );
    double min_weights =  -dg::blas1::reduce(w2d, max_weights, dg::AbsMin<double>() );
    const double kappa = sqrt(max_weights/min_weights); //condition number
    std::cout << "#   min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
    double hxhy = 1.; // ((2pi)/lx )^2

    auto sqrt_f =  [](double x) { return sqrt(x);};
    auto sqrt_inv_f =  [](double x) { return 1./sqrt(x);};
    auto exp_inv_f =  [](double x) { return 1./exp(x);};
    auto besseli0_inv_f =  [](double x) {
        return 1./boost::math::cyl_bessel_i(0, x);};
    auto gamma0_inv_f = [](double x){
        return 1./exp(x)/boost::math::cyl_bessel_i(0, x);
    };
    {
        std::cout << "\n#Compute x = Sqrt(1+ alpha Delta) b " << std::endl;
        dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( g, alpha, dg::centered);

        double EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0);
        double EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny())); //EVs of helmholtz

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        dg::blas1::scal(x_exac, sqrt_f(helm_fac));

        double res_fac = kappa*1./sqrt_f(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
        std::cout << "SQRT (M-CG+Eigen):\n";
        dg::mat::LanczosFuncEigenSolve<Container> krylovfunceigensolve( x, max_iter);
        t.tic();
        iter = krylovfunceigensolve(x, sqrt_f, A, b, w2d, eps, 1., res_fac);
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter << "\n";
    }
    {
        std::cout << "\n#Compute  x = Sqrt(1+ alpha Delta)^(-1) b " << std::endl;
        dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( g, alpha, dg::centered);
        //EVs of helmholtz
        double EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0);
        double EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        dg::blas1::scal(x_exac, sqrt_inv_f(helm_fac));

        double res_fac = kappa*sqrt_inv_f(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
        std::cout << "SQRT (M-CG+Eigen):\n";
        dg::mat::LanczosFuncEigenSolve<Container> krylovfunceigeninvert( x,   max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, sqrt_inv_f, A,  b, w2d, eps, 1., res_fac);
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter << "\n";
    }
    {
        std::cout << "\n#Compute x = e^(-alpha Delta) b" << std::endl;

        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( g, g.bcx(), g.bcy(), dg::centered); //negative laplace
        Container chi = dg::evaluate(dg::one, g);
        dg::blas1::scal(chi, -alpha); //-alpha must be positive for SPD operator! // MW ???
        A.set_chi(chi);
        //EVs of elliptic
        double EVmin = -alpha*hxhy*(1.0 + 1.0);
        double EVmax = -alpha*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        dg::blas1::scal(x_exac, exp_inv_f(ell_fac));

        double res_fac = kappa*exp_inv_f(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";

        std::cout << "EXP (M-CG+Eigen):\n";
        dg::mat::LanczosFuncEigenSolve<Container> krylovfunceigeninvert( x,   max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, exp_inv_f, A, b, w2d, eps, 1., res_fac);
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter << "\n";

    }
    {
        std::cout << "\n#Compute x = I_0(-alpha Delta) b" << std::endl;

        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( g, g.bcx(), g.bcy(), dg::centered); //negative laplace
        Container chi = dg::evaluate(dg::one, g);
        dg::blas1::scal(chi, -alpha); //-alpha must be positive for SPD operator!
        A.set_chi(chi);
        //EVs of elliptic
        double EVmin = -alpha*hxhy*(1.0 + 1.0);
        double EVmax = -alpha*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));


        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        dg::blas1::scal(x_exac, besseli0_inv_f(ell_fac));

        double res_fac = kappa*besseli0_inv_f(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";

        std::cout << "BESSELI0 (M-CG+Eigen):\n";
        dg::mat::LanczosFuncEigenSolve<Container> krylovfunceigeninvert( x, max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, besseli0_inv_f, A, b, w2d, eps, 1., res_fac);
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter << "\n";

    }
    {
        std::cout << "\n#Compute x = I_0(alpha Delta)^-1 e^(-alpha Delta) b " << std::endl;

        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( g, g.bcx(), g.bcy(), dg::centered); //negative laplace
        Container chi = dg::evaluate(dg::one, g);
        dg::blas1::scal(chi, -alpha); //-alpha must be positive for SPD operator!
        A.set_chi(chi);
        //EVs of elliptic
        double EVmin = -alpha*hxhy*(1.0 + 1.0);
        double EVmax = -alpha*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        dg::blas1::scal(x_exac, gamma0_inv_f(ell_fac));

        double res_fac = kappa*gamma0_inv_f(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";

        std::cout << "GAMMA0 (M-CG+Eigen):\n";
        dg::mat::LanczosFuncEigenSolve<Container> krylovfunceigeninvert( x,   max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, gamma0_inv_f, A, b, w2d, eps, 1., res_fac);
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter << "\n";

    }
    return 0;
}
