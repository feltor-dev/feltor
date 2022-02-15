#define SILENT

#include <iostream>
#include <iomanip>

#include "matrixsqrt.h"
#include "matrixfunction.h" 
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -0.5;
const double m=4.;
const double n=4.;
double lhs( double x, double y){ return sin(x*m)*sin(y*n);}
double rhsHelmholtz( double x, double y){ return (1.-(m*m+n*n)*alpha)*sin(x*m)*sin(y*n);}
double rhsHelmholtzsqrt( double x, double y){ return sqrt(1.-(m*m+n*n)*alpha)*sin(x*m)*sin(y*n);}


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
    double epsCG = 1e-8;
    double epsTrel = 1e-6;
    double epsTabs = 1e-8;
    std::cout << "# Type epsilon for CG (1e-8), and eps_rel (1e-9) and eps_abs (1e-12) for TimeStepper\n";
//     std::cin >> epsCG >> epsTrel >> epsTabs;
    unsigned max_iter = 500;
    unsigned max_iterC = 40;
    std::cout << "# Type max_iter of tridiagonalization (500) and of Cauchy integral (40) ?\n";
//     std::cin >> max_iter >> max_iterC;
    std::cout << "# Type in eps of tridiagonalization (1e-7)\n";
    double eps = 1e-7; //# of pcg iter increases very much if
//     std::cin >> eps;
    std::cout <<"# You typed\n"
              <<"epsCG:  "<<epsCG<<"\n"
              <<"epsTrel: "<<epsTrel<<"\n"
              <<"epsTabs: "<<epsTabs<<"\n"
              <<"max_iter: "<<max_iter<<"\n"
              <<"max_iterC: "<<max_iterC<<"\n"
              <<"eps: "<<eps<<std::endl;


    double erel = 0;
    unsigned iter_arr;

    dg::Grid2d g( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
   //start and end vectors
    Container x = dg::evaluate(lhs, g);
    const Container x_exac = dg::evaluate(lhs, g);
    Container b = dg::evaluate(rhsHelmholtzsqrt, g), error(b);
    Container bs = dg::evaluate(rhsHelmholtz, g);
    const Container bs_exac(bs), b_exac(b);

    const Container w2d = dg::create::weights( g);

    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( g, alpha, dg::centered);
    auto invert = [ eps = epsCG, pcg = dg::PCG<Container>( x, g.size())] (
            auto& A, auto& x, const auto& y) mutable
    {
        dg::blas1::copy( 0, x);
        return pcg.solve( A, x, y, A.precond(), A.weights(), eps);
    };

    double hxhy = 1.; // ((2pi)/lx )^2
    double max_weights = dg::blas1::reduce(A.weights(), 0., dg::AbsMax<double>() );
    double min_weights = dg::blas1::reduce(A.weights(), max_weights, dg::AbsMin<double>() );
    double EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0);
    double EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));
    double kappa = sqrt(max_weights/min_weights); //condition number of weight matrix



    ////////////////////////Direct Cauchy integral solve
    {
        std::cout << "\nCauchy-Inv: \n";
        dg::mat::DirectSqrtCauchy<Container>
            directsqrtcauchy(A, w2d, epsCG, max_iterC, EVmin, EVmax, -1);

        t.tic();
        iter_arr = directsqrtcauchy(b_exac, x);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        double time = t.diff();

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<iter_arr << "\n";

        std::cout << "\nCauchy-Inv+A: \n";
        t.tic();
        dg::blas2::symv( A, x, bs);
        t.toc();
        time+=t.diff();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));
        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<1 << "\n";
    }
    //////////////////Krylov solve via Lanczos method and Cauchy solve
    {
        std::cout << "\nM-Lanczos+Cauchy-Inv:\n";
        dg::mat::KrylovSqrtCauchy<Container> krylovsqrtcauchy(A, -1, w2d,
                EVmin, EVmax, max_iterC, eps, max_iter);
        t.tic();
        iter_arr = krylovsqrtcauchy(b_exac, x);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        double time = t.diff();

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter_arr << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<max_iterC << "\n";

        std::cout << "\nM-Lanczos+Cauchy:\n";
        krylovsqrtcauchy.construct(A, +1, w2d,
                EVmin, EVmax, max_iterC, eps, max_iter);
        t.tic();
        iter_arr = krylovsqrtcauchy(b_exac, bs);
        t.toc();
        time = t.diff();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));
        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter_arr << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<max_iterC << "\n";
    }

    //////////////////Krylov solve via Lanczos method and ODE sqrt solve
    {
        std::cout << "\nM-Lanczos+ODE:\n";
        dg::mat::KrylovSqrtODE<Container>
            krylovsqrtode(A, +1, w2d, "Dormand-Prince-7-4-5", epsTrel,
                    epsTabs, max_iter, eps);
        b = dg::evaluate(rhsHelmholtzsqrt, g);
        t.tic();
        auto iterA = krylovsqrtode(b_exac, bs);
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));
        double time = t.diff();

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iterA[0] << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<iterA[1] << "\n";

        std::cout << "\nM-Lanczos+ODE-Inv:\n";
        krylovsqrtode.construct(A, -1, w2d, "Dormand-Prince-7-4-5", epsTrel,
                    epsTabs, max_iter, eps);
        t.tic();
        iterA = krylovsqrtode( bs, b);
        t.toc();
        time = t.diff();
        dg::blas1::axpby(1.0, b, -1.0, b_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac));
        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iterA[0] << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<iterA[1] << "\n";
    }
    //////////////////Krylov solve via Lanczos method and ODE sqrt solve
    {
        std::cout << "\nUniversal-M-Lanczos-Eigen:\n";
        //EVs of Helmholtz
        double res_fac = kappa*sqrt(EVmin);

        dg::mat::Lanczos<Container> krylovfunceigen(x,   max_iter);
        b = dg::evaluate(rhsHelmholtzsqrt, g);
        auto sqrt_f =  [](double x) { return sqrt(x);};
        t.tic();
        iter_arr  = krylovfunceigen.solve( bs, sqrt_f, A, b_exac,
                A.weights(),  eps, 1., "universal", res_fac);
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));
        double time = t.diff();

        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter_arr << "\n";

        std::cout << "\nUniversal-M-Lanczos-Eigen-Inv:\n";
        auto inv_sqrt_f =  [](double x) { return 1./sqrt(x);};
        t.tic();
        iter_arr  = krylovfunceigen.solve( b, inv_sqrt_f, A, bs,
                A.weights(),  eps, 1., "universal", res_fac);
        t.toc();
        time = t.diff();
        dg::blas1::axpby(1.0, b, -1.0, b_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac));
        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter_arr << "\n";
    }
    //sqrt invert schemes
    {
        std::cout << "\nM-CG+Cauchy:\n";
        dg::mat::KrylovSqrtCauchy<Container>
            krylovsqrtcauchy(A, -1, w2d, EVmin, EVmax, max_iterC, eps, max_iter);
        t.tic();
        iter_arr = krylovsqrtcauchy( b_exac, x);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        std::cout << "    time: "<<t.diff()<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter_arr << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<max_iterC << "\n";
    }
    {
        std::cout << "\nM-CG+ODE:\n";
        dg::mat::KrylovSqrtODE<Container> krylovsqrtode(A, -1, w2d,
                "Dormand-Prince-7-4-5",  epsTrel, epsTabs, max_iter, eps);
        t.tic();
        auto iterA= krylovsqrtode( b_exac, x);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        std::cout << "    time: "<<t.diff()<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iterA[0] << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<iterA[1] << "\n";
    }
    {
        std::cout << "\nM-CG+EIGEN:\n";
        //EVs of inverse Helmholtz
        double res_fac = kappa*sqrt(EVmin);

        dg::mat::MCGFuncEigen< Container> krylovfunceigen( x, max_iter);
        t.tic();
        auto sqrt_inv_f =  [](double x) { return 1./sqrt(x);};
        iter_arr  = krylovfunceigen( x, sqrt_inv_f, A, b_exac,
                A.weights(),  eps, 1., res_fac);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        std::cout << "    time: "<<t.diff()<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<iter_arr  << "\n";
    }
    //////////////////////Direct sqrt ODE solve
    {
        std::cout << "\nODE-Inv:\n";
        auto inv_sqrt = dg::mat::make_inv_sqrtodeCG(A, A.precond(),
                A.weights(), epsCG);
        auto directsqrtodesolve = dg::mat::make_directODESolve( inv_sqrt,
                "Dormand-Prince-7-4-5", epsTrel, epsTabs, iter_arr);
        t.tic();
        dg::blas2::symv( directsqrtodesolve, b_exac, x);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        double time = t.diff();
        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iterT: "<<std::setw(3)<<iter_arr << "\n";

        std::cout << "\nODE-Inv+A:\n";
        t.tic();
        dg::blas2::symv( A, x, bs);
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));
        time+=t.diff();
        std::cout << "    time: "<<time<<"s \n";
        std::cout << "    error: "<<erel  << "\n";
        std::cout << "    iter: "<<std::setw(3)<<1 << "\n";

    }

    return 0;
}
