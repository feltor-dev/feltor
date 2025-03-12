#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"


#include "lanczos.h"
#include "mcg.h"
#include "matrixfunction.h"
#include "matrixsqrt.h"

#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/lapack/lapack.h>

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
// const double m=3./2.;
// const double n=4.;
// const double m=1./2.;
// const double n=1.;
const double m=1./2.;
const double n=1.;
const double ms=1./2.;
const double ns=2.;
const double alpha = 1./2.;
const double ell_fac = (m*m+n*n);
const double ell_facs = (ms*ms+ns*ns);

const double amp=10.0;
const double bgamp=1.0;

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}
double lhss( double x, double y){ return sin(x*ms)*sin(y*ns);}
double sin2( double x, double y){ return amp*sin(x*m)*sin(y*n)*sin(x*m)*sin(y*n);}
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
    Container w2d_scaled(w2d);

    double max_weights = dg::blas1::reduce(w2d, 0., dg::AbsMax<double>() );
    double min_weights = dg::blas1::reduce(w2d, max_weights, dg::AbsMin<double>() );
    std::cout << "#   min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
    dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( {g, dg::centered, 1.0});

    std::vector<std::string> outs = {
            "K_0(-alpha A)",
            "K_0(d, -alpha A)",
            "K_0(-alpha A, d)",
    };

    for( unsigned u=0; u<outs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        Container b_h(b);
        Container one = dg::evaluate(dg::ONE(), g);

        //note that d must fulfill boundary conditions and should be positive definite!
        Container d = dg::evaluate(dg::Cauchy(lx/2., ly/2., 3., 3., amp), g);
        //add constant background field to d
        dg::blas1::plus(d, bgamp);

        std::cout << outs[u] << ":\n";

        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);
        dg::mat::UniversalLanczos<Container> krylovfunceigend( x, max_iter);
        dg::mat::ProductMatrixFunction<Container> krylovproduct( x, max_iter);


        dg::mat::GyrolagK<double> func(0, -alpha);
        auto funcE1 = dg::mat::make_FuncEigen_Te1( func);
        double time = 0.;
        unsigned iter_sum=0;

        //MLanczos-universal
        if (u==0)
        {
            t.tic();
            iter= krylovfunceigen.solve(x, funcE1, A, b, w2d, eps, 1., "universal");
            t.toc();
            time = t.diff();
        }
        if (u==1)
        {
            t.tic();
            iter = krylovproduct.apply( x, func, d, A, b, w2d, eps, 1.);
            t.toc();
            time = t.diff();
        }
        if (u==2)
        {
            t.tic();
            iter = krylovproduct.apply_adjoint( x, func, A, d, b, w2d, eps, 1.);
            t.toc();
            time = t.diff();
        }
        //Compute errors
        if (u==0)
        {
            dg::blas1::scal(x_exac, func(ell_fac));
        }
        else
        {
            Container fd(d); // helper variable
            //Compute absolute and relative error in adjointness
            if (u==2 )
            {
                x_h = dg::evaluate(lhss, g); // -> g
                dg::blas1::axpby(ell_facs, d, 0.0, fd);
                dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
                dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(ms^2+ns^2) d) sin(x*ms) cos(y*ms) \equiv exp(d,-alpha A) g
                x_h = dg::evaluate(lhs, g); // -> f
                double fOg = dg::blas2::dot( x_h, w2d, x_exac); //<f,exp(d,-alpha A) g>
                std::cout << "#    <f, exp(d,-alpha A) g> = " << fOg << std::endl;
                x_h = dg::evaluate(lhss, g); // -> g
                double gOadjf = dg::blas2::dot( x, w2d, x_h); //<exp(-alpha A, d)f, g>
                std::cout << "#    <exp(-alpha A, d)f, g> = " << gOadjf << std::endl;

                double eabs_adj = fOg-gOadjf; // <f,exp(d,-alpha A) g> -<exp(-alpha A, d)f, g>
                std::cout << "#    Errors in adjointness"<< "\n";
                std::cout << "#    universal-abserror: "<< eabs_adj  << "\n";
                std::cout << "    universal-error: "<< eabs_adj/fOg  << "\n";
            }
            //Compute exact error for product exponential (is used also for adjoint product exponential since we have no analytical solution there)
            x_h = dg::evaluate(lhs, g);
            dg::blas1::axpby(ell_fac, d, 0.0, fd);
            dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
            dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(m^2+n^2) d) sin(m x) cos(n y)
        }
        std::cout << "    universal-time: "<<time<<"s \n";
        if (u==0 || u==1) {
            dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
            erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
            std::cout << "    universal-error: "<<erel  << "\n";
            std::cout << "    universal-iter: " <<std::setw(3)<< iter << "\n";
        }
        else std::cout << "    universal-iter: " <<std::setw(3)<< iter_sum << "\n";
    }

    return 0;
}
