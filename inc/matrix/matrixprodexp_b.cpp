#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"


#include "lanczos.h"
#include "mcg.h"
#include "matrixfunction.h"
#include "matrixprod.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
// const double m=3./2.;
// const double n=4.;
// const double m=1./2.;
// const double n=1.;
const double ms=3./2.;
const double ns=2.;
const double ms_s=1./2.;
const double ns_s=2.;
const double alpha = 1./2.;
const double ell_fac = (ms*ms+ns*ns);
const double ell_facs = (ms_s*ms_s+ns_s*ns_s);
const double sigmax=2.;
const double sigmay=3.;
const double amp=10.0;
const double bgamp=1.0;

double lhs( double x, double y){ return sin(x*ms)*sin(y*ns);}
double lhss( double x, double y){ return sin(x*ms_s)*sin(y*ns_s);}

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
    const dg::DVec w2d = dg::create::weights( g);
    dg::DVec w2d_scaled(w2d);

    double max_weights = dg::blas1::reduce(w2d, 0., dg::AbsMax<double>() );
    double min_weights = dg::blas1::reduce(w2d, max_weights, dg::AbsMin<double>() );
    std::cout << "#   min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> A( {g, dg::centered, 1.0});

    std::vector<std::string> outs = {
            "K_0(-alpha A)",    // UniversalLanczos
            "K_0(d, -alpha A)", // ProductMatrixFunction
            "K_0(-alpha A, d)", // ProductMatrixFunction
            "K_0(-A, alpha d)"  // CauchyMatrixProductAdj
    };
    dg::mat::UniversalLanczos<dg::DVec> krylovfunceigen( w2d, max_iter);
    dg::mat::ProductMatrixFunction<dg::DVec> krylovproduct( w2d, max_iter);
    unsigned num_stages = 3;
    dg::mat::CauchyMatrixProductAdj<dg::CartesianGrid2d, dg::DMatrix, dg::cDVec> cauchy( 1e-5, g, num_stages);
    std::vector<dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec, dg::cDVec>> multipol( num_stages);
    for( unsigned u=0; u<num_stages; u++)
        multipol[u].construct( cauchy.multigrid().grid(u), dg::centered, 1.0);



    for( unsigned u=0; u<outs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        dg::DVec x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        dg::DVec b_h(b);
        dg::DVec one = dg::evaluate(dg::ONE(), g);

        //note that d must fulfill boundary conditions and should be positive definite!
        dg::DVec d = dg::evaluate(dg::Cauchy(lx/2., ly/2., sigmax, sigmay, amp), g);
        //add constant background field to d
        dg::blas1::plus(d, bgamp);

        std::cout << outs[u] << ":\n";



        dg::mat::GyrolagK<double> func(0, -alpha);
        auto funcE1 = dg::mat::make_FuncEigen_Te1( func);
        double time = 0.;

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
            //iter = krylovproduct.apply( x, func, d, A, b, w2d, eps, 1.);
            double max = dg::blas1::reduce( d, -1e308, thrust::maximum<double>());
            auto unary_func = dg::mat::make_FuncEigen_Te1( [&](double x) {return func( max, x);});
            auto T = krylovproduct.lanczos().tridiag( unary_func, A, b, w2d, eps, 1.,
                "universal", 1.0, 1);
            iter = T.M.size();
            krylovproduct.compute_vlcl( func, d, A, T, x, b, krylovproduct.lanczos().get_bnorm());
            t.toc();
            time = t.diff();
        }
        if (u==2)
        {
            t.tic();
            //iter = krylovproduct.apply_adjoint( x, func, A, d, b, w2d, eps, 1.);
            double max = dg::blas1::reduce( d, -1e308, thrust::maximum<double>());
            auto unary_func = dg::mat::make_FuncEigen_Te1( [&](double x) {return func( x, max);});
            auto T = krylovproduct.lanczos().tridiag( unary_func, A, b, w2d, eps, 1.,
                "universal", 1.0, 1);
            iter = T.M.size();
            krylovproduct.compute_vlcl_adjoint( func, A, d, T, x, b,
                w2d, krylovproduct.lanczos().get_bnorm());
            t.toc();
            time = t.diff();
        }
        if( u == 3)
        {
            cauchy.set_verbose(true);
            t.tic();
            auto func = dg::mat::GyrolagK<thrust::complex<double>>(0,1);
            auto dxlnfunc = dg::mat::DLnGyrolagK<thrust::complex<double>>(0,1);
            cauchy.solve( x, func , dxlnfunc, alpha, multipol, d, b, std::vector<double>(num_stages, eps));
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
            dg::DVec fd(d); // helper variable
            //Compute absolute and relative error in adjointness
            if (u==2 || u == 3)
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
        }
        std::cout << "    universal-iter: " <<std::setw(3)<< iter << "\n";
    }

    return 0;
}
