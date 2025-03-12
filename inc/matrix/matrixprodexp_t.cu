#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "lanczos.h"
#include "mcg.h"
#include "outer.h"
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
const double m=3./2.;
const double n=2.;
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

// compute y = delta*A*gamma*x
template<class ContainerType>
struct Wrapper
{
    template<class MatrixType>
    Wrapper( MatrixType& A, const ContainerType& gamma, const ContainerType& delta):
        m_gamma(gamma), m_delta( delta), m_temp(gamma){
        m_A = [&]( const ContainerType& x, ContainerType& y){
            return dg::apply( A, x, y);
        };
    }
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0 x, ContainerType1& y)
    {
        dg::blas1::pointwiseDot( m_gamma, x, m_temp);
        dg::apply( m_A, m_temp, y);
        dg::blas1::pointwiseDot( m_delta, y, y);
    }

    private:
    std::function< void( const ContainerType&, ContainerType&)> m_A;
    ContainerType m_gamma, m_delta, m_temp;

};

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
    const Container w2d = dg::create::weights( g);//=M

    dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( {g, dg::centered, 1.0});
    dg::mat::LaplaceDecomposition<Container> laplaceM{g, bcx, bcy, dg::centered, 1.0};

    std::vector<std::string> outs_k = {
        "K_0",
        //"K_1",
        //"K_2"
    };
    std::vector<std::string> outs = {
            "(-alpha A)",
            "(d, -alpha A)",
            "(-alpha A, d)",
//             "_naive(d, -alpha A)",
//             "_naive(-alpha A, d)",
            "(-alpha d A)",
            "(-alpha A d )"
    };
    //Plot into netcdf file
    dg::file::NcFile file( "visual11.nc", dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"}}, g.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"}}, g.abscissas(1));

//     std::string names[5] = {"K0","K0_prod","K0_prodadj","K0_prod_naive","K0_prodadj_naive"};
    std::string names[5] = {"K0","K0_prod","K0_prodadj","K0_app","K0_appadj"};

    for( unsigned k=0; k<outs_k.size(); k++)
    for( unsigned u=0; u<outs.size(); u++)
    {
        dg::mat::GyrolagK<double> func(k, -alpha);
        std::cout << "\n#Compute x = "<<outs_k[k]<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        Container b_h(b);
        Container one = dg::evaluate(dg::ONE(), g);
        
        //note that d must fulfill boundary conditions and and must be >0
        //initialize constant d
//         Container d = dg::evaluate(dg::ZERO(), g);   
        
        //initialize d = amp*(sin( x/2) sin(y))^2
//         Container d = dg::evaluate(sin2, g);

        //initialize d = heaviside bump function
        Container d = dg::evaluate(dg::Cauchy(lx/2., ly/2., 2., 3., amp), g);
//         b_h = dg::evaluate(dg::SinXSinY(amp, 0.0, 4.0, 4.0), g); //superimpose sinxsiny
//         dg::blas1::pointwiseDot(b_h,b_h,b_h);
//         dg::blas1::pointwiseDot(d,b_h,d);

        //add constant background field to d
        dg::blas1::plus(d, bgamp);

        Container w2d_AD = dg::create::weights( g);
        Container w2d_DA = dg::create::weights( g);
        dg::blas1::pointwiseDot( w2d, d, w2d_AD); //scale norm for A D self adjoint in the scaled norm M D , requires d\neq 0
        dg::blas1::pointwiseDivide( w2d, d, w2d_DA); //scale norm for D A self adjoint in the scaled norm M D^{-1}, requires d\neq 0


        std::cout << outs_k[k]<<outs[u] << ":\n";

        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);
        dg::mat::UniversalLanczos<Container> krylovfunceigend( x, max_iter);
        dg::mat::ProductMatrixFunction<Container> krylovproduct( x, max_iter);

        auto funcE1 = dg::mat::make_FuncEigen_Te1( func);
        unsigned iter_sum=0;
        double time = 0;

        //MLanczos-universal
        if (u==0)
        {
            t.tic();
            //iter= krylovfunceigen.solve(x, funcE1, A, b, w2d, eps, 1., "universal");
            iter= laplaceM.matrix_function(x, func, b, eps, 1.);
            
            t.toc();
            time = t.diff();
        }
        if (u==1)
        {
            t.tic();
            //iter = krylovproduct.apply( x, func, d, A, b, w2d, eps, 1.);
            iter= laplaceM.product_function(x, func, d, b, eps, 1.);
            t.toc();
            time = t.diff();
        }
        if (u==2)
        {
            t.tic();
            //iter = krylovproduct.apply_adjoint( x, func, A, d, b, w2d, eps, 1.);
            iter= laplaceM.product_function_adjoint(x, func, d, b, eps, 1.);
            t.toc();
            time = t.diff();
        }
//         if (u==3)
//         {
//             t.tic();
//             double lambda_d = 0.;
//             for( unsigned k=0; k<x.size(); k++)
//             {
//                 lambda_d = d[k];
//                 A.set_chi(lambda_d);
//                 iter = krylovfunceigen.solve(x_h, funcE1, A, b, w2d, eps, 1., "universal");
//                 iter_sum+=iter;
//                 x[k] = x_h[k];
//             }
//             t.toc();
//             time = t.diff();
//             A.set_chi(one);
//         }
//         if (u==4)
//         {
//             dg::blas1::scal(x, 0.0);
//             double lambda_d = 0.;
//             iter_sum=0;
//             Container fd(b); // helper variable
//             t.tic();
//             for( unsigned k=0; k<x.size(); k++)
//             {
//                 lambda_d = d[k];
//                 A.set_chi(lambda_d);
//                 dg::blas1::scal(b_h, 0.0);
//                 b_h[k] = b[k]; //instead of b
//                 iter = krylovfunceigen.solve(x_h, funcE1, A, b_h, w2d, eps, 1.0, "universal");
//                 iter_sum+=iter;
//                 dg::blas1::axpby(1.0, x_h, 1.0, x);
//             }
//             t.toc();m_precond
//             time = t.diff();
//            A.set_chi(one);
//         }
        if (u==3)
        {
            Wrapper<Container> wrap( A, one, d);
            t.tic();
            iter= krylovfunceigen.solve(x, funcE1, wrap, b, w2d_DA, eps, 1., "universal");
            t.toc();
            time = t.diff();
        }
        if (u==4)
        {
            Wrapper<Container> wrap( A, d, one);
            t.tic();
            iter= krylovfunceigen.solve(x, funcE1, wrap, b, w2d_AD, eps, 1., "universal");
            //weights of adjoint missing?
            t.toc();
            time = t.diff();
        }
        //write solution into file
        file.defput_var( names[u], {"y", "x"}, {}, {g}, x);
        //Compute errors
        if (u==0)
        {
            dg::blas1::scal(x_exac, func(ell_fac));
        }
        else
        {
            Container fd(d); // helper variable
            //Compute absolute and relative error in adjointness
            if (u==2 || u==4)
            {
                x_h = dg::evaluate(lhss, g); // -> g
                dg::blas1::axpby(ell_facs, d, 0.0, fd);
                dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
                dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(ms^2+ns^2) d) sin(x*ms) cos(y*ms) \equiv exp(d,-alpha A) g
                x_h = dg::evaluate(lhs, g); // -> f
                double fOg = dg::blas2::dot( x_h, w2d, x_exac); //<f,exp(d,-alpha A) g>
                std::cout << "    <f, exp(d,-alpha A) g> = " << fOg << std::endl;
                x_h = dg::evaluate(lhss, g); // -> g
                double gOadjf = dg::blas2::dot( x, w2d, x_h); //<exp(-alpha A, d)f, g>
                std::cout << "    <exp(-alpha A, d)f, g> = " << gOadjf << std::endl;

                double eabs_adj = fOg-gOadjf; // <f,exp(d,-alpha A) g> -<exp(-alpha A, d)f, g>
                std::cout << "    universal-abserror-adjointness: "<< eabs_adj  << "\n";
                std::cout << "    universal-relerror-adjointness: "<< eabs_adj/fOg  << "\n";
            }
            //Compute exact error for product exponential (is used also for adjoint product exponential since we have no analytical solution there)
            x_h = dg::evaluate(lhs, g);
            dg::blas1::axpby(ell_fac, d, 0.0, fd);
            dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
            dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(m^2+n^2) d) sin(m x) cos(n y)
        }
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        std::cout << "    universal-iter: "<<std::setw(3)<< iter << "\n";
        if (u==3 || u==4) std::cout << "    universal-iter_sum: "<<std::setw(3)<<iter_sum << "\n";
        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
    }
    file.close();

    return 0;
}
