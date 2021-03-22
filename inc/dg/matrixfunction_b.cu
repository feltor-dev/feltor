// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "blas.h"
#include "backend/typedefs.h"
#include "topology/evaluation.h"
#include "backend/timer.h"
#include "helmholtz.h"
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
    const Container v2d = dg::create::inv_weights( g);    
    double max_weights =   dg::blas1::reduce(w2d, 0., dg::AbsMax<double>() );
    double min_weights =  -dg::blas1::reduce(w2d, max_weights, dg::AbsMin<double>() );
    std::cout << "#   min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
    double hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());

    dg::SQRT<double> sqrt_f;
    dg::EXP<double> exp_f;
    dg::BESSELI0<double> besseli0_f;
    dg::GAMMA0<double> gamma0_f;
    auto func = std::make_tuple(sqrt_f, exp_f, besseli0_f, gamma0_f); 
    
    {
        std::cout << "\n#Compute (1+ alpha Delta) x = b " << std::endl;
        dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( g, alpha, dg::centered); //not_normed

        double EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0);
        double EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny())); //EVs of helmholtz
    
        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), b_exac(x), error(x);
        dg::blas1::scal(b_exac, std::get<0>(func)(helm_fac));

        double kappa = std::get<0>(func)(max_weights/min_weights); //condition number 
        double res_fac = kappa*std::get<0>(func)(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
               
        std::cout << "SQRT (M-Lanczos+Eigen):\n";
        dg::KrylovFuncEigenSolve<Container> krylovfunceigensolve( x,   max_iter);
        t.tic();
        iter = krylovfunceigensolve(x, b, std::get<0>(func), A, A.inv_weights(), A.weights(),  eps, res_fac); 
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, b, -1.0, b_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac));   

        std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iter: "<<std::setw(3)<<iter << "\n"; 
        
    }
    {
        std::cout << "\n#Compute  x = (1+ alpha Delta)^(-1) b " << std::endl;
        dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( g, alpha, dg::centered); //not_normed
        //EVs of inverse helmholtz
        double EVmin = 1./(1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny())));
        double EVmax = 1./(1.-A.alpha()*hxhy*(1.0 + 1.0));
    
        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), b_exac(x), error(x);
        dg::blas1::scal(b_exac, std::get<0>(func)(helm_fac));

        double kappa = std::get<0>(func)(max_weights/min_weights); //condition number 
        double res_fac = kappa*std::get<0>(func)(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
               
        std::cout << "SQRT (M-CG+Eigen):\n";
        dg::KrylovFuncEigenInvert<Container> krylovfunceigeninvert( x,   max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, b_exac, std::get<0>(func), A, A.inv_weights(), A.weights(),  eps, res_fac); 
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

        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( g, g.bcx(), g.bcy(), dg::not_normed, dg::centered); //negative laplace
        Container chi = dg::evaluate(dg::one, g);
        dg::blas1::scal(chi, -alpha); //-alpha must be positive for SPD operator!
        A.set_chi(chi); 
        //EVs of inverse elliptic
        double EVmin = 1./(-alpha*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()))); 
        double EVmax = 1./(-alpha*hxhy*(1.0 + 1.0)); 

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), b_exac(x), error(x);
        dg::blas1::scal(b_exac, std::get<1>(func)(ell_fac));

        double kappa = std::get<1>(func)(max_weights/min_weights); //condition number 
        double res_fac = kappa*std::get<1>(func)(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
        
        std::cout << "EXP (M-CG+Eigen):\n";
        dg::KrylovFuncEigenInvert<Container> krylovfunceigeninvert( x,   max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, b_exac, std::get<1>(func), A, A.inv_weights(), A.weights(),  eps, res_fac); 
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
        
        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( g, g.bcx(), g.bcy(), dg::not_normed, dg::centered); //negative laplace
        Container chi = dg::evaluate(dg::one, g);
        dg::blas1::scal(chi, -alpha); //-alpha must be positive for SPD operator!
        A.set_chi(chi); 
        //EVs of inverse elliptic
        double EVmin = 1./(-alpha*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()))); 
        double EVmax = 1./(-alpha*hxhy*(1.0 + 1.0));

    
        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), b_exac(x), error(x);
        dg::blas1::scal(b_exac, std::get<2>(func)(ell_fac));

        double kappa = std::get<2>(func)(max_weights/min_weights); //condition number 
        double res_fac = kappa*std::get<2>(func)(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
        
        std::cout << "BESSELI0 (M-CG+Eigen):\n";
        dg::KrylovFuncEigenInvert<Container> krylovfunceigeninvert( x,   max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, b_exac, std::get<2>(func), A, A.inv_weights(), A.weights(),  eps, res_fac); 
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

        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( g, g.bcx(), g.bcy(), dg::not_normed, dg::centered); //negative laplace
        Container chi = dg::evaluate(dg::one, g);
        dg::blas1::scal(chi, -alpha); //-alpha must be positive for SPD operator!
        A.set_chi(chi); 
        //EVs of inverse elliptic
        double EVmin = 1./(-alpha*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()))); 
        double EVmax = 1./(-alpha*hxhy*(1.0 + 1.0));

    
        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), b_exac(x), error(x);
        dg::blas1::scal(b_exac, std::get<3>(func)(ell_fac));

        double kappa = std::get<3>(func)(max_weights/min_weights); //condition number 
        double res_fac = kappa*std::get<3>(func)(EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
        
        std::cout << "GAMMA0 (M-CG+Eigen):\n";
        dg::KrylovFuncEigenInvert<Container> krylovfunceigeninvert( x,   max_iter);
        t.tic();
        iter = krylovfunceigeninvert(x, b_exac, std::get<3>(func), A, A.inv_weights(), A.weights(),  eps, res_fac); 
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
