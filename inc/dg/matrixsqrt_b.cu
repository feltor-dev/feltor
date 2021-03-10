#define SILENT
// #undef DG_BENCHMARK
// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "blas.h"
#include "backend/typedefs.h"
#include "topology/evaluation.h"
#include "backend/timer.h"
#include "matrixsqrt.h"
#include "eve.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -1.;
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
    double epsTrel = 1e-9;
    double epsTabs = 1e-12;
    unsigned counter = 0;
    std::cout << "# Type epsilon for CG (1e-8), and eps_rel (1e-9) and eps_abs (1e-12) for TimeStepper\n";
    std::cin >> epsCG >> epsTrel >> epsTabs;
    unsigned max_iter = 1;
    unsigned max_iterC = 1;
    std::cout << "# Type max_iter of tridiagonalization (500) and of Cauchy integral (40) ?\n";
    std::cin >> max_iter >> max_iterC;    
    std::cout << "# Type in eps of tridiagonalization (1e-7)\n";
    double eps = 1e-7; //# of pcg iter increases very much if
    std::cin >> eps;
    std::cout <<"# You typed\n"
              <<"epsCG:  "<<epsCG<<"\n"
              <<"epsTrel: "<<epsTrel<<"\n"
              <<"epsTabs: "<<epsTabs<<"\n"
              <<"max_iter: "<<max_iter<<"\n"
              <<"max_iterC: "<<max_iterC<<"\n"
              <<"eps: "<<eps<<std::endl;
    
    
    double erel = 0;

    dg::Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
   //start and end vectors
    Container x = dg::evaluate(lhs, grid);
    Container x_exac = dg::evaluate(lhs, grid);
    Container b = dg::evaluate(rhsHelmholtzsqrt, grid), b_exac(b), error(b_exac);
    Container bs = dg::evaluate(rhsHelmholtz, grid), bs_exac(bs);
    
    const Container w2d = dg::create::weights( grid);
    const Container v2d = dg::create::inv_weights( grid);

    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( grid, alpha, dg::centered); //not_normed
    dg::Invert<Container> invert( x, grid.size(), epsCG);

    ////////////////////////Direct Cauchy integral solve
    {
        std::cout << "\nCauchy: \n";
        dg::DirectSqrtCauchySolve<dg::CartesianGrid2d, Matrix, Container> directsqrtcauchysolve(A, grid, epsCG, max_iterC);

        t.tic();
        counter = directsqrtcauchysolve(b, bs);
        t.toc();   
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));  
        double time = t.diff();

        std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iterT: "<<std::setw(3)<<counter << "\n"; 

        std::cout << "\nCauchy+CG: \n";
        t.tic();
        unsigned number = invert(A,x,bs);
        t.toc();
        time+=t.diff();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iter: "<<std::setw(3)<<number << "\n";   
    }
    //////////////////////Direct sqrt ODE solve
    {
        std::cout << "\nODE:\n";
        dg::DirectSqrtODESolve<dg::CartesianGrid2d, Matrix, Container> directsqrtodesolve(A, grid, epsCG, epsTrel, epsTabs);
        t.tic();
        counter = directsqrtodesolve(b, bs);
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));
        double time = t.diff();
        std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iterT: "<<std::setw(3)<<counter << "\n"; 
        
        std::cout << "\nODE+CG:\n";
        t.tic();
        unsigned number = invert(A,x,bs);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac)); 
        time+=t.diff();
        std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iter: "<<std::setw(3)<<number << "\n";     
        
    }
    
    //////////////////Krylov solve via Lanczos method and Cauchy solve
    {
        std::cout << "\nM-Lanczos+Cauchy:\n";
        dg::KrylovSqrtCauchySolve<dg::CartesianGrid2d, Matrix, Container> krylovsqrtcauchysolve(A, grid, x,  epsCG, max_iter, max_iterC, eps);
        t.tic();
        counter = krylovsqrtcauchysolve(b, bs); 
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
        double time = t.diff();

//         std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iterT: "<<std::setw(3)<<counter << "\n"; 

        std::cout << "\nM-Lanczos+Cauchy+CG:\n";
        t.tic();
        unsigned number = invert(A,x,bs);
        t.toc();
        time += t.diff();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iter: "<<std::setw(3)<<number << "\n";   
    }
    
    //////////////////Krylov solve via Lanczos method and ODE sqrt solve
    {
        std::cout << "\nM-Lanczos+ODE:\n";  
        dg::KrylovSqrtODESolve<dg::CartesianGrid2d, Matrix, Container> krylovsqrtodesolve(A, grid, x,  epsCG, epsTrel, epsTabs, max_iter, eps);
        b = dg::evaluate(rhsHelmholtzsqrt, grid);
        t.tic();
        counter = krylovsqrtodesolve(b, bs); //overwrites b
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
        double time = t.diff();

//         std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iterT: "<<std::setw(3)<<counter << "\n"; 
        
        std::cout << "\nM-Lanczos+ODE+CG:\n";
        t.tic();
        unsigned number = invert(A,x,bs);
        t.toc();
        time += t.diff();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        std::cout << "    time: "<<time<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iterT: "<<std::setw(3)<<number << "\n"; 
    }
    //sqrt invert schemes
    {
        std::cout << "\nM-CG+Cauchy:\n";
        dg::blas1::scal(x, 0.0);  //must be initialized with zero
        dg::KrylovSqrtCauchyinvert<dg::CartesianGrid2d, Matrix,  Container> krylovsqrtcauchyinvert(A, grid, x,  epsCG, max_iter, max_iterC, eps);
        t.tic();
        counter = krylovsqrtcauchyinvert( x, b_exac);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
//         std::cout << "    time: "<<t.diff()<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iterT: "<<std::setw(3)<<counter << "\n"; 
    }
    {
        std::cout << "\nM-CG+ODE:\n";
        dg::blas1::scal(x, 0.0); //must be initialized with zero
        dg::KrylovSqrtODEinvert<dg::CartesianGrid2d, Matrix, Container> krylovsqrtodeinvert(A, grid, x,  epsCG, epsTrel, epsTabs, max_iter, eps);
        t.tic();
        counter = krylovsqrtodeinvert( x, b_exac);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
//         std::cout << "    time: "<<t.diff()<<"s \n"; 
        std::cout << "    error: "<<erel  << "\n"; 
        std::cout << "    iterT: "<<std::setw(3)<<counter << "\n"; 
    }


    return 0;
}
