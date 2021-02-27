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

using DiaMatrix = cusp::dia_matrix<int, double, cusp::device_memory>;
using CooMatrix = cusp::coo_matrix<int, double, cusp::device_memory>;
using Matrix = dg::DMatrix;
using Container = dg::DVec;
using SubContainer = dg::DVec;

// using DiaMatrix = cusp::dia_matrix<int, double, cusp::host_memory>;
// using CooMatrix = cusp::coo_matrix<int, double, cusp::host_memory>;
// using Matrix = dg::HMatrix;
// using Container = dg::HVec;
// using SubContainer = dg::HVec;

int main()
{
    dg::Timer t;
    
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
   //start and end vectors
    Container x = dg::evaluate(lhs, grid);
    Container x_exac = dg::evaluate(lhs, grid);
    Container b = dg::evaluate(rhsHelmholtzsqrt, grid), b_exac(b), error(b_exac);
    Container bs = dg::evaluate(rhsHelmholtz, grid), bs_exac(bs);
    
    const Container w2d = dg::create::weights( grid);
    const Container v2d = dg::create::inv_weights( grid);

    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( grid, alpha, dg::centered); //not_normed
    double epsCG, epsTimerel, epsTimeabs;    

    std::cout << "Type epsilon for CG (1e-5), and eps_rel (1e-5) and eps_abs (1e-10) for TimeStepper\n";
    std::cin >> epsCG >> epsTimerel >> epsTimeabs;
    epsCG = 1e-14;
    epsTimerel = 1e-13;
    epsTimeabs = 1e-14;
    int counter = 0;
    double erel = 0;
    unsigned iter = 1;
    unsigned iterCauchy = 1;
    
    std::cout << "# of Lanczos/CG iterations for tridiagonalization?\n";
    std::cin >> iter;    
    std::cout << "# of Cauchy terms?\n";
    std::cin >> iterCauchy;       
    std::cout << "Type in eps of CG/Lanczos for tridiagonalization\n";
    double eps = 1e-6; //# of pcg iterations increases very much if
    std::cin >> eps;
    
    dg::Invert<Container> invert( x, grid.size(), epsCG);

//   
//     ////////////////////////Direct Cauchy integral solve
//     {
//         std::cout << "Cauchy \n";
//         dg::DirectSqrtCauchySolve<dg::CartesianGrid2d, Matrix, Container> directsqrtcauchysolve(A, grid, epsCG, iterCauchy);
// 
//         t.tic();
//         directsqrtcauchysolve(b, bs);
//         t.toc();   
//         dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
//         erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
//         std::cout << "   Time: "<<t.diff()<<"s  Relative b error: "<<erel <<"\n";    
//     //     solve for x=\sqrt{A}^{-1} b'
//         t.tic();
//         invert(A,x,bs);
//         t.toc();
//         dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
//         erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
//         std::cout << "   Time: "<<t.diff()<<"s  Relative x error: "<<erel <<"\n";    
//     }
//     //////////////////////Direct sqrt ODE solve
//     {
//         std::cout << "ODE\n";
//         dg::DirectSqrtODESolve<dg::CartesianGrid2d, Matrix, Container> directsqrtodesolve(A, grid, epsCG, epsTimerel, epsTimeabs);
//         t.tic();
//         counter = directsqrtodesolve(b, bs);
//         t.toc();
// 
//         dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
//         erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
//         std::cout  << "   Time: "<<t.diff()<<"s  Relative b error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
//     //     solve for x=\sqrt{A}^{-1} b'
//         t.tic();
//         invert(A,x,bs);
//         t.toc();
//         dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
//         erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
//         std::cout << "   Time: "<<t.diff()<<"s  Relative x error: "<<erel <<"\n";  
//     }
    
    ////////////////////Krylov solve via Lanczos method and Cauchy solve
    {
        std::cout << "Lanczos + Cauchy ";
        dg::KrylovSqrtCauchySolve<dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> krylovsqrtcauchysolve(A, grid, x,  epsCG, iter, iterCauchy, eps);
        t.tic();
        krylovsqrtcauchysolve(b, bs); 
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
        std::cout << " Time: "<<t.diff()<<"s  Relative b error: "<<erel <<"\n";
    //     solve for x=\sqrt{A}^{-1} b'
        t.tic();
        invert(A,x,bs);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        std::cout << " Time: "<<t.diff()<<"s  Relative x error: "<<erel <<"\n";   
    }
    
    //////////////////Krylov solve via Lanczos method and ODE sqrt solve
    {
        std::cout << "Lanczos + ODE \n";  
        dg::KrylovSqrtODESolve<dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> krylovsqrtodesolve(A, grid, x,  epsCG, epsTimerel, epsTimeabs, iter, eps);
        b = dg::evaluate(rhsHelmholtzsqrt, grid);
        t.tic();
        counter = krylovsqrtodesolve(b, bs); //overwrites b
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
        std::cout  << " Time: "<<t.diff()<<"s  Relative b error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
    //     solve for x=\sqrt{A}^{-1} b'
        t.tic();
        invert(A,x,bs);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        std::cout << " Time: "<<t.diff()<<"s  Relative x error: "<<erel <<"\n"; 
    }
// 
//     
//     //sqrt invert schemes
//     {
//         std::cout << "CG + ODE\n";
//         dg::blas1::scal(x,0.0); //must be initialized with zero
//         dg::KrylovSqrtODEinvert<dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> krylovsqrtodeinvert(A, grid, x,  epsCG, epsTimerel, epsTimeabs, iter, eps);
//         t.tic();
//         counter = krylovsqrtodeinvert( x, b_exac);
//         t.toc();
//         dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
//         erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
//         std::cout << "   Time: "<<t.diff()<<"s  Relative x error: "<<erel <<"  Iterations: "<<std::setw(3)<<counter << "\n"; 
//     }
// 
//     {
//         std::cout << "CG + Cauchy\n";
//         dg::blas1::scal(x, 0.0);  //must be initialized with zero
//         dg::KrylovSqrtCauchyinvert<dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> krylovsqrtcauchyinvert(A, grid, x,  epsCG, iter, iterCauchy, eps);
//         t.tic();
//         counter = krylovsqrtcauchyinvert( x, b_exac);
//         t.toc();
//         dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
//         erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
//         std::cout << "   Time: "<<t.diff()<<"s  Relative x error: "<<erel <<"  Iterations: "<<std::setw(3)<<counter << "\n"; 
//     }
    return 0;
}
