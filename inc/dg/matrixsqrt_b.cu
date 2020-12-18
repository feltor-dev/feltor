#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#define SILENT
#include <boost/math/special_functions/jacobi_elliptic.hpp>

#include <iostream>
#include <iomanip>

#include "blas.h"
#include "backend/typedefs.h"
#include "topology/evaluation.h"
#include "adaptive.h"
#include "helmholtz.h"
#include "backend/timer.h"

// #include "cg.h"
#include "lanczos.h"
#include "sqrt_cauchy.h"
#include "matrixsqrt.h"
#include "eve.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -0.5;

double lhs( double x, double y){ return sin(x)*sin(y);}
double rhsHelmholtz( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
double rhsHelmholtzsqrt( double x, double y){ return sqrt(1.-2.*alpha)*sin(x)*sin(y);}

using DiaMatrix =  cusp::dia_matrix<int, double, cusp::host_memory>;
using CooMatrix =  cusp::coo_matrix<int, double, cusp::host_memory>;
int main()
{
    
    dg::Timer t;

    std::cout << "Test program for A^(1/2) x computation \n";
    
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
   //start and end vectors
    dg::HVec x = dg::evaluate(lhs, grid);
    dg::HVec x_exac = dg::evaluate(lhs, grid);
    dg::HVec b = dg::evaluate(rhsHelmholtzsqrt, grid), b_exac(b), error(b_exac);
    dg::HVec bs = dg::evaluate(rhsHelmholtz, grid), bs_exac(bs);

    const dg::HVec w2d = dg::create::weights( grid);
    const dg::HVec v2d = dg::create::inv_weights( grid);

    dg::Helmholtz<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> A( grid, alpha, dg::centered); //not_normed
    double epsCG, epsTimerel, epsTimeabs;
    

//     std::cout << "Type epsilon for CG (1e-5), and eps_rel (1e-5) and eps_abs (1e-10) for TimeStepper\n";
//     std::cin >> epsCG >> epsTimerel >> epsTimeabs;
    epsCG=1e-14;
    epsTimerel=1e-4;
    epsTimeabs=1e-14;
    int counter = 0;
    double erel = 0;
    unsigned iter = 1;
    unsigned iterCauchy = 1;
    
    dg::Invert<dg::HVec> invert( x, grid.size(), epsCG);


    //////////////////////////Direct Cauchy integral solve
    std::cout << "Solving  via Cauchy integral\n";
    CauchySqrtInt<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> cauchysqrtint(A, grid, epsCG);
    dg::EVE<dg::HVec> eve(b, 100);
    std::cout << "# of Cauchy terms?\n";
    std::cin >> iter;
//     double lambda_min = 1; //Exact estimate missing, However as long as chi in helmholtz is 1 it is correct
    double lambda_max;
    t.tic();
    eve(A, bs, bs, A.inv_weights(),lambda_max);
    std::cout << "Maximum EV from EVE is: "<< lambda_max << "\n";
    
    //analyitcal estimate
    double lmin = 1+1, lmax = n*n*Nx*Nx + n*n*Ny*Ny; //Eigenvalues of Laplace
    double hxhy = lx*ly/(n*n*Nx*Ny);
    lmin *= hxhy, lmax *= hxhy; //we multiplied the matrix by w2d
    std::cout << "Min and Maximum EV is: "<< -lmin*alpha+1 << "  "<<-lmax*alpha+1<< "\n";
   
    cauchysqrtint(b, bs,-lmin*alpha+1 ,-lmax*alpha+1, iter);
    t.toc();
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much smaller after a few iterations with correct EVs, reason is most likely that the EVs are not exactly estimated, error is also very sensible to min and max EVs
    
    //solve for x=\sqrt{A}^{-1} b'
    t.tic();
    invert(A,x,bs);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
    std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much 
    
    //////////////////////////Direct sqrt ODE solve
    std::cout << "Solving  via Direct sqrt ODE\n";
    DirectSqrtODESolve<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> directsqrtodesolve(A, grid, epsCG, epsTimerel, epsTimeabs);
    t.tic();
    counter = directsqrtodesolve(b, bs); //overwrites b
    t.toc();

    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout  << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
    //solve for x=\sqrt{A}^{-1} b'
    t.tic();
    invert(A,x,bs);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
    std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much 


        ////////////////////////Krylov solve via Lanczos method and ODE sqrt solve
    std::cout << "Solving  via Krylov method and Cauchy ODE\n";
    std::cout << "# of Lanczos iterations and Cauchy terms?\n";
    std::cin >> iter >> iterCauchy;
  
    KrylovSqrtCauchySolve<dg::CartesianGrid2d, dg::HMatrix, DiaMatrix, CooMatrix, dg::HVec> krylovsqrtcauchysolve(A, grid, x,  epsCG, iter);
    t.tic();
    krylovsqrtcauchysolve(b, bs, 5); //overwrites b
    t.toc();
    
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";
    
    //solve for x=\sqrt{A}^{-1} b'
    t.tic();
    invert(A,x,bs);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
    std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much 
    
    ////////////////////////Krylov solve via Lanczos method and ODE sqrt solve
    std::cout << "Solving  via Krylov method and sqrt ODE\n";
    std::cout << "# of Lanczos iterations?\n";
    std::cin >> iter;
  
    KrylovSqrtODESolve<dg::CartesianGrid2d, dg::HMatrix, DiaMatrix, CooMatrix, dg::HVec> krylovsqrtodesolve(A, grid, x,  epsCG, epsTimerel, epsTimeabs, iter);
    t.tic();
    counter = krylovsqrtodesolve(b, bs); //overwrites b
    t.toc();
    
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout  << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
   
    //solve for x=\sqrt{A}^{-1} b'
    t.tic();
    invert(A,x,bs);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
    std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much 
    
    
    //Direct CG sqrt solve TODO
    
    return 0;
}
