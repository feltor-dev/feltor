#include <iostream>
#include <iomanip>

#include "blas.h"
#include "backend/typedefs.h"
#include "topology/evaluation.h"
#include "adaptive.h"
#include "helmholtz.h"
#include "backend/timer.h"

#include "lanczos.h"

#include "matrixsqrt.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -0.5;

double lhs( double x, double y){ return sin(x)*sin(y);}
// double rhsHelmholtz( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
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
    dg::HVec x = dg::evaluate(lhs, grid), b(x), b_exac(x), error(b_exac);
    const dg::HVec w2d = dg::create::weights( grid);
    const dg::HVec v2d = dg::create::inv_weights( grid);
    b_exac = dg::evaluate(rhsHelmholtzsqrt, grid);

    dg::Helmholtz<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> A( grid, alpha, dg::centered); //not_normed
    double epsCG, epsTimerel, epsTimeabs;
//     std::cout << "Type epsilon for CG (1e-5), and eps_rel (1e-5) and eps_abs (1e-10) for TimeStepper\n";
//     std::cin >> epsCG >> epsTimerel >> epsTimeabs;
    epsCG=1e-8;
    epsTimerel=1e-8;
    epsTimeabs=1e-12;
    int counter =0;
    double erel=0;
    
    //////////////////////////Direct sqrt ODE solve
    std::cout << "Solving  via Direct sqrt ODE\n";
    DirectSqrtSolve<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> directsqrtsolve(A, grid, epsCG, epsTimerel, epsTimeabs);
    t.tic();
    counter = directsqrtsolve(x, b); //overwrites b
    t.toc();

    dg::blas1::axpby(1.0, b, -1.0, b_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac));   
    std::cout << "Time steps: "<<std::setw(6)<<counter  << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    

    ////////////////////////Krylov solve via Lanczos method and ODE sqrt solve
    std::cout << "Solving  via Krylov method and sqrt ODE\n";
    unsigned iter;
    std::cout << "# of Lanczos iterations?\n";
    std::cin >> iter;
  
    KrylovSqrtSolve<dg::CartesianGrid2d, dg::HMatrix, DiaMatrix, CooMatrix, dg::HVec> krylovsqrtsolve(A, grid, x,  epsCG, epsTimerel, epsTimeabs, iter);
    t.tic();
    counter = krylovsqrtsolve(x, b); //overwrites b
    t.toc();
    
    dg::blas1::axpby(1.0, b, -1.0, b_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac));   
    std::cout << "Time steps: "<<std::setw(6)<<counter  << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";
   
    return 0;
}
