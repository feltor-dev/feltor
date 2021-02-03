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
const double alpha = -0.5;

double lhs( double x, double y){ return sin(x)*sin(y);}
double rhsHelmholtz( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
double rhsHelmholtzsqrt( double x, double y){ return sqrt(1.-2.*alpha)*sin(x)*sin(y);}

double lhs2( double x, double y){ return sin(x)*sin(4.*y);}
double rhsHelmholtz2( double x, double y){ return (1.-17.*alpha)*sin(x)*sin(4.*y);}
double rhsHelmholtzsqrt2( double x, double y){ return sqrt(1.-17.*alpha)*sin(x)*sin(4.*y);}

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
    std::cout << "Type in eps of CGsqrt and Lanczos\n";
    double eps = 1e-6; //# of pcg iterations increases very much if
    std::cin >> eps;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
   //start and end vectors
    Container x = dg::evaluate(lhs, grid);
    Container x_exac = dg::evaluate(lhs, grid);
    Container b = dg::evaluate(rhsHelmholtzsqrt, grid), b_exac(b), error(b_exac);
    Container bs = dg::evaluate(rhsHelmholtz, grid), bs_exac(bs);
    
    Container x2 = dg::evaluate(lhs2, grid);
    Container x_exac2 = dg::evaluate(lhs2, grid);
    Container b2 = dg::evaluate(rhsHelmholtzsqrt2, grid), b_exac2(b2);
    Container bs2 = dg::evaluate(rhsHelmholtz2, grid), bs_exac2(bs2);

    const Container w2d = dg::create::weights( grid);
    const Container v2d = dg::create::inv_weights( grid);

    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( grid, alpha, dg::centered); //not_normed
    double epsCG, epsTimerel, epsTimeabs;    

//     std::cout << "Type epsilon for CG (1e-5), and eps_rel (1e-5) and eps_abs (1e-10) for TimeStepper\n";
//     std::cin >> epsCG >> epsTimerel >> epsTimeabs;
    epsCG=1e-14;
    epsTimerel=1e-8;
    epsTimeabs=1e-14;
    int counter = 0;
    double erel = 0;
    unsigned iter = 1;
    unsigned iterCauchy = 1;
    
    dg::Invert<Container> invert( x, grid.size(), epsCG);

  
    //////////////////////////Direct Cauchy integral solve
    std::cout << "Solving  via Cauchy integral\n";
//     CauchySqrtInt<dg::CartesianGrid2d, Matrix, Container> cauchysqrtint(A, grid, epsCG);
//     std::cout << "# of Cauchy terms?\n";
//     std::cin >> iterCauchy;
//     //analytical estimate
//     double lmin = 1+1, lmax = n*n*Nx*Nx + n*n*Ny*Ny; //Eigenvalues of Laplace
//     double hxhy = lx*ly/(n*n*Nx*Ny);
//     lmin *= hxhy, lmax *= hxhy; //we multiplied the matrix by w2d
//     std::cout << "Min and Maximum EV is: "<< -lmin*alpha+1 << "  "<<-lmax*alpha+1<< "\n";
//     t.tic();
//     cauchysqrtint(b, bs,-lmin*alpha+1 ,-lmax*alpha+1, iterCauchy);
//     t.toc();
    DirectSqrtCauchySolve<dg::CartesianGrid2d, Matrix, Container> directsqrtcauchysolve(A, grid, epsCG);
    std::cout << "# of Cauchy terms?\n";
    std::cin >> iterCauchy;   
    t.tic();
    directsqrtcauchysolve(b, bs, iterCauchy);
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
    DirectSqrtODESolve<dg::CartesianGrid2d, Matrix, Container> directsqrtodesolve(A, grid, epsCG, epsTimerel, epsTimeabs);
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

    //////////////////////Krylov solve via Lanczos method and Cauchy solve
    std::cout << "Solving  via Krylov method and Cauchy\n";
    std::cout << "# of Lanczos iterations and Cauchy terms?\n";
    std::cin >> iter >> iterCauchy;
  
    KrylovSqrtCauchySolve<dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> krylovsqrtcauchysolve(A, grid, x,  epsCG, iter,eps);
    t.tic();
    krylovsqrtcauchysolve(b, bs, iterCauchy); //overwrites b
    t.toc();
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";
    
//     t.tic();
//     b = dg::evaluate(rhsHelmholtzsqrt, grid);
//     krylovsqrtcauchysolve(b, bs, iterCauchy); //overwrites b
//     t.toc(); 
//     dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
//     erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
//     std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";
//     
//     t.tic();
//     krylovsqrtcauchysolve(x, b, iterCauchy); //overwrites b
//     t.toc(); 
//     dg::blas1::axpby(1.0, b, -1.0, b_exac, error);
//     erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac));   
//     std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    
    
    // solve for x=\sqrt{A}^{-1} b'
    t.tic();
    invert(A,x,bs);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much 
//    
    ////////////////////Krylov solve via Lanczos method and ODE sqrt solve
    std::cout << "Solving  via Krylov method and sqrt ODE\n";
    std::cout << "# of Lanczos iterations?\n";
    std::cin >> iter;
  
    KrylovSqrtODESolve<dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> krylovsqrtodesolve(A, grid, x,  epsCG, epsTimerel, epsTimeabs, iter, eps);
    b = dg::evaluate(rhsHelmholtzsqrt, grid);
    t.tic();
    counter = krylovsqrtodesolve(b, bs); //overwrites b
    t.toc();
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout  << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 

//     b = dg::evaluate(rhsHelmholtzsqrt, grid);
//     t.tic();
//     counter = krylovsqrtodesolve(b, bs); //overwrites b
//     t.toc();
//     dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
//     erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
//     std::cout  << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
//     
//     t.tic();
// //     b = dg::evaluate(rhsHelmholtzsqrt, grid);
//     counter = krylovsqrtodesolve(x2, b2); //overwrites b
//     t.toc();
//     dg::blas1::axpby(1.0, b2, -1.0, b_exac2, error);
//     erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac2));   
//     std::cout  << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
    
    //solve for x=\sqrt{A}^{-1} b'
    t.tic();
    invert(A,x,bs);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much 
    
    
    //Direct CG sqrt solve
    std::cout << "Solving  via CG method and sqrt ODE\n";
    dg::blas1::scal(x,0.0);

    CGsqrt<Container, SubContainer, DiaMatrix, CooMatrix> cgsqrt(x,1000);
    dg::blas2::symv(w2d, b, b);
    t.tic();
    counter = cgsqrt(A, x, b, v2d, v2d, eps,1.);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
    std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Iterations: "<<std::setw(3)<<counter << "\n"; 

    return 0;
}
