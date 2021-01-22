#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#define SILENT
// #undef DG_BENCHMARK
// #define DG_DEBUG
#include <boost/math/special_functions/jacobi_elliptic.hpp>

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

using dia_type = cusp::dia_matrix<int, double, cusp::device_memory>;
using coo_type = cusp::coo_matrix<int, double, cusp::device_memory>;
using Mat_type = dg::DMatrix;
using Container_type = dg::DVec;
int main()
{
    dg::Timer t;
    std::cout << "Test program for A^(1/2) x computation \n";
    
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Type in eps of CGsqrt and Lanczos\n";
    double eps = 1e-6; //# of pcg iterations increases very much if
    std::cin >> eps;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
   //start and end vectors
    Container_type x = dg::evaluate(lhs, grid);
    Container_type x_exac = dg::evaluate(lhs, grid);
    Container_type b = dg::evaluate(rhsHelmholtzsqrt, grid), b_exac(b), error(b_exac);
    Container_type bs = dg::evaluate(rhsHelmholtz, grid), bs_exac(bs);
    
    Container_type x2 = dg::evaluate(lhs2, grid);
    Container_type x_exac2 = dg::evaluate(lhs2, grid);
    Container_type b2 = dg::evaluate(rhsHelmholtzsqrt2, grid), b_exac2(b2);
    Container_type bs2 = dg::evaluate(rhsHelmholtz2, grid), bs_exac2(bs2);

    const Container_type w2d = dg::create::weights( grid);
    const Container_type v2d = dg::create::inv_weights( grid);

    dg::Helmholtz<dg::CartesianGrid2d, Mat_type, Container_type> A( grid, alpha, dg::centered); //not_normed
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
    
    dg::Invert<Container_type> invert( x, grid.size(), epsCG);

  /*
    //////////////////////////Direct Cauchy integral solve
    std::cout << "Solving  via Cauchy integral\n";
    CauchySqrtInt<dg::CartesianGrid2d, Mat_type, Container_type> cauchysqrtint(A, grid, epsCG);
    dg::EVE<Container_type> eve(b, 100);
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
    DirectSqrtODESolve<dg::CartesianGrid2d, Mat_type, Container_type> directsqrtodesolve(A, grid, epsCG, epsTimerel, epsTimeabs);
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
    std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much */

    //////////////////////Krylov solve via Lanczos method and Cauchy solve
    std::cout << "Solving  via Krylov method and Cauchy\n";
    std::cout << "# of Lanczos iterations and Cauchy terms?\n";
    std::cin >> iter >> iterCauchy;
  
    KrylovSqrtCauchySolve<dg::CartesianGrid2d, Mat_type, dia_type, coo_type, Container_type> krylovsqrtcauchysolve(A, grid, x,  epsCG, iter,eps);
    t.tic();
    krylovsqrtcauchysolve(b, bs, 5); //overwrites b
    t.toc();
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";
    
    t.tic();
    b = dg::evaluate(rhsHelmholtzsqrt, grid);
    krylovsqrtcauchysolve(b, bs, 5); //overwrites b
    t.toc(); 
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";
    
    t.tic();
    krylovsqrtcauchysolve(x, b, 5); //overwrites b
    t.toc(); 
    dg::blas1::axpby(1.0, b, -1.0, b_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    
    
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
  
    KrylovSqrtODESolve<dg::CartesianGrid2d, Mat_type, dia_type, coo_type, Container_type> krylovsqrtodesolve(A, grid, x,  epsCG, epsTimerel, epsTimeabs, iter, eps);
    b = dg::evaluate(rhsHelmholtzsqrt, grid);
    t.tic();
    counter = krylovsqrtodesolve(b, bs); //overwrites b
    t.toc();
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout  << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 

    b = dg::evaluate(rhsHelmholtzsqrt, grid);
    t.tic();
    counter = krylovsqrtodesolve(b, bs); //overwrites b
    t.toc();
    dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
    std::cout  << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
    
    t.tic();
//     b = dg::evaluate(rhsHelmholtzsqrt, grid);
    counter = krylovsqrtodesolve(x2, b2); //overwrites b
    t.toc();
    dg::blas1::axpby(1.0, b2, -1.0, b_exac2, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b_exac2));   
    std::cout  << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Time steps: "<<std::setw(3)<<counter << "\n"; 
    
    //solve for x=\sqrt{A}^{-1} b'
    t.tic();
    invert(A,x,bs);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
    erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
    std::cout << " Time: "<<t.diff()<<"s  Relative error: "<<erel <<"\n";    //error should be much 
    
    
//     //Direct CG sqrt solve
//     std::cout << "Solving  via CG method and sqrt ODE\n";
//     dg::blas1::scal(x,0.0);
// 
//     CGsqrt<Container_type> cgsqrt(x,1000);
//     dg::blas2::symv(w2d, b, b);
//     t.tic();
//     counter = cgsqrt(A, x, b, v2d, v2d, eps,1.);
//     t.toc();
//     dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
//     erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
//     std::cout << "   Time: "<<t.diff()<<"s  Relative error: "<<erel <<"  Iterations: "<<std::setw(3)<<counter << "\n"; 

    return 0;
}
