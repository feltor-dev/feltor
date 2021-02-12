// #undef DG_BENCHMARK

#include <iostream>
#include <iomanip>
#include "backend/timer.h"
#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>
#include "lanczos.h"
#include "helmholtz.h"


const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -0.5;

const double m =4.;
const double n=4.;


double lhs( double x, double y) {return sin(m*x)*sin(n*y);}
double rhs( double x, double y){ return (1.-(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}

using DiaMatrix = cusp::dia_matrix<int, double, cusp::device_memory>;
using CooMatrix = cusp::coo_matrix<int, double, cusp::device_memory>;
using Matrix = dg::DMatrix;
using Container_type = dg::DVec;
using SubContainer_type = dg::DVec;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    unsigned max_iter;
    std::cout << "# of max_iterations\n"; 
    std::cin >> max_iter;
    std::cout << "Type in eps\n";
    double eps = 1e-6; 
    std::cin >> eps;
    dg::CartesianGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy);
    
    const Container_type w2d = dg::create::weights( grid);
    const Container_type v2d = dg::create::inv_weights( grid);
        
    Container_type x = dg::evaluate( lhs, grid), b(x), zero(x), one(x), bsymv(x), error(x), bsymv2(x), helper(x), xexac(x);
    Container_type bexac = dg::evaluate( rhs, grid);
    dg::blas1::scal(zero, 0.0);
    one = dg::evaluate(dg::one, grid);
    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container_type> A( grid, alpha, dg::centered); //not_normed
    
    //Create Lanczos class
    t.tic();
    dg::Lanczos< Container_type, SubContainer_type, DiaMatrix, CooMatrix > lanczos(x, max_iter);
    t.toc();
    std::cout << "Creation of Lanczos  took "<< t.diff()<<"s   \n";

    DiaMatrix T; 
    CooMatrix V, Vt;
    std::pair<DiaMatrix, CooMatrix> TVpair; 
    
    std::cout << "Computing with Lanczos method \n";
    t.tic();
    TVpair = lanczos(A, x, b); 
    dg::blas2::symv( v2d, b, b);     //normalize
    t.toc();
    T = TVpair.first; 
    V = TVpair.second;
    cusp::transpose(V, Vt);
    
    //Compute error with method 1
    dg::blas2::symv(A, x, helper);
    dg::blas2::symv( v2d, helper, bsymv); //normalize operator
    dg::blas1::axpby(-1.0, bsymv, 1.0, b,error);
    std::cout << "# of Lanczos Iterations: "<< lanczos.get_iter() <<" | time: "<< t.diff()<<"s \n";
    std::cout << "# Relative error between b=||x||_S S^{-1}V^T T e_1 and to b=S^{-1} A x: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bsymv)) << " \n";
    dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
    std::cout << "# Relative error between b=||x||_S S^{-1}V^T T e_1 and b: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";   
    //Compute error with method 2
    Container_type e1( lanczos.get_iter(), 0.), temp(e1);
    e1[0]=1.;  
//     dg::blas2::symv( w2d, x,x); //normalize
    dg::blas2::symv(Vt, x, e1); //V^T x
    dg::blas2::symv(T, e1, temp); //T V^T x
    dg::blas2::symv(V, temp, x); // V T V^T x
    dg::blas2::symv( v2d, x, b);     //normalize
    dg::blas1::axpby(-1.0, bsymv, 1.0, b,error);
    std::cout << "# Relative error between b=S^{-1}V T V^T x  and to b=S^{-1} A x:" << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bsymv)) << "\n";    
    dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
    std::cout << "# Relative error between b=S^{-1}V T V^T  x  and b: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";     


    std::cout << "\nComputing with M-Lanczos method \n";
    x = dg::evaluate( lhs, grid);
    dg::blas2::symv(A, x, helper);
    dg::blas2::symv( v2d, helper, bsymv); //normalize operator
    bexac= dg::evaluate( rhs, grid);

    t.tic();
    TVpair = lanczos(A, x, b, w2d, v2d, eps); 
    t.toc();
    T = TVpair.first; 
    V = TVpair.second;
    cusp::transpose(V, Vt);
    //Compute error with Method 1
    dg::blas1::axpby(-1.0, bsymv, 1.0, b,error);
    std::cout << "# of Lanczos Iterations: "<< lanczos.get_iter() <<" | time: "<< t.diff()<<"s \n";
    std::cout << "# Relative error between b=||x||_S V^T T e_1 and to b=S^{-1} A x: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bsymv)) << " \n";
    dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
    std::cout << "# Relative error between b=||x||_S V^T T e_1 and b: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";
    //Compute error with method 2
    e1.resize( lanczos.get_iter(), 0.), temp.resize( lanczos.get_iter(), 0.);
    e1[0]=1.;
    dg::blas2::symv( w2d, x,helper); //normalize
    dg::blas2::symv(Vt, helper, e1); //V^T x
    dg::blas2::symv(T, e1, temp); //T V^T x
    dg::blas2::symv(V, temp, b); // V T V^T x
    dg::blas1::axpby(-1.0, bsymv, 1.0, b,error);
    std::cout << "# Relative error between b=V T V^T S x  and to b=S^{-1} A x:" << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bsymv)) << "\n";    
    dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
    std::cout << "# Relative error between b=V T V^T S x  and b: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";  
    
    std::cout << "\nComputing with CG method \n";
    
    
    CooMatrix R, Tinv;
    std::pair<CooMatrix, CooMatrix> TinvRpair;
    
    dg::CGtridiag<Container_type, SubContainer_type, DiaMatrix, CooMatrix> cgtridiag(x, max_iter);
    dg::blas1::scal(x, 0.0);
    dg::blas2::symv(w2d, bexac, b); //multiply weights
    t.tic();
    TinvRpair = cgtridiag(A, x, b, v2d, w2d, eps, 1.); 
    t.toc();
    Tinv = TinvRpair.first; 
    R    = TinvRpair.second;

    dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
    std::cout << "# of CG Iterations: "<< cgtridiag.get_iter() <<" | time: "<< t.diff()<<"s \n";
    std::cout << "# Relative error between x= R T^{-1} e_1 and x: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";
   

    
    return 0;
}
