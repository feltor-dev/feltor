// #define DG_DEBUG
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
const double n = 4.;

double lhs( double x, double y) {return sin(m*x)*sin(n*y);}
double rhs( double x, double y){ return (1.-(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}

using DiaMatrix = cusp::dia_matrix<int, double, cusp::host_memory>;
using CooMatrix = cusp::coo_matrix<int, double, cusp::host_memory>;
using Matrix = dg::DMatrix;
using Container = dg::DVec;
using SubContainer = dg::HVec;
// using DiaMatrix = cusp::dia_matrix<int, double, cusp::device_memory>;
// using CooMatrix = cusp::coo_matrix<int, double, cusp::device_memory>;
// using Matrix = dg::DMatrix;
// using Container = dg::DVec;
// using SubContainer = dg::DVec;
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
    
    const Container w2d = dg::create::weights( grid);
    const Container v2d = dg::create::inv_weights( grid);
        
    Container x = dg::evaluate( lhs, grid), b(x), zero(x), one(x), bsymv(x), error(x),  helper(x), xexac(x);
    Container bexac = dg::evaluate( rhs, grid);
    dg::blas1::scal(zero, 0.0);
    one = dg::evaluate(dg::one, grid);
    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( grid, alpha, dg::centered); //not_normed
    
    {
        std::cout << "Computing with Lanczos method \n";
        t.tic();
        dg::Lanczos< Container, SubContainer, DiaMatrix, CooMatrix > lanczos(x, max_iter);
        t.toc();
        std::cout << "Creation of Lanczos  took "<< t.diff()<<"s   \n";

        DiaMatrix T; 
        
        t.tic();
        T = lanczos(A, x, b, eps, true); 
        dg::blas2::symv( v2d, b, b);     //normalize
        t.toc();
        
        //Compute error with method 1
        dg::blas2::symv(A, x, helper);
        dg::blas2::symv( v2d, helper, bsymv); //normalize operator
        std::cout << "# of Lanczos Iterations: "<< lanczos.get_iter() <<" | time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
        std::cout << "# Relative error between b=||x||_2 V^T T e_1 and b: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";   

    
        std::cout << "\nComputing with M-Lanczos method \n";
        x = dg::evaluate( lhs, grid);
        dg::blas2::symv(A, x, helper);
        dg::blas2::symv( v2d, helper, bsymv); //normalize operator
        bexac = dg::evaluate( rhs, grid);
        t.tic();
        T= lanczos(A, x, b, v2d, w2d, eps, true); 
        t.toc();
        //Compute error with Method 1
        std::cout << "# of Lanczos Iterations: "<< lanczos.get_iter() <<" | time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
        std::cout << "# Relative error between b=||x||_M V^T T e_1 and b: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";  
    } 
    {
        std::cout << "\nComputing with CG method \n";
        CooMatrix R, Tinv;
        std::pair<CooMatrix, CooMatrix> TinvRpair;    
        t.tic();
        dg::MCG<Container, SubContainer, DiaMatrix, CooMatrix> mcg(x, max_iter);
        t.toc();
        std::cout << "Creation of CG took "<< t.diff()<<"s   \n";
        dg::blas1::scal(x, 0.0); //initialize with zero
        dg::blas2::symv(w2d, bexac, b); //multiply weights
        DiaMatrix T; 
        t.tic();
        T = mcg(A, x, b, v2d, w2d, eps, 1., true); 
        t.toc();

        dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
        std::cout << "# of CG Iterations: "<< mcg.get_iter() <<" | time: "<< t.diff()<<"s \n";   
        std::cout << "# Relative error between x= R T^{-1} e_1 and x: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";
    }

    
    return 0;
}
