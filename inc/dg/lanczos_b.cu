#define DG_DEBUG
#include <iostream>
#include <iomanip>
#include "backend/timer.h"
#include "lanczos.h"
#include "helmholtz.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -1;
const double m = 1.;
const double n = 1.;

double lhs( double x, double y) {return sin(m*x)*sin(n*y);}
double rhs( double x, double y){ return (1.-(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}

using Matrix = dg::DMatrix;
using Container = dg::DVec;
using HDiaMatrix = cusp::dia_matrix<int, double, cusp::host_memory>;
using HCooMatrix = cusp::coo_matrix<int, double, cusp::host_memory>;

int main(int argc, char * argv[])
{
    dg::Timer t;
    unsigned n, Nx, Ny;
    std::cout << "# Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<std::endl;
    unsigned max_iter;
    std::cout << "# Type in max_iter and eps\n"; 
    double eps = 1e-6; 
    std::cin >> max_iter>> eps;
    std::cout <<"# You typed\n"
              <<"max_iter:  "<<max_iter<<"\n"
              <<"eps: "<<eps <<std::endl;  
    dg::CartesianGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy);
    
    const Container w2d = dg::create::weights( grid);
    const Container v2d = dg::create::inv_weights( grid);
        
    Container x = dg::evaluate( lhs, grid), b(x), zero(x), one(x), error(x),  helper(x), xexac(x);
    Container bexac = dg::evaluate( rhs, grid);
    dg::blas1::scal(zero, 0.0);
    one = dg::evaluate(dg::one, grid);
    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( grid, alpha, dg::centered); //not_normed
    
    {
        t.tic();
        dg::Lanczos< Container > lanczos(x, max_iter);
        t.toc();
        std::cout << "# Lanczos creation took "<< t.diff()<<"s   \n";

        HDiaMatrix T; 
        std::cout << "Lanczos:\n";
       
        t.tic();
        T = lanczos( A, x, b, eps, true); 
        dg::blas2::symv( v2d, b, b);     //normalize
        t.toc();
        
        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
        std::cout << "    # Relative error between b=||x||_2 V^T T e_1 and b: \n";   
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";   

        std::cout << "\nM-Lanczos:\n";
        x = dg::evaluate( lhs, grid);
        t.tic();
        T = lanczos(A, x, b, v2d, w2d, eps, true); 
        t.toc();
        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
        std::cout << "    # Relative error between b=||x||_M V^T T e_1 and b: \n";  
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";   

    } 
    {
        std::cout << "\nM-CG: \n";
        t.tic();
        dg::MCG<Container> mcg(x, max_iter);
        t.toc();
        std::cout << "#    M-CG creation took "<< t.diff()<<"s   \n";
//         dg::blas1::scal(x, 0.0); //initialize with zero
//         dg::blas1::scal(x, 0.0); //initialize with zero
        x =    dg::evaluate(dg::one, grid);
        dg::blas1::scal(x,1000.0); //initialize not with zero

        dg::blas2::symv(w2d, bexac, b); //multiply weights
        t.tic();
        HDiaMatrix T = mcg(A, x, b, v2d, w2d, eps, 1., true); 
        t.toc();
        


        dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
        std::cout << "    iter: "<< mcg.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x= R T^{-1} e_1 and x: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";
    }

    
    return 0;
}
