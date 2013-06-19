#include <iostream>

#include "blas.h"

#include "gamma.cuh"
#include "xspacelib.cuh"

#include "cg.cuh"

const unsigned n=3;
const double eps = 1e-4;
const double alpha = -0.5; 
double lhs( double x, double y){ return sin(x)*sin(y);}
double rhs( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
//double rhs( double x, double y){ return lhs(x,y);}
int main()
{
    
    unsigned Nx, Ny; 
    std::cout << "Type Nx and Ny\n";
    std::cin >> Nx >> Ny;
    dg::Grid<double, n> grid( 0, 2.*M_PI, 0, 2.*M_PI, Nx, Ny, dg::DIR, dg::DIR);
    dg::W2D<double, n> w2d( grid.hx(), grid.hy());
    dg::V2D<double, n> v2d( grid.hx(), grid.hy());
    dg::DVec rho = dg::evaluate( rhs, grid);
    dg::DVec sol = dg::evaluate( lhs, grid);
    dg::DVec x(rho.size(), 0.);
    //dg::DVec x(rho);

    dg::DMatrix A = dg::create::laplacianM( grid, dg::normed, dg::XSPACE); 
    dg::Gamma< dg::DMatrix, dg::W2D<double, n> > gamma1( A, w2d, alpha);

    dg::CG< dg::DVec > cg(x, x.size());
    dg::blas2::symv( w2d, rho, rho);
    unsigned number = cg( gamma1, x, rho, v2d, eps);
    dg::blas1::axpby( 1., sol, -1., x);

    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;




    return 0;
}



