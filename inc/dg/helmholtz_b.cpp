#include <iostream>

#include "blas.h"

#include "backend/timer.h"
#include "helmholtz.h"

#include "pcg.h"



const double amp = 0.9;
const double alpha = -0.5;
double lhs( double x, double y){ return sin(x)*sin(y);}
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
double rhs( double x, double y){ return (1.-2.*alpha*pol(x,y))*sin(x)*sin(y);}
//double rhs( double x, double y){ return lhs(x,y);}
int main()
{

    unsigned n, Nx, Ny;
    double eps;
    std::cout << "Type n, Nx and Ny and eps\n";
    std::cin >> n>> Nx >> Ny >> eps;
    dg::Grid2d grid( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny, dg::DIR, dg::PER);
    const dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec one(grid.size(), 1.);
    dg::DVec rho = dg::evaluate( rhs, grid);
    dg::DVec chi = dg::evaluate( pol, grid), chi_inv(chi);;
    dg::blas1::pointwiseDivide( 1., chi, chi_inv);
    const dg::DVec sol = dg::evaluate( lhs, grid);
    dg::DVec x(rho.size(), 0.);
    //dg::DVec x(rho);

    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma1( alpha, {grid, dg::centered});
    gamma1.set_chi( chi_inv);
    dg::blas1::pointwiseDot( chi_inv, rho, rho);

    dg::PCG< dg::DVec > pcg(x, x.size());
    dg::Timer t;
    t.tic();
    unsigned number = pcg.solve( gamma1, x, rho, 1., w2d, eps);
    t.toc();
    dg::blas1::axpby( 1., sol, -1., x);
    std::cout << "DG   performance:\n";
    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;
    std::cout << "took  " << t.diff()<<"s"<<std::endl;


    return 0;
}



