#include <iostream>

#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

#include "blas.h"

#include "backend/timer.h"
#include "helmholtz.h"

#include "pcg.h"



const double eps = 1e-2;
const double alpha = -0.5;
double lhs( double x, double y){ return sin(x)*sin(y);}
double rhs( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
//double rhs( double x, double y){ return lhs(x,y);}
int main()
{

    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n>> Nx >> Ny;
    dg::Grid2d grid( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny, dg::DIR, dg::PER);
    const dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec one(grid.size(), 1.);
    dg::DVec rho = dg::evaluate( rhs, grid);
    const dg::DVec sol = dg::evaluate( lhs, grid);
    dg::DVec x(rho.size(), 0.);
    //dg::DVec x(rho);

    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma1( alpha, {grid, dg::centered});

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



