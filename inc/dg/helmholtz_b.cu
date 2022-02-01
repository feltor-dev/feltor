#include <iostream>

#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

#include "blas.h"

#include "backend/timer.h"
#include "helmholtz.h"
#include "pcg.h"



const double eps = 1e-10;
const double alpha = -0.5;
double lhs( double x, double y){ return sin(x)*sin(y);}
double rhs( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
double rhs2( double x, double y){ return rhs(x,y) - lhs(x,y);}
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
    dg::DVec x(rho.size(), 0.), err(rho);
    dg::DVec rho2 = dg::evaluate( rhs2, grid);
    dg::Timer t;

    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma1( grid, alpha, dg::centered);
    t.tic();
    dg::blas2::symv(gamma1, sol, x);
    t.toc();
    std::cout << "symv1 took "<< t.diff()<<"s\n";
    dg::blas1::pointwiseDot(x, v2d, x);
    dg::blas1::axpby( -1., rho, 1., x,err);
    double app_err = sqrt( dg::blas2::dot( w2d, err)/dg::blas2::dot( w2d, rho) );
    std::cout << "error " << app_err   << "\n";
/*
    //symv via Projection
    dg::CartesianGrid2d f_grid = grid;
    f_grid.set( (2*grid.n()), 2.*grid.Nx(), 2.*grid.Ny());
    dg::IDMatrix inter = dg::create::interpolation( f_grid, grid);
    dg::IDMatrix project = dg::create::projection( grid, f_grid);
    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > f_gamma1( f_grid, alpha, dg::centered);
    dg::DVec f_sol = dg::evaluate( lhs, f_grid), f_x(f_sol);
    const dg::DVec f_v2d = dg::create::inv_weights( f_grid);
    dg::blas2::symv( inter, sol, f_sol);
    dg::blas2::symv(f_gamma1, f_sol, f_x);
    dg::blas1::pointwiseDot(f_x, f_v2d, f_x);
    dg::blas2::symv( project, f_x, x);
//     dg::blas1::pointwiseDot(x, v2d, x);
    dg::blas1::axpby( -1., rho, 1., x, err);
    app_err = sqrt( dg::blas2::dot( w2d, err)/dg::blas2::dot( w2d, rho) );
    std::cout << "error " << app_err   << "\n";*/
    


    dg::PCG< dg::DVec > pcg(x, x.size());
    dg::Timer t;
    t.tic();
    unsigned number = pcg.solve( gamma1, x, rho, 1., w2d, eps);
    t.toc();
    dg::blas1::axpby( 1., sol, -1., x);
    std::cout << "DG   performance:\n";
    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, x)/dg::blas2::dot( w2d, sol)) << std::endl;
    std::cout << "took  " << t.diff()<<"s"<<std::endl;

    return 0;
}



