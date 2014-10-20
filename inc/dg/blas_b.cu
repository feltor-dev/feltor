#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
#include "backend/evaluation.cuh"
#include "backend/derivatives.cuh"
#include "backend/typedefs.cuh"
#include "blas.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double function(double x, double y){ return sin(y)*sin(x);}

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny; 
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    dg::Grid2d<double> grid( 0., lx, 0, ly, n, Nx, Ny);
    const dg::DVec w2d = dg::create::weights( grid);
    std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    dg::DVec x = dg::evaluate( function, grid);
    t.toc();
    std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    t.tic();
    double norm = dg::blas2::dot( w2d, x);
    t.toc();
    std::cout<<"DOT took                         " <<t.diff()<<"s   result: "<<norm<<"\n";
    dg::DVec y(x);
    dg::DMatrix M = dg::create::dx( grid, dg::normed, dg::centered);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"SYMV took                        "<<t.diff()<<"s (centered x derivative!)\n";
    M = dg::create::dx( grid, dg::normed, dg::forward);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"SYMV took                        "<<t.diff()<<"s (forward x derivative!)\n";
    M = dg::create::dy( grid, dg::normed, dg::forward);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"SYMV took                        "<<t.diff()<<"s (forward y derivative!)\n";
    M = dg::create::dy( grid, dg::normed, dg::centered);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"SYMV took                        "<<t.diff()<<"s (centered y derivative!)\n";
    M = dg::create::jump2d( grid, dg::not_normed);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"SYMV took                        "<<t.diff()<<"s (2d jump!)\n";
    t.tic();
    dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    std::cout<<"AXPBY took                       "<<t.diff()<<"s\n";
    t.tic();
    dg::blas1::pointwiseDot( y, x, x);
    t.toc();
    std::cout<<"pointwiseDot took                "<<t.diff()<<"s\n";

    return 0;
}
