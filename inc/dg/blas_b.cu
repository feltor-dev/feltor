#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
#include "blas.h"
#include "backend/derivatives.h"
#include "backend/derivativesX.h"
#include "backend/evaluation.cuh"

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
    dg::GridX2d gridX( 0., lx, 0, ly, 0.2, 0., n, Nx, Ny);
    const dg::DVec w2d = dg::create::weights( grid);
    std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    dg::DVec x = dg::evaluate( function, grid);
    t.toc();
    std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    double gbytes=x.size()*8/1e9;
    t.tic();
    double norm = dg::blas2::dot( w2d, x);
    t.toc();
    std::cout<<"DOT took                         " <<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    dg::DVec y(x);
    dg::DMatrix M = dg::create::dx( grid, dg::centered);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered x derivative took       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    M = dg::create::dx( grid, dg::forward);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward x derivative took        "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    M = dg::create::dy( grid, dg::forward);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward y derivative took        "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    M = dg::create::dy( grid, dg::centered);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered y derivative took       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    M = dg::create::jumpX( grid);
    t.tic();
    dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"jump X took                      "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    t.tic();
    dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    std::cout<<"AXPBY took                       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    t.tic();
    dg::blas1::pointwiseDot( y, x, x);
    t.toc();
    std::cout<<"pointwiseDot took                "<<t.diff()<<"s\t" <<x.size()*8/1e9/t.diff()<<"GB/s\n";
    t.tic();
    norm = dg::blas2::dot( w2d, y);
    t.toc();
    std::cout<<"DOT(w,y) took                    " <<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    t.tic();
    norm = dg::blas2::dot( x, w2d, y);
    t.toc();
    std::cout<<"DOT(x,w,y) took                  " <<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";
    norm++;//get rid of compiler warning

    dg::Composite<dg::DMatrix> Mat = dg::create::dx( gridX, dg::centered);
    t.tic();
    dg::blas2::symv( Mat, x, y);
    t.toc();
    std::cout<<"centered x Xderivative took       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    Mat = dg::create::dx( gridX, dg::forward);
    t.tic();
    dg::blas2::symv( Mat, x, y);
    t.toc();
    std::cout<<"forward x Xderivative took        "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    Mat = dg::create::dy( gridX, dg::forward);
    t.tic();
    dg::blas2::symv( Mat, x, y);
    t.toc();
    std::cout<<"forward y Xderivative took        "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    Mat = dg::create::dy( gridX, dg::centered);
    t.tic();
    dg::blas2::symv( Mat, x, y);
    t.toc();
    std::cout<<"centered y Xderivative took       "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";

    Mat = dg::create::jumpX( gridX);
    t.tic();
    dg::blas2::symv( Mat, x, y);
    t.toc();
    std::cout<<"jump X took                      "<<t.diff()<<"s\t"<<gbytes/t.diff()<<"GB/s\n";


    return 0;
}
