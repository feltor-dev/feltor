#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
#include "blas.h"
#include "backend/derivatives.h"
#include "backend/evaluation.cuh"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double function(double x, double y, double z){ return sin(y)*sin(x);}

//typedef float value_type;
//typedef dg::fDVec Vector;
//typedef dg::fDMatrix Matrix;
//typedef cusp::array1d<float, cusp::device_memory> Vector;
typedef double value_type;
typedef dg::DVec Vector;
//typedef thrust::device_vector<double> Vector;
typedef dg::DMatrix Matrix;
//typedef cusp::array1d<double, cusp::device_memory> Vector;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz; 
    std::cout << "Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d<double> grid( 0., lx, 0, ly, 0, ly, n, Nx, Ny, Nz);
    Vector w2d;
    dg::blas1::transfer( dg::create::weights(grid), w2d);

    std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    Vector x;
    dg::blas1::transfer( dg::evaluate( function, grid), x);
    t.toc();
    std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    std::cout << "Sizeof value type is "<<sizeof(value_type)<<"\n";
    value_type gbytes=(value_type)x.size()*sizeof(value_type)/1e9;
    std::cout << "Sizeof vectors is "<<gbytes<<" GB\n";
    t.tic();
    value_type norm=0;
    for( unsigned i=0; i<20; i++)
        norm += dg::blas1::dot( w2d, x);
    t.toc();
    std::cout<<"DOT took                         " <<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";
    Vector y(x);
    Matrix M;
    dg::blas2::transfer(dg::create::dx( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered x derivative took       "<<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dx( grid, dg::backward), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward x derivative took        "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::backward), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward y derivative took        "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered y derivative took       "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::jumpX( grid), M);
    t.tic();
    for( int i=0; i<20; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"jump X took                      "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<20; i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    std::cout<<"AXPBY took                       "<<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<20; i++)
        dg::blas1::pointwiseDot( y, x, x);
    t.toc();
    std::cout<<"pointwiseDot took                "<<t.diff()/20<<"s\t" <<gbytes*20/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<20; i++)
        norm += dg::blas2::dot( w2d, y);
    t.toc();
    std::cout<<"DOT(w,y) took                    " <<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<20; i++)
    {
        norm += dg::blas2::dot( x, w2d, y);
    }
    t.toc();
    std::cout<<"DOT(x,w,y) took                  " <<t.diff()/20<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";
    std::cout<<norm<<std::endl;

    return 0;
}
