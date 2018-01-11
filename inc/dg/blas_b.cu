#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
#include "blas.h"
#include "geometry/derivatives.h"
#include "geometry/evaluation.cuh"
#include "geometry/fast_interpolation.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double function(double x, double y, double z){ return sin(y)*sin(x);}

//typedef float value_type;
//typedef dg::fDVec Vector;
//typedef dg::fDMatrix Matrix;
//typedef cusp::array1d<float, cusp::device_memory> Vector;
typedef double value_type;
using Vector = thrust::device_vector<double>;
using ArrayVec = std::array<Vector, 10>;
typedef dg::DMatrix Matrix;
//typedef cusp::array1d<double, cusp::device_memory> Vector;
typedef dg::IDMatrix IMatrix;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz; 
    std::cout << "Type n, Nx, Ny and Nz ( Nx and Ny shall be multiples of 2)\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d grid(      0., lx, 0, ly, 0, ly, n, Nx, Ny, Nz);
    dg::Grid3d grid_half = grid; grid_half.multiplyCellNumbers(0.5, 0.5);
    Vector w2d;
    dg::blas1::transfer( dg::create::weights(grid), w2d);

    std::cout<<"Evaluate a function on the grid\n";
    t.tic();
    ArrayVec x;
    dg::blas1::transfer( dg::evaluate( function, grid), x);
    t.toc();
    std::cout<<"Evaluation of a function took    "<<t.diff()<<"s\n";
    std::cout << "Sizeof value type is "<<sizeof(value_type)<<"\n";
    value_type gbytes=(value_type)x.size()*x[0].size()*sizeof(value_type)/1e9;
    std::cout << "Sizeof vectors is "<<gbytes<<" GB\n";
    std::cout << "Generate interpolation and projection\n";
    dg::MultiMatrix<Matrix, ArrayVec> inter, project; 
    dg::blas2::transfer(dg::create::fast_interpolation( grid_half, 2,2), inter);
    dg::blas2::transfer(dg::create::fast_projection( grid, 2,2), project);
    //dg::IDMatrix inter = dg::create::interpolation( grid, grid_half);
    //dg::IDMatrix project = dg::create::projection( grid_half, grid);
    std::cout << "Done...\n";
    int multi=100;
    t.tic();
    value_type norm=0;
    for( int i=0; i<multi; i++)
        norm += dg::blas1::dot( w2d, x[0]);
    t.toc();
    std::cout<<"DOT took                         " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    ArrayVec y(x), z(x), u(x), v(x);
    Matrix M;
    dg::blas2::transfer(dg::create::dx( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered x derivative took       "<<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dx( grid, dg::backward), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward x derivative took        "<<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::backward), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"forward y derivative took        "<<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::dy( grid, dg::centered), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"centered y derivative took       "<<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";

    dg::blas2::transfer(dg::create::jumpX( grid), M);
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::symv( M, x, y);
    t.toc();
    std::cout<<"jump X took                      "<<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    ArrayVec x_half = dg::transfer<ArrayVec>(dg::evaluate( dg::zero, grid_half));
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( inter, x_half, x);
    t.toc();
    std::cout<<"Interpolation half to full grid  "<<t.diff()/multi<<"s\t"<<1.5*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas2::gemv( project, x, x_half);
    t.toc();
    std::cout<<"Projection full to half grid     "<<t.diff()/multi<<"s\t"<<1.5*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    std::cout<<"AXPBY (1*y-1*x=x)                "<<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 2., z);
    t.toc();
    std::cout<<"AXPBYPGZ (1*x-1*y+2*z=z)         "<<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::axpbypgz( 1., x, -1., y, 3., x);
    t.toc();
    std::cout<<"AXPBYPGZ (1*x-1.*y+3*x=x)        "<<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot(  y, x, x);
    t.toc();
    std::cout<<"pointwiseDot (yx=x)              "<<t.diff()/multi<<"s\t" <<2*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  z);
    t.toc();
    std::cout<<"pointwiseDot (1*yx+2*uv=z)       "<<t.diff()/multi<<"s\t" <<5*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 1., y, x, 2.,u,v,0.,  v);
    t.toc();
    std::cout<<"pointwiseDot (1*yx+2*uv=v)       "<<t.diff()/multi<<"s\t" <<4*gbytes*multi/t.diff()<<"GB/s\n";
    t.tic();
    for( int i=0; i<multi; i++)
        norm += dg::blas2::dot( w2d, y);
    t.toc();
    std::cout<<"DOT(w,y) took                    " <<t.diff()/multi<<"s\t"<<2*gbytes*multi/t.diff()<<"GB/s\n";

    t.tic();
    for( int i=0; i<multi; i++)
    {
        norm += dg::blas2::dot( x, w2d, y);
    }
    t.toc();
    std::cout<<"DOT(x,w,y) took                  " <<t.diff()/multi<<"s\t"<<3*gbytes*multi/t.diff()<<"GB/s\n"; //DOT should be faster than axpby since it is only loading vectors and not writing them 
    std::cout<<norm<<std::endl;

    return 0;
}
