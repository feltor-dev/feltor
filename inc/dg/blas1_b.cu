#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#include <viennacl/ocl/backend.hpp>

#include "backend/timer.cuh"
#include "blas.h"
#include "backend/derivatives.h"
#include "backend/evaluation.cuh"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double function(double x, double y){ return sin(y)*sin(x);}

//typedef float value_type;
//typedef dg::fDVec Vector;
typedef double value_type;
//typedef dg::DVec Vector;
//typedef viennacl::vector<double> Vector;
typedef cusp::array1d<double, cusp::device_memory> Vector;

int main()
{
    //viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());
    //viennacl::ocl::current_context().build_options("-cl-mad-enable");
    dg::Timer t;
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    dg::Grid2d grid( 0., lx, 0, ly, n, Nx, Ny);
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
    Vector y(x);
    t.tic();
    for( unsigned i=0; i<20;i++)
        value_type norm = dg::blas1::dot( w2d, x);
    t.toc();
    std::cout<<"DOT took                         " <<t.diff()/20.<<"s\t"<<gbytes*20./t.diff()<<"GB/s\n";

    t.tic();
    for( unsigned i=0; i<20;i++)
        dg::blas1::axpby( 1., y, -1., x);
    t.toc();
    std::cout<<"AXPBY took                       "<<t.diff()/20.<<"s\t"<<gbytes*20/t.diff()<<"GB/s\n";

    t.tic();
    for( unsigned i=0; i<20;i++)
        dg::blas1::pointwiseDot( y, x, x);
    t.toc();
    std::cout<<"pointwiseDot took                "<<t.diff()/20<<"s\t" <<gbytes*20/t.diff()<<"GB/s\n";

    return 0;
}
