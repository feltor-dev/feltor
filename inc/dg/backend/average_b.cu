#include <iostream>
#include "timer.cuh"
#include "average.cuh"

namespace dg
{
typedef thrust::host_vector<double> HVec;
typedef thrust::device_vector<double> DVec;
}

const double lx = 2.*M_PI;
const double ly = M_PI;

double function( double x, double y) {return cos(x)*sin(y);}
double pol_average( double x, double y) {return cos(x)*2./M_PI;}

dg::bc bcx = dg::PER; 
dg::bc bcy = dg::PER;

int main()
{
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny!\n";
    std::cin >> n >> Nx >> Ny;
    const dg::Grid2d<double> g( 0, lx, 0, ly, n, Nx, Ny);

    dg::PoloidalAverage<dg::HVec, thrust::host_vector<int> > pol(g);
    dg::PoloidalAverage<dg::DVec, thrust::device_vector<int> > pol_device(g);
    dg::Timer t;

    dg::HVec vector = dg::evaluate( function ,g), vector_y( vector);
    dg::DVec dvector= dg::evaluate( function ,g), dvector_y(dvector);
    dg::HVec solution = dg::evaluate( pol_average, g);
    dg::DVec dsolution(solution);
    dg::HVec w2d = dg::create::weights( g);
    dg::DVec w2d_device( w2d);
    t.tic();
    pol( vector, vector_y);
    t.toc();
    std::cout << "Assembly of average vector on host took:      "<<t.diff()<<"s\n";
    t.tic();
    pol_device( dvector, dvector_y);
    t.toc();
    std::cout << "Assembly of average vector on device took:    "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., vector_y, vector);
    std::cout << "Result of integration on host is:     "<<dg::blas1::dot( vector, w2d)<<std::endl; //should be zero
    dg::blas1::axpby( 1., dsolution, -1., dvector_y, dvector);
    std::cout << "Result of integration on device is:   "<<dg::blas1::dot( dvector, w2d_device)<<std::endl;



    return 0;
}
