#include <iostream>
#include "average.cuh"


const double lx = M_PI;
const double ly = M_PI;
double function( double x, double y) {return cos(x)*sin(y);}

int main()
{
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny!\n";
    std::cin >> n >> Nx >> Ny;
    const dg::Grid<double> g( 0, lx, 0, ly, n, Nx, Ny);

    dg::PoloidalAverage<dg::HVec, thrust::host_vector<int> > pol(g);

    dg::HVec vector = dg::evaluate( function ,g), vector_y( vector);
    dg::HVec w2d = dg::create::w2d( g);
    std::cout << "Integration is: "<<dg::blas1::dot( vector, w2d)<<std::endl;

    pol( vector, vector_y);
    dg::blas1::axpby( 1., vector, -1., vector_y, vector);
    std::cout << "Result of integration is: "<<dg::blas1::dot( vector, w2d)<<std::endl;



    return 0;
}
