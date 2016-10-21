#include <iostream>
#include "average.cuh"
#include "../blas2.h"
#include "typedefs.cuh"


const double lx = 2.*M_PI;
const double ly = M_PI;
double function( double x, double y) {return cos(x)*sin(y);}
double pol_average( double x, double y) {return cos(x)*2./M_PI;}

int main()
{
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny!\n";
    std::cin >> n >> Nx >> Ny;
    const dg::Grid2d g( 0, lx, 0, ly, n, Nx, Ny);
    dg::HVec w2d = dg::create::weights( g);

    dg::PoloidalAverage<dg::HVec, thrust::host_vector<int> > pol(g);

    dg::HVec vector = dg::evaluate( function ,g), average_y( vector);
    const dg::HVec solution = dg::evaluate( pol_average, g);
    std::cout << "Averaging ... \n";
    pol( vector, average_y);
    dg::blas1::axpby( 1., solution, -1., average_y, vector);
    std::cout << "Distance to solution is: "<<sqrt(dg::blas2::dot( vector, w2d, vector))<<std::endl;



    return 0;
}
