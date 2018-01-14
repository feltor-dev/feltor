#include <iostream>
#include "evaluation.cuh"
#include "average.h"
#include "dg/blas.h"


const double lx = M_PI/2.;
const double ly = M_PI;
double function( double x, double y) {return cos(x)*sin(y);}
double pol_average( double x, double y) {return cos(x)*2./M_PI;}
double tor_average( double x, double y) {return sin(y)*2./M_PI;}

int main()
{
    unsigned n = 3, Nx = 32, Ny = 48;
    //![doxygen]
    const dg::Grid2d g( 0, lx, 0, ly, n, Nx, Ny);

    dg::Average< dg::HVec > pol(g, dg::coo2d::y);

    const dg::HVec vector = dg::evaluate( function ,g); 
    dg::HVec average_y( vector);
    std::cout << "Averaging y ... \n";
    pol( vector, average_y);
    //![doxygen]
    const dg::HVec w2d = dg::create::weights( g);
    dg::HVec solution = dg::evaluate( pol_average, g);
    dg::blas1::axpby( 1., solution, -1., average_y);
    exblas::udouble res; 
    res.d = sqrt( dg::blas2::dot( average_y, w2d, average_y));
    std::cout << "Distance to solution is: "<<res.d<<"\t"<<res.i<<std::endl;
    std::cout << "Averaging x ... \n";
    dg::Average< dg::HVec> tor( g, dg::coo2d::x);
    tor( vector, average_y);
    solution = dg::evaluate( tor_average, g);
    dg::blas1::axpby( 1., solution, -1., average_y);
    res.d = sqrt( dg::blas2::dot( average_y, w2d, average_y));
    std::cout << "Distance to solution is: "<<res.d<<"\t"<<res.i<<std::endl;



    return 0;
}
