#include <iostream>
#include "evaluation.h"
#include "average.h"
#include "dg/blas.h"


const double lx = M_PI/2.;
const double ly = M_PI;
double function( double x, double y) {return cos(x)*sin(y);}
double pol_average( double x) {return cos(x)*2./M_PI;}
double pol_average( double x, double y) {return cos(x)*2./M_PI;}
double tor_average( double y) {return sin(y)*2./M_PI;}
double tor_average( double x, double y) {return sin(y)*2./M_PI;}

int main()
{
    std::cout << "Program to test the average in x and y direction\n";
    unsigned n = 3, Nx = 32, Ny = 48;
    //![doxygen]
    const dg::Grid2d g( 0, lx, 0, ly, n, Nx, Ny);

    dg::Average< dg::IDMatrix, dg::DVec > pol(g, dg::coo2d::y);

    const dg::DVec vector = dg::evaluate( function ,g);
    dg::DVec average_y;
    std::cout << "Averaging y ... \n";
    pol( vector, average_y, false);
    //![doxygen]
    const dg::DVec w1d = dg::create::weights( g.gx());
    dg::DVec solution = dg::evaluate( pol_average, g.gx());
    dg::blas1::axpby( 1., solution, -1., average_y);
    std::cout << "TEST prolongation y\n";
    dg::IDMatrix prolong = dg::create::prolongation( g, std::array{1u});
    dg::DVec sol2d = dg::evaluate( pol_average, g), test(sol2d);
    dg::blas2::symv( prolong, solution, test);
    dg::blas1::axpby( 1., sol2d, -1., test);
    double err = dg::blas1::dot( test, test);
    std::cout << "Distance to solution is: "<<sqrt(err)<<" (Must be 0)\n";

    double res = sqrt( dg::blas2::dot( average_y, w1d, average_y));
    std::cout << "Distance to solution is: "<<res<<"\t(Should be 0)"<<std::endl;
    std::cout << "Averaging x ... \n";
    dg::Average< dg::IDMatrix, dg::DVec> tor( g, dg::coo2d::x);
    average_y = vector;
    tor( vector, average_y, true);
    solution = dg::evaluate( tor_average, g);
    dg::blas1::axpby( 1., solution, -1., average_y);
    const dg::DVec w2d = dg::create::weights( g);
    res = sqrt( dg::blas2::dot( average_y, w2d, average_y));
    std::cout << "Distance to solution is: "<<res<<"\t(Should be 0)"<<std::endl;

    std::cout << "TEST prolongation x\n";
    prolong = dg::create::prolongation( g, std::array{0u});
    dg::DVec sol1d = dg::evaluate( tor_average, g.gy());
    dg::blas2::symv( prolong, sol1d, test);
    dg::blas1::axpby( 1., solution, -1., test);
    err = dg::blas1::dot( test, test);
    std::cout << "Distance to solution is: "<<sqrt(err)<<" (Must be 0)\n";
    //std::cout << "\n Continue with \n\n";

    return 0;
}
