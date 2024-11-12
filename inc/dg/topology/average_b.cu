#include <iostream>
#include "average.h"
#include "evaluation.h"
#include "dg/backend/timer.h"
#include "dg/backend/typedefs.h"

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

// Makes no sense any more to compare exact and simple
    dg::Average<dg::IHMatrix, dg::HVec> pol(g, dg::coo2d::y);
    dg::Average<dg::IDMatrix, dg::DVec> pol_device(g, dg::coo2d::y);
    dg::Average<dg::IHMatrix, dg::HVec> pol_ex(g, dg::coo2d::y);
    dg::Average<dg::IDMatrix, dg::DVec> pol_device_ex(g, dg::coo2d::y);
    dg::Timer t;

    dg::HVec vector = dg::evaluate( function ,g), vector_y( vector);
    dg::DVec dvector= dg::evaluate( function ,g), dvector_y(dvector);
    dg::HVec solution = dg::evaluate( pol_average, g);
    dg::DVec dsolution(solution);
    dg::HVec w2d = dg::create::weights( g);
    dg::DVec w2d_device( w2d);
    t.tic();
    for( unsigned i=0; i<100; i++)
        pol_ex( vector, vector_y);
    t.toc();
    std::cout << "Assembly of average (exact)  vector on host took:      "<<t.diff()/100.<<"s\n";
    t.tic();
    for( unsigned i=0; i<100; i++)
        pol_device_ex( dvector, dvector_y);
    t.toc();
    std::cout << "Assembly of average (exact)  vector on device took:    "<<t.diff()/100.<<"s\n";
    t.tic();
    for( unsigned i=0; i<100; i++)
        pol( vector, vector_y);
    t.toc();
    std::cout << "Assembly of average (simple) vector on host took:      "<<t.diff()/100.<<"s\n";
    t.tic();
    for( unsigned i=0; i<100; i++)
        pol_device( dvector, dvector_y);
    t.toc();
    std::cout << "Assembly of average (simple) vector on device took:    "<<t.diff()/100.<<"s\n";
    dg::blas1::axpby( 1., solution, -1., vector_y, vector);
    std::cout << "Result of integration on host is:     "<<dg::blas1::dot( vector, w2d)<<std::endl; //should be zero
    dg::blas1::axpby( 1., dsolution, -1., dvector_y, dvector);
    std::cout << "Result of integration on device is:   "<<dg::blas1::dot( dvector, w2d_device)<<std::endl;



    return 0;
}
