#include <iostream>
#include <iomanip>
#include <cmath>

#include "simpsons.h"
#include "topology/evaluation.h"
#include "topology/geometry.h"


int main()
{
    std::cout << "Program to test the Simpson's rule quadrature algorithm\n";
    //![docu]
    // Let us construct an artificial list of data points
    unsigned N=20;
    dg::Grid1d g1d( 0, M_PI/2., 3, N );
    dg::HVec times = dg::evaluate( dg::cooX1d, g1d);
    dg::HVec values = dg::evaluate( cos, g1d);

    dg::Simpsons<double> simpsons;

    //Init the left side boundary
    simpsons.init( 0., 1.);

    //Now add (time, value) pairs to the integral
    for ( unsigned i=0; i<g1d.size(); i++)
        simpsons.add( times[i], values[i]);

    //Add a last point since the dg grid is cell-centered
    simpsons.add( M_PI/2., 0.);

    //Ask for the integral
    double integral = simpsons.get_integral();
    std::cout << "Error Simpsons is "<<fabs(integral-1.)<<std::endl;
    //![docu]
    //
    std::array<double,2> boundaries = simpsons.get_boundaries();
    std::cout << "Integrated from "<<boundaries[0]<<" (0) to "<<boundaries[1]<<" ("<<M_PI/2.<<") "<<std::endl;

    g1d = dg::Grid1d( M_PI/2., M_PI, 3, N );
    times = dg::evaluate( dg::cooX1d, g1d);
    values = dg::evaluate( cos, g1d);
    simpsons.flush();
    simpsons.set_order(2);
    for ( unsigned i=0; i<g1d.size(); i++)
        simpsons.add( times[i], values[i]);
    simpsons.add( M_PI, -1.);
    integral = simpsons.get_integral();
    std::cout << "Error Trapezoidal is "<<fabs(integral+1.)<<std::endl;
    boundaries = simpsons.get_boundaries();
    std::cout << "Integrated from "<<boundaries[0]<<" ("<<M_PI/2.<<") to "<<boundaries[1]<<" ("<<M_PI<<") "<<std::endl;


    return 0;
}
