#include <iostream>
#include <iomanip>
#include <cmath>

#include "simpsons.h"
#include "topology/evaluation.h"
#include "topology/geometry.h"

double sine( double t ) { return sin(t);}
double cosine( double t ) { return cos(t);}


int main()
{
    std::cout << "Program to test the Simpson's rule quadrature algorithm\n";
    unsigned N=20;
    dg::Grid1d g1d( 0, M_PI/2., 3, N );
    dg::HVec times = dg::evaluate( dg::cooX1d, g1d);
    dg::HVec values = dg::evaluate( cosine, g1d);

    dg::SimpsonsRule<double> simpsons( values[0]);
    simpsons.init( 0., 1.);
    for ( unsigned i=0; i<g1d.size(); i++)
        simpsons.add( times[i], values[i]);
    simpsons.add( M_PI/2., 0.);
    double integral = simpsons.get_sum();
    std::cout << "Integral Error is "<<fabs(integral-1.)<<std::endl;

    g1d = dg::Grid1d( M_PI/2., M_PI, 3, N );
    times = dg::evaluate( dg::cooX1d, g1d);
    values = dg::evaluate( cosine, g1d);
    simpsons.flush();
    for ( unsigned i=0; i<g1d.size(); i++)
        simpsons.add( times[i], values[i]);
    simpsons.add( M_PI, -1.);
    integral = simpsons.get_sum();
    std::cout << "Integral Error is "<<fabs(integral+1.)<<std::endl;


    return 0;
}
