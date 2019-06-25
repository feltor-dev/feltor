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
    dg::Grid1d g1d( 0, 2.*M_PI, 1, 20 );
    dg::HVec times = dg::evaluate( dg::cooX1d, g1d);
    dg::HVec values = dg::evaluate( cosine, g1d);

    dg::SimpsonsRule<double> simpsons( values[0]);
    simpsons.init( 0., 1.);
    double integral = simpsons.get_sum();
    std::cout << "Integral is "<<integral<<std::endl;
    for ( unsigned i=0; i<10; i++)
    {
        simpsons.add( times[i], values[i]);
        integral = simpsons.get_sum();
        std::cout << "Integral is "<<integral<<std::endl;
    }
    simpsons.add( M_PI, -1.);
    integral = simpsons.get_sum();
    std::cout << "Integral is "<<integral<<std::endl;


    return 0;
}
