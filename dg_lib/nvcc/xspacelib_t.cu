#include <iostream>

#include "xspacelib.cuh"

const unsigned n=3;

double sine( double x, double y){ return sin(x);}

int main()
{
    dg::Grid<double, n> g( 0, 2.*M_PI, 0, 2.*M_PI, 10, 10);

    dg::HVec x = dg::evaluate( sine, g);
    return 0;
}
