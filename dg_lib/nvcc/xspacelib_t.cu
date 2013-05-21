#include <iostream>

#include "xspacelib.cuh"

const unsigned n=3;


double sine( double x, double y){ return sin(x);}

int main()
{
    const dg::Grid<double, n> g( 0, 2.*M_PI, 0, 2.*M_PI, 10, 10, dg::DIR, dg::PER);

    dg::HVec x = dg::evaluate( sine, g);
    dg::ArakawaX<double, n, dg::DVec> arakawa( g);
    return 0;
}
