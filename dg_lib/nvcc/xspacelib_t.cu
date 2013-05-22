#include <iostream>

#include "xspacelib.cuh"

const unsigned n=3;


double sine( double x, double y){ return sin(x);}

int main()
{
    const dg::Grid<double, n> grid( 0, 2.*M_PI, 0, 2.*M_PI, 10, 10, dg::DIR, dg::PER);

    dg::HVec x = dg::evaluate( sine, grid);
    dg::ArakawaX<double, n, dg::DVec> arakawa( grid);

    dg::Polarisation2dX<double, n, dg::DVec> polarisation ( grid);
    return 0;
}
