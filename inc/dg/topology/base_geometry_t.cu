#include <iostream>
#include <iomanip>
#include "base_geometry.h"

int main()
{
    dg::RealCartesianGrid2d g2d(
            1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10, dg::DIR, dg::DIR);
    dg::RealCartesianGrid2d g2d_test{ g2d.gx(),g2d.gy()};
    assert( g2d.shape(0) == 30);
    assert( g2d.shape(1) == 30);
    assert( g2d_test.shape(0) == 30);
    assert( g2d_test.shape(1) == 30);
    dg::RealCartesianGrid3d g3d(
        1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0., 2.*M_PI, 3, 10, 10, 10,
        dg::DIR, dg::DIR, dg::PER);
    assert( g3d.shape(0) == 30);
    assert( g3d.shape(1) == 30);
    assert( g3d.shape(2) == 10);
    dg::RealCylindricalGrid3d c3d(
        1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0., 2.*M_PI, 3, 10, 10, 10,
        dg::DIR, dg::DIR, dg::PER);
    assert( c3d.shape(0) == 30);
    assert( c3d.shape(1) == 30);
    assert( c3d.shape(2) == 10);
    std::cout << "PASSED\n";

    return 0;
}
