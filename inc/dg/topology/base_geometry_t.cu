#include <iostream>
#include <iomanip>
#include "base_geometry.h"
#include "catch2/catch.hpp"


TEST_CASE( "Base geometry")
{
    SECTION( "2d shapes are correct")
    {
        dg::RealCartesianGrid2d g2d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10,
                dg::DIR, dg::DIR);
        CHECK( g2d.shape(0) == 30);
        CHECK( g2d.shape(1) == 30);

        dg::RealCartesianGrid2d g2d_test{ g2d.gx(),g2d.gy()};
        CHECK( g2d_test.shape(0) == 30);
        CHECK( g2d_test.shape(1) == 30);
    }
    SECTION( "3d shapes are correct")
    {
        dg::RealCartesianGrid3d g3d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0.,
                2.*M_PI, 3, 10, 10, 10, dg::DIR, dg::DIR, dg::PER);
        CHECK( g3d.shape(0) == 30);
        CHECK( g3d.shape(1) == 30);
        CHECK( g3d.shape(2) == 10);
        dg::RealCylindricalGrid3d c3d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0.,
                2.*M_PI, 3, 10, 10, 10, dg::DIR, dg::DIR, dg::PER);
        CHECK( c3d.shape(0) == 30);
        CHECK( c3d.shape(1) == 30);
        CHECK( c3d.shape(2) == 10);
    }
}
