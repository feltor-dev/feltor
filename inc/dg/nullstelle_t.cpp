#include <iostream>
#include "nullstelle.h"


#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Nullstelle")
{
    SECTION( "Linear function")
    {
        double xmin = 0, xmax = 1;
        int num = dg::bisection1d( [](double x){ return 2*x - 1.;}, xmin, xmax, 1e-10);
        // Should use only three function calls
        CHECK( num == 3);
        CHECK( xmin == 0.5);
        CHECK( xmax == 0.5);
    }
    SECTION( "Find sqrt")
    {
        //! [sqrt]
        double xmin = 0, xmax = 2;
        int num = dg::bisection1d( [](double x){ return x*x - 2.;}, xmin, xmax, 1e-10);
        // Num calls = 10* ln (10)/ln(2) = 33.2...
        CHECK( num == 35);
        CHECK( fabs( xmin - sqrt(2)) < 1e-9);
        CHECK( fabs( xmax - sqrt(2)) < 1e-9);
        //! [sqrt]
    }


}
