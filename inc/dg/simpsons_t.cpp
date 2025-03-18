#include <iostream>
#include <iomanip>
#include <cmath>

#include "simpsons.h"
#include "topology/evaluation.h"
#include "topology/geometry.h"

#include "catch2/catch_all.hpp"

TEST_CASE("Simpsons")
{
    // Program to test the Simpson's rule quadrature algorithm
    SECTION( "Documentation")
    {
        //! [docu]
        // Construct a list of data points to integrate
        dg::Grid1d g1d( 0, M_PI/2., 3, 20 );
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
        INFO( "Error Simpsons is "<<fabs(integral-1.));
        CHECK( fabs( integral - 1.) < 1e-8);

        std::array<double,2> boundaries = simpsons.get_boundaries();
        INFO( "Integrated from "<<boundaries[0]<<" (0) to "<<boundaries[1]<<" ("<<M_PI/2.<<")");
        CHECK( boundaries[0] == 0);
        CHECK( boundaries[1] == M_PI/2.);
        //! [docu]
    }
    SECTION( "init function")
    {
        dg::Simpsons<double> simpsons;
        // Forgetting init function will throw
        CHECK_THROWS_AS( simpsons.add( -1, 32), dg::Error);
    }
    SECTION( "2nd order")
    {
        dg::Grid1d g1d( M_PI/2., M_PI, 3, 20 );
        dg::HVec times = dg::evaluate( dg::cooX1d, g1d);
        dg::HVec values = dg::evaluate( cos, g1d);
        dg::Simpsons<double> simpsons;
        // add some random values
        simpsons.init( 0, 49);
        simpsons.add( M_PI/2., cos(M_PI/2.));
        simpsons.flush();
        CHECK( simpsons.get_integral() == 0);
        CHECK( simpsons.get_boundaries()[0] == M_PI/2.);
        CHECK( simpsons.get_boundaries()[1] == M_PI/2.);
        simpsons.set_order(2);
        CHECK( simpsons.get_order() == 2);
        for ( unsigned i=0; i<g1d.size(); i++)
            simpsons.add( times[i], values[i]);
        simpsons.add( M_PI, -1.);
        double integral = simpsons.get_integral();
        INFO("Error Trapezoidal is "<<fabs(integral+1.));
        CHECK( fabs(integral+1.) < 1e-4);
        auto boundaries = simpsons.get_boundaries();
        INFO("Integrated from "<<boundaries[0]<<" ("<<M_PI/2.<<") to "<<boundaries[1]<<" ("<<M_PI<<")");
        CHECK( boundaries[0] == M_PI/2.);
        CHECK( boundaries[1] == M_PI);
    }
}

