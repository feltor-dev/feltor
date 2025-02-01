#include <iostream>
#include "evaluation.h"
#include "average.h"
#include "dg/blas.h"

#include "catch2/catch.hpp"

static const double lx = M_PI/2.;
static const double ly = M_PI;
static double function( double x, double y) {return cos(x)*sin(y);}
static double pol_average( double x) {return cos(x)*2./M_PI;}
static double pol_average( double x, double y) {return cos(x)*2./M_PI;}
static double tor_average( double y) {return sin(y)*2./M_PI;}
static double tor_average( double x, double y) {return sin(y)*2./M_PI;}

TEST_CASE( "Average in x and y direction")
{
    using namespace Catch::Matchers;
    unsigned n = 3, Nx = 32, Ny = 48;
    const dg::Grid2d g( 0, lx, 0, ly, n, Nx, Ny);
    SECTION( "Averaging x ")
    {
        dg::Average< dg::IDMatrix, dg::DVec> tor( g, dg::coo2d::x);
        const dg::DVec vector = dg::evaluate( function ,g);
        auto average_x = vector;
        tor( vector, average_x, true);
        dg::DVec solution = dg::evaluate( tor_average, g);
        dg::blas1::axpby( 1., solution, -1., average_x);
        const dg::DVec w2d = dg::create::weights( g);
        double res = sqrt( dg::blas2::dot( average_x, w2d, average_x));
        INFO("Distance to solution is: "<<res<<"\t(Should be 0)");
        CHECK_THAT( res, WithinAbs( 0.0, 1e-13));
    }
    SECTION( "Avarage y")
    {
        //![doxygen]
        dg::Average< dg::IDMatrix, dg::DVec > pol(g, dg::coo2d::y);

        const dg::DVec vector = dg::evaluate( function ,g);
        dg::DVec average_y;
        pol( vector, average_y, false);
        //![doxygen]
        const dg::DVec w1d = dg::create::weights( g.gx());
        dg::DVec solution = dg::evaluate( pol_average, g.gx());
        dg::blas1::axpby( 1., solution, -1., average_y);
        double res = sqrt( dg::blas2::dot( average_y, w1d, average_y));
        INFO("Distance to solution is: "<<res<<"\t(Should be 0)")
        CHECK_THAT( res, WithinAbs( 0.0, 1e-13));
    }
    SECTION( "TEST prolongation y")
    {
        dg::IDMatrix prolong = dg::create::prolongation( g, std::array{1u});
        dg::DVec sol2d = dg::evaluate( pol_average, g), test(sol2d);
        dg::DVec sol1d = dg::evaluate( pol_average, g.gx());
        dg::blas2::symv( prolong, sol1d, test);
        dg::blas1::axpby( 1., sol2d, -1., test);
        double err = dg::blas1::dot( test, test);
        INFO("Distance to solution is: "<<sqrt(err)<<" (Must be 0)");
        CHECK_THAT( sqrt( err), WithinAbs( 0.0, 1e-13));
    }
    SECTION( "TEST prolongation x")
    {
        dg::IDMatrix prolong = dg::create::prolongation( g, std::array{0u});
        dg::DVec sol2d = dg::evaluate( tor_average, g), test(sol2d);
        dg::DVec sol1d = dg::evaluate( tor_average, g.gy());
        dg::blas2::symv( prolong, sol1d, test);
        dg::blas1::axpby( 1., sol2d, -1., test);
        double err = dg::blas1::dot( test, test);
        INFO( "Distance to solution is: "<<sqrt(err)<<" (Must be 0)");
        CHECK_THAT( sqrt( err), WithinAbs( 0.0, 1e-13));
    }
}
