#include <iostream>

#include "topology/evaluation.h"
#include "extrapolation.h"

#include "catch2/catch_all.hpp"

TEST_CASE( "Extrapolation")
{
    //Test our least squares extrapolation algorithm
    SECTION( "least_squares")
    {
        //! [least_squares]
        // Least squares fit through three points
        std::array<double,3> ones = {1,1,1};
        std::array<double,3> xs = {0,1,2};
        std::array<double,3> ys = {6,0,0};
        // The ones refer to the constant a_0 in the formula y = a_0 + a_1 x
        auto a = dg::least_squares( std::vector{ones, xs}, ys);
        REQUIRE( a.size() == 2);
        INFO( "Least squares fit: a[0] "<<a[0]<< " (5), a[1] "<<a[1]<<" (-3)");
        CHECK( a[0]== 5);
        CHECK( a[1]== -3);
        //! [least_squares]
    }

    SECTION( "Polynomial fit")
    {
        // Polynomial fit through three points
        double value;
        dg::Extrapolation<double> extra(3,-1);
        extra.update( 0, 0);
        CHECK( extra.exists( -1e-15));
        extra.update( 1, 1);
        extra.extrapolate( 5, value);
        INFO( "Linear Extrapolated value is "<<value<< " (5)");
        CHECK( value == 5);
        extra.update( 1, 1); //should not change anything
        extra.derive( 4, value);
        INFO( "Linear Derived value is "<<value<< " (1)");
        CHECK( value == 1);
        extra.update( 3, 9);
        extra.update( 2, 4);
        extra.extrapolate( 5, value);
        INFO( "Extrapolated value is "<<value<< " (25)");
        CHECK( value == 25);
        extra.derive( 4, value);
        INFO( "Derived value is "<<value<< " (8)");
        CHECK( value == 8);
        extra.set_max(2,-1);
        extra.update( 0, 0);
        extra.update( 1, 1);
        extra.update( 3, 9);
        CHECK( extra.exists( 3));
        extra.update( 2, 4);
        extra.extrapolate( 5, value);
        INFO( "Linear Extrapolated value is "<<value<< " (19)");
        CHECK( value == 19);
        extra.derive( 4, value);
        INFO( "Linear Derived value is "<<value<< " (5)");
        CHECK( value == 5);
        extra.set_max(1,-1);
        extra.extrapolate( 5, value);
        INFO( "Empty Extrapolated value is "<<value<< " (0)");
        CHECK( value == 0);
        extra.derive( 4, value);
        INFO( "Empty Derived value is "<<value<< " (0)");
        CHECK( value == 0);
        extra.update( 0, 0);
        extra.update( 1, 1);
        extra.update( 3, 9);
        extra.update( 2, 4);
        extra.update( 2, 7); //overwrite existing value
        extra.extrapolate( 5, value);
        INFO( "Monomial Extrapolated value is "<<value<< " (7)");
        CHECK( value == 7);
        extra.derive( 4, value);
        INFO( "Monomial Derived value is "<<value<< " (0)");
        CHECK( value == 0);
    }

    SECTION( "LeastSquaresExtrapolation")
    {
        //! [LeastSquaresExtrapolation]
        //Least-squares fit through many points
        dg::Grid1d grid( 0., 2.*M_PI, 1, 200);
        // We have a travelling wave at various times
        // Note that the xs are linearly dependent e.g. x2 = a*x1 + b*x2
        dg::HVec x0 = dg::evaluate( [=](double x) { return sin( x -0.0);}, grid);
        dg::HVec x1 = dg::evaluate( [=](double x) { return sin( x -0.1);}, grid);
        dg::HVec x2 = dg::evaluate( [=](double x) { return sin( x -0.2);}, grid);
        dg::HVec x3 = dg::evaluate( [=](double x) { return sin( x -0.3);}, grid);

        // An imaginary associated vector to each xs
        dg::HVec y0 = dg::evaluate( [=](double x) { return cos( x -0.0);}, grid);
        dg::HVec y1 = dg::evaluate( [=](double x) { return cos( x -0.1);}, grid);
        dg::HVec y2 = dg::evaluate( [=](double x) { return cos( x -0.2);}, grid);
        dg::HVec y  = dg::evaluate( [=](double x) { return cos( x -0.3);}, grid);
        dg::HVec guess ( y);
        double norm = dg::blas1::dot( y,y);

        // Given (x0, y0), (x1, y1), (x2, y2) give a good guess for y at x3
        dg::LeastSquaresExtrapolation<dg::HVec, dg::HVec> extraLS(3,guess, guess);

        // Since the input xs, ys are linearly dependent we expect the exact solution
        extraLS.update( x0, y0);
        extraLS.update( x1, y1);
        extraLS.extrapolate( x3, guess);
        dg::blas1::axpby( 1., y, -1., guess);
        double err = sqrt( dg::blas1::dot( guess, guess) /norm);
        INFO( "Difference LS 1 is "<<err);
        CHECK( err < 1e-13);

        extraLS.update( x1, x1); // rejected (already there)
        extraLS.update( x2, y2); // also rejected (linearly dependent)
        extraLS.extrapolate( x3, guess);
        dg::blas1::axpby( 1., y, -1., guess);
        err = sqrt( dg::blas1::dot( guess, guess) /norm);
        INFO( "Difference LS 2 is "<<err);
        CHECK( err < 1e-13);
        //! [LeastSquaresExtrapolation]
    }
    SECTION( "Extrapolation")
    {
        //! [Extrapolation]
        //Least-squares fit through many points
        dg::Grid1d grid( 0., 2.*M_PI, 1, 200);
        // We have a travelling wave at various times
        dg::HVec y0 = dg::evaluate( [=](double x) { return cos( x -0.0);}, grid);
        dg::HVec y1 = dg::evaluate( [=](double x) { return cos( x -0.1);}, grid);
        dg::HVec y2 = dg::evaluate( [=](double x) { return cos( x -0.2);}, grid);
        dg::HVec y  = dg::evaluate( [=](double x) { return cos( x -0.3);}, grid);
        dg::HVec guess ( y);
        double norm = dg::blas1::dot( y,y);

        // Given (t0, y0), (t1, y1), (t2, y2) give a good guess for y at t3

        dg::Extrapolation<dg::HVec> extra(3,y);

        extra.update( 0.0, y0);
        extra.extrapolate( 0.3, guess);
        // From 1 point we expect simply the same point
        dg::blas1::axpby( 1., y0, -1., guess);
        double err = sqrt( dg::blas1::dot( guess, guess) /norm);
        INFO( "Difference 1 is "<<err);
        CHECK( err < 1e-14);

        // From 3 points we have a parabolic extrapolation
        extra.update( 0.1, y1);
        extra.update( 0.1, y1); // This one silently overwrites previous t1=0.1
        extra.update( 0.2, y2);
        extra.extrapolate( 0.3, guess);
        dg::blas1::axpby( 1., y, -1., guess);
        err = sqrt( dg::blas1::dot( guess, guess) /norm);
        INFO( "Difference 2 is "<<err);
        // The accuracy of polynomial extrapolation is not great ...
        CHECK( err < 1e-3);
        //! [Extrapolation]
    }
}
