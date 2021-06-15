#include <iostream>

#include "topology/evaluation.h"
#include "extrapolation.h"

int main()
{
    std::cout << "Test our least squares extrapolation algorithm\n";
    std::cout << "First test least squares fit through three points\n";
    std::vector<std::array<double,3>> bs(2);
    bs[0] = {1,1,1};
    bs[1] = {0,1,2};
    std::array<double,3> b = {6,0,0};
    auto a = dg::least_squares( bs, b);
    std::cout<< "Least squares fit is: a[0] "<<a[0]<< " (5), a[1] "<<a[1]<<" (-3)\n";
    assert( a[0]== 5);
    assert( a[1]== -3);
    std::cout << "TEST PASSED!\n";

    {
    std::cout << "\nNow test polynomial fit through three points\n\n";
    double value;
    dg::Extrapolation<double> extra(3,-1);
    extra.update( 0, 0);
    assert( extra.exists( -1e-15));
    extra.update( 1, 1);
    extra.extrapolate( 5, value);
    std::cout << "Linear Extrapolated value is "<<value<< " (5)\n";
    extra.update( 1, 1); //should not change anything
    extra.derive( 4, value);
    std::cout << "Linear Derived value is "<<value<< " (1)\n";
    extra.update( 3, 9);
    extra.update( 2, 4);
    extra.extrapolate( 5, value);
    std::cout << "Extrapolated value is "<<value<< " (25)\n";
    extra.derive( 4, value);
    std::cout << "Derived value is "<<value<< " (8)\n";
    extra.set_max(2,-1);
    extra.update( 0, 0);
    extra.update( 1, 1);
    extra.update( 3, 9);
    assert( extra.exists( 3));
    extra.update( 2, 4);
    extra.extrapolate( 5, value);
    std::cout << "Linear Extrapolated value is "<<value<< " (19)\n";
    extra.derive( 4, value);
    std::cout << "Linear Derived value is "<<value<< " (5)\n";
    extra.set_max(1,-1);
    extra.extrapolate( 5, value);
    std::cout << "Empty Extrapolated value is "<<value<< " (0)\n";
    extra.derive( 4, value);
    std::cout << "Empty Derived value is "<<value<< " (0)\n";
    extra.update( 0, 0);
    extra.update( 1, 1);
    extra.update( 3, 9);
    extra.update( 2, 4);
    extra.update( 2, 7); //overwrite existing value
    extra.extrapolate( 5, value);
    std::cout << "Monomial Extrapolated value is "<<value<< " (7)\n";
    extra.derive( 4, value);
    std::cout << "Monomial Derived value is "<<value<< " (0)\n";
    }

    /////////////////////////
    {
    std::cout << "\nNow test least-squares fit through many points\n\n";
    dg::Grid1d grid( 0., 2.*M_PI, 1, 200);
    dg::HVec x0 = dg::evaluate( [=](double x) { return sin( x -0.0);}, grid);
    dg::HVec x1 = dg::evaluate( [=](double x) { return sin( x -0.1);}, grid);
    dg::HVec x2 = dg::evaluate( [=](double x) { return sin( x -0.2);}, grid);
    dg::HVec x  = dg::evaluate( [=](double x) { return sin( x -0.3);}, grid);

    // the values are the derivative which is linear
    dg::HVec y0 = dg::evaluate( [=](double x) { return cos( x -0.0);}, grid);
    dg::HVec y1 = dg::evaluate( [=](double x) { return cos( x -0.1);}, grid);
    dg::HVec y2 = dg::evaluate( [=](double x) { return cos( x -0.2);}, grid);
    dg::HVec y  = dg::evaluate( [=](double x) { return cos( x -0.3);}, grid);
    double norm = dg::blas1::dot( y,y);
    dg::HVec guess ( x);
    dg::LeastSquaresExtrapolation<dg::HVec, dg::HVec> extraLS(3,guess, guess);
    dg::Extrapolation<dg::HVec> extra(3,guess);
    extra.update( 0., y0);
    extra.extrapolate( 0.3, guess);
    dg::blas1::axpby( 1., y, -1., guess);
    std::cout << "Difference    0 is "<<dg::blas1::dot( guess, guess) / norm<< " \n";
    extraLS.update( x0, y0);
    extraLS.extrapolate( x, guess);
    dg::blas1::axpby( 1., y, -1., guess);
    std::cout << "Difference LS 0 is "<<dg::blas1::dot( guess, guess) / norm<< " \n";
    extra.update( 0.1, y1);
    extra.extrapolate( 0.3, guess);
    dg::blas1::axpby( 1., y, -1., guess);
    std::cout << "Difference    1 is "<<dg::blas1::dot( guess, guess) / norm<< " \n";
    extraLS.update( x1, y1);
    extraLS.extrapolate( x, guess);
    dg::blas1::axpby( 1., y, -1., guess);
    std::cout << "Difference LS 1 is "<<dg::blas1::dot( guess, guess) / norm<< " \n";
    extra.update( 0.1, y1);
    extra.update( 0.2, y2);
    extra.extrapolate( 0.3, guess);
    dg::blas1::axpby( 1., y, -1., guess);
    std::cout << "Difference    2 is "<<dg::blas1::dot( guess, guess) / norm<< " \n";
    extraLS.update( x1, x1); // rejected
    extraLS.update( x2, y2); // also rejected
    extraLS.extrapolate( x, guess);
    dg::blas1::axpby( 1., y, -1., guess);
    std::cout << "Difference LS 2 is "<<dg::blas1::dot( guess, guess) / norm<< " \n";
    }


    return 0;
}
