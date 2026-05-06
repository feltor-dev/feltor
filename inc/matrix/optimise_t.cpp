#include <iostream>
#include "functors.h"

#include "optimise.h"

#include "catch2/catch_all.hpp"
// Taken from scipy least_squares doc

void fun_rosenbrock( const std::array<double,2>& x, std::array<double,2>& rs)
{
    rs[0] = 10 * (x[1]-x[0]*x[0]);
    rs[1] = ( 1 - x[0]);
    // f(x) = 0.5 (rs[0]^2 + rs[1]^2) has a global minimum at x = (1, 1) where f =0
}
void jac_rosenbrock( const std::array<double,2>& x, std::vector<std::array<double,2>>& jacs)
{
    jacs[0] = {-20*x[0],  -1};
    jacs[1] = {10, 0};
}

TEST_CASE("Rosenbrock")
{
    std::array<double,2> x0 = {2,2};
    unsigned num_steps = dg::mat::levenberg_marquardt(  fun_rosenbrock, jac_rosenbrock, x0, x0);
    INFO( "Found opt "<<x0[0]<<" "<<x0[1]<<" in "<< num_steps<<"steps");
    CHECK( x0[0] == 1);
    CHECK( x0[1] == 1);


}
