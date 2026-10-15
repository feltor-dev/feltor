
#include <iostream>
#include "functors.h"

#include "matrixsqrt.h"

#include "catch2/catch_all.hpp"


double rho_ana( double x, double y)  { return sin(x)*sin(y);}
double sol_ana( double x, double y)  { return sqrt(3) * (sin(x)*sin(y));}

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const dg::bc bcx = dg::DIR;
const dg::bc bcy = dg::PER;

using Matrix = dg::DMatrix;
using Container = dg::DVec;

TEST_CASE( "MatrixSqrt")
{
    // Compute Sqrt( 1-Laplace ) sin(x)sin(y) = Sqrt(3) sin(x)sin(y)

    const unsigned n = 3, Nx = 64, Ny = 64;
    double eps_gamma = 1e-7;

    dg::CartesianGrid2d grid2d( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);

    const Container w2d = dg::create::weights( grid2d);
    const Container rho = dg::evaluate( rho_ana, grid2d);
    const Container sol = dg::evaluate( sol_ana, grid2d);
    Container x(rho), error(rho);

    dg::Helmholtz< dg::CartesianGrid2d, Matrix, Container > gamma0inv(
          -1, {grid2d, bcx, bcy, dg::centered, 1.0});
    dg::mat::MatrixSqrt<Container> sqrtsolve( gamma0inv, +1, w2d,
            eps_gamma);
    sqrtsolve.set_benchmark(false);

    dg::apply( sqrtsolve, rho, x);

    dg::blas1::axpby( 1., sol, -1., x, error);
    double res = sqrt( dg::blas2::dot( w2d, error));
    //std::cout << "    error_abs: " << res<<std::endl;
    //std::cout << "    error_rel: " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
    CHECK( res < 1e-4);
}
