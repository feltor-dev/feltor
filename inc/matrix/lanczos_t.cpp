
#include <iostream>
#include "functors.h"

#include "lanczos.h"

#include "catch2/catch_all.hpp"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const dg::bc bcx = dg::DIR;
const dg::bc bcy = dg::DIR;

using Matrix = dg::DMatrix;
using Container = dg::DVec;

TEST_CASE( "Lanczos")
{
    unsigned n = 3, Nx = 16, Ny = 16;
    dg::CartesianGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy);
    const Container w2d = dg::create::weights( grid);
    const Container rnd = dg::evaluate( dg::RandomNumbers<double>(0,1), grid);

    dg::mat::UniversalLanczos<Container> lanczos( w2d, 2000);
    SECTION( "Dirichlet")
    {
        grid.set_bcs( {dg::DIR, dg::DIR});
        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> ell( {grid, dg::centered, 1.0});
        //lanczos.set_verbose(true);
        auto T = lanczos.tridiag( ell, rnd, w2d);
        auto extremeEVs = dg::mat::compute_extreme_EV( T);
        // Eigenvalues computed with pyfeltor and scipy
        CHECK( extremeEVs[0] - 0.5000000674740441 < 1e-3);
        CHECK( 895.3216230750616 - extremeEVs[1] < 1e-3*895);
    }
    SECTION( "NEU")
    {
        grid.set_bcs( {dg::NEU, dg::NEU});
        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> ell( {grid, dg::centered, 1.0});
        //lanczos.set_verbose(true);
        auto T = lanczos.tridiag( ell, rnd, w2d);
        auto extremeEVs = dg::mat::compute_extreme_EV( T);

        CHECK( extremeEVs[0] - 0.0 < 1e-10*897);
        CHECK( 897.6777549436483 - extremeEVs[1] < 1e-3*897);
    }
    SECTION( "PER")
    {
        grid.set_bcs( {dg::PER, dg::PER});
        dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> ell( {grid, dg::centered, 1.0});
        //lanczos.set_verbose(true);
        auto T = lanczos.tridiag( ell, rnd, w2d );
        auto extremeEVs = dg::mat::compute_extreme_EV( T);

        CHECK( extremeEVs[0] - 0.0 < 1e-10*897);
        CHECK( 896.6482950890147 - extremeEVs[1] < 1e-3*897);

        auto Tmax = lanczos.tridiag( ell, rnd, w2d, 1e-4, 1, "compute_max_EV" );
        extremeEVs = dg::mat::compute_extreme_EV( Tmax);

        CHECK( 896.6482950890147 - extremeEVs[1] < 1e-3*897);
        // the max EV needs fewer iterations!
        CHECK( Tmax.size() < T.size());
    }
}
