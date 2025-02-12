#include <iostream>
#ifdef WITH_MPI
#include <mpi.h>
#include "../backend/mpi_init.h"
#endif
#include "evaluation.h"
#include "average.h"
#include "dg/blas.h"

#include "catch2/catch_all.hpp"

// 2d
static double function( double x, double y) {return cos(x)*sin(y);}
static double pol_average( double x) {return cos(x)*2./M_PI;}
static double pol_average( double x, double y) {return cos(x)*2./M_PI;}
static double tor_average( double y) {return sin(y)*2./M_PI;}
static double tor_average( double x, double y) {return sin(y)*2./M_PI;}

// 3d
static double function( double x, double y, double z) {return cos(x)*sin(z);}
static double z_average( double x, double y) {return cos(x)*2./M_PI;}
static double x_average( double x, double y, double z) {return sin(z)*2./M_PI;}

TEST_CASE( "2d Average in x and y direction")
{
#ifdef WITH_MPI
    int size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    std::vector<int> dims = {0,0};
    MPI_Dims_create( size, 2, &dims[0]);
    auto i = GENERATE( 0,1);
    std::sort( dims.begin(), dims.end());
    for( int u=0; u<i; u++)
        std::next_permutation( dims.begin(), dims.end());
    INFO( "Permutation of dims "<<dims[0]<<" "<<dims[1]);
    MPI_Comm comm2d = dg::mpi_cart_create( MPI_COMM_WORLD, dims, {1, 1});
#endif
    using namespace Catch::Matchers;
    const unsigned n = 3, Nx = 32, Ny = 48;
    const double lx = M_PI/2., ly = M_PI;
    const dg::x::Grid2d g( 0, lx, 0, ly, n, Nx, Ny
#ifdef WITH_MPI
                , comm2d
#endif
    );
    SECTION( "Averaging x ")
    {
        dg::Average< dg::x::IDMatrix, dg::x::DVec> tor( g, dg::coo2d::x);
        const dg::x::DVec vector = dg::evaluate( function ,g);
        auto average_x = vector;
        tor( vector, average_x, true);
        dg::x::DVec solution = dg::evaluate( tor_average, g);
        dg::blas1::axpby( 1., solution, -1., average_x);
        const dg::x::DVec w2d = dg::create::weights( g);
        double res = sqrt( dg::blas2::dot( average_x, w2d, average_x));
        INFO("Distance to solution is: "<<res<<"\t(Should be 0)");
        CHECK_THAT( res, WithinAbs( 0.0, 1e-13));
    }
    SECTION( "Avarage y")
    {
        //![doxygen]
        dg::Average< dg::x::IDMatrix, dg::x::DVec > pol(g, dg::coo2d::y);

        const dg::x::DVec vector = dg::evaluate( function ,g);
        dg::x::DVec average_y;
        pol( vector, average_y, false);
        const dg::x::DVec w1d = dg::create::weights( g.gx());
        dg::x::DVec solution = dg::evaluate( pol_average, g.gx());
        dg::blas1::axpby( 1., solution, -1., average_y);
        double res = sqrt( dg::blas2::dot( average_y, w1d, average_y));
        INFO("Distance to solution is: "<<res<<"\t(Should be 0)");
        CHECK_THAT( res, WithinAbs( 0.0, 1e-13));
        //![doxygen]
    }
    SECTION( "TEST prolongation y")
    {
        dg::x::IDMatrix prolong = dg::create::prolongation( g, std::array{1u});
        dg::x::DVec sol2d = dg::evaluate( pol_average, g), test(sol2d);
        dg::x::DVec sol1d = dg::evaluate( pol_average, g.gx());
        dg::blas2::symv( prolong, sol1d, test);
        dg::blas1::axpby( 1., sol2d, -1., test);
        double err = dg::blas1::dot( test, test);
        INFO("Distance to solution is: "<<sqrt(err)<<" (Must be 0)");
        CHECK_THAT( sqrt( err), WithinAbs( 0.0, 1e-13));
    }
    SECTION( "TEST prolongation x")
    {
        dg::x::IDMatrix prolong = dg::create::prolongation( g, std::array{0u});
        dg::x::DVec sol2d = dg::evaluate( tor_average, g), test(sol2d);
        dg::x::DVec sol1d = dg::evaluate( tor_average, g.gy());
        dg::blas2::symv( prolong, sol1d, test);
        dg::blas1::axpby( 1., sol2d, -1., test);
        double err = dg::blas1::dot( test, test);
        INFO( "Distance to solution is: "<<sqrt(err)<<" (Must be 0)");
        CHECK_THAT( sqrt( err), WithinAbs( 0.0, 1e-13));
    }
}

TEST_CASE( "3d Average in x and z direction")
{
#ifdef WITH_MPI
    int size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    std::vector<int> dims = {0,0,0};
    MPI_Dims_create( size, 3, &dims[0]);
    auto i = GENERATE( 0,1,2,3,4,5);
    std::sort( dims.begin(), dims.end());
    for( int u=0; u<i; u++)
         std::next_permutation( dims.begin(), dims.end());
    INFO( "Permutation of dims "<<dims[0]<<" "<<dims[1]<<" "<<dims[2]);
    MPI_Comm comm3d = dg::mpi_cart_create( MPI_COMM_WORLD, dims, {1, 1, 1});
#endif
    using namespace Catch::Matchers;
    unsigned n = 3, Nx = 32, Ny = 48, Nz = 64;
    const double lx = M_PI/2., ly = M_PI, lz = M_PI/2.;
    dg::x::Grid3d g( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz
#ifdef WITH_MPI
                , comm3d
#endif
    );
    const dg::x::DVec vector = dg::evaluate( function ,g);
    SECTION( "Averaging x ")
    {
        dg::Average<dg::x::IDMatrix, dg::x::DVec > tor(g, dg::coo3d::x);
        auto average_z = vector;
        tor( vector, average_z);
        dg::x::DVec solution = dg::evaluate( x_average, g);
        dg::blas1::axpby( 1., solution, -1., average_z);
        const dg::x::DVec w3d = dg::create::weights( g);
        dg::exblas::udouble res;
        res.d = sqrt( dg::blas2::dot( average_z, w3d, average_z));
        INFO("Solution is: "<<res.d<<"\t"<<res.i);
        CHECK_THAT( res.d, WithinAbs( 0.0, 1e-13));
    }
    SECTION("Averaging z")
    {
        dg::Average<dg::x::IDMatrix, dg::x::DVec > avg(g, dg::coo3d::z);
        dg::x::DVec average_z;
        avg( vector, average_z, false);

        dg::x::Grid2d gxy{ g.gx(), g.gy()};
        const dg::x::DVec w2d = dg::create::weights( gxy);
        dg::x::DVec solution = dg::evaluate( z_average, gxy);
        dg::blas1::axpby( 1., solution, -1., average_z);
        dg::exblas::udouble res;
        res.d = sqrt( dg::blas2::dot( average_z, w2d, average_z));
        INFO( "(Converges with 2nd order).");
        INFO( "Solution is: "<<res.d<<"\t"<<res.i);
        CHECK_THAT( res.d, WithinAbs( 0.0, 2.6e-5));
    }

}
