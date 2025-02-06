#include <iostream>
#include <mpi.h>

#include "dg/backend/mpi_init.h"
#include "mpi_evaluation.h"
#include "stencil.h"
#include "filter.h"
#include "../blas2.h"

#include "catch2/catch_all.hpp"

// TODO Fix Test gets stuck with mpirun -n 4

TEST_CASE( "MPI stencil")
{
    const double lx = 2.*M_PI;
    const double ly = 2.*M_PI;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm2dPER = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
    MPI_Comm comm2d    = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {0,0});
    auto bc = GENERATE( dg::DIR, dg::NEU, dg::PER);
    INFO( "Boundary "<<dg::bc2str( bc));
    dg::MPIGrid2d g2d( 0,lx, 0,ly, 3, 40, 20, bc, bc, bc == dg::PER ? comm2dPER :
        comm2d);
    const auto w2d = dg::create::weights( g2d);

    // We just test that MPI Version does the same as serial version
    auto x = dg::evaluate( dg::one, g2d), y(x);
    auto xg = dg::evaluate( dg::one, g2d.global()), yg(xg);
    SECTION( "Window stencil")
    {
        auto stencil = dg::create::window_stencil( {3,3}, g2d, bc, bc);
        dg::blas2::symv( stencil, x, y);

        auto stencilg = dg::create::window_stencil( {3,3}, g2d.global(), bc, bc);
        dg::blas2::symv( stencilg, xg, yg);


    }
    SECTION("Test filtered symv")
    {
        auto stencil = dg::create::window_stencil( {3,3}, g2d, bc, bc);
        dg::blas2::stencil( dg::CSRSymvFilter(), stencil, x, y);

        auto stencilg = dg::create::window_stencil( {3,3}, g2d.global(), bc, bc);
        dg::blas2::stencil( dg::CSRSymvFilter(), stencilg, xg, yg);

    }
    // Compare local part of gy is same as y
    auto yl = dg::global2local( yg, g2d);
    dg::blas1::axpby( 1., yl, -1., y);
    double err = sqrt( dg::blas2::dot( y, w2d, y));
    CHECK( err < 1e-15);
}
