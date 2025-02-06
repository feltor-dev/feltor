#include <iostream>
#include <iomanip>
#ifdef WITH_MPI
#include <mpi.h>
#include "mpi_base_geometry.h"
#include "../backend/mpi_init.h"
#endif
#include "base_geometry.h"
#include "catch2/catch_all.hpp"


TEST_CASE( "Base geometry")
{
    SECTION( "2d shapes are correct")
    {
#ifdef WITH_MPI
        MPI_Comm comm2d = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {0,0});
#endif
        dg::x::CartesianGrid2d g2d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10,
                dg::DIR, dg::DIR
#ifdef WITH_MPI
                , comm2d
#endif
        );
        CHECK( g2d.shape(0) == 30);
        CHECK( g2d.shape(1) == 30);

        dg::x::CartesianGrid2d g2d_test{ g2d.gx(),g2d.gy()};
        CHECK( g2d_test.shape(0) == 30);
        CHECK( g2d_test.shape(1) == 30);
    }
    SECTION( "3d shapes are correct")
    {
#ifdef WITH_MPI
        MPI_Comm comm3d = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {0,0,0});
#endif
        dg::x::CartesianGrid3d g3d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0.,
                2.*M_PI, 3, 10, 10, 10, dg::DIR, dg::DIR, dg::PER
#ifdef WITH_MPI
                , comm3d
#endif
        );
        CHECK( g3d.shape(0) == 30);
        CHECK( g3d.shape(1) == 30);
        CHECK( g3d.shape(2) == 10);
        dg::x::CylindricalGrid3d c3d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0.,
                2.*M_PI, 3, 10, 10, 10, dg::DIR, dg::DIR, dg::PER
#ifdef WITH_MPI
                , comm3d
#endif
        );
        CHECK( c3d.shape(0) == 30);
        CHECK( c3d.shape(1) == 30);
        CHECK( c3d.shape(2) == 10);
    }
}
