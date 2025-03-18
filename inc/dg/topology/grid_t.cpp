#include <iostream>
#include <iomanip>
#include "interpolation.h"
#include "grid.h"
#include "catch2/catch_all.hpp"


TEST_CASE( "The grid class")
{
    const unsigned n = 3, Nx = 9, Ny = 5, Nz = 4;
    SECTION("Grid2d set functions")
    {
        dg::Grid2d g( -M_PI, 0, -5*M_PI, -4*M_PI, n, Nx, Ny);
        CHECK( g.nx() == 3);
        CHECK( g.ny() == 3);
        CHECK( g.Nx() == 9);
        CHECK( g.Ny() == 5);
        g.set(2,7,3);
        CHECK( g.nx() == 2);
        CHECK( g.ny() == 2);
        CHECK( g.Nx() == 7);
        CHECK( g.Ny() == 3);
    }
    SECTION( "Grid3d set functions")
    {
        dg::Grid3d g( -10, 10, -5, 5, -7, -3, n, Nx, Ny, Nz);
        CHECK( g.nx() == 3);
        CHECK( g.ny() == 3);
        CHECK( g.nz() == 1);

        CHECK( g.Nx() == 9);
        CHECK( g.Ny() == 5);
        CHECK( g.Nz() == 4);

        g.set( 2,5,7,3);
        CHECK( g.nx() == 2);
        CHECK( g.ny() == 2);
        CHECK( g.nz() == 1);

        CHECK( g.Nx() == 5);
        CHECK( g.Ny() == 7);
        CHECK( g.Nz() == 3);

        g.set( {n,n,1}, {Nx, Ny,Nz});
        CHECK( g.nx() == 3);
        CHECK( g.ny() == 3);
        CHECK( g.nz() == 1);

        CHECK( g.Nx() == 9);
        CHECK( g.Ny() == 5);
        CHECK( g.Nz() == 4);
    }
    SECTION( "The shift function")
    {
        dg::Grid1d g1d( 1., 1.+2.*M_PI, 3, 10, dg::PER);
        auto bcx = GENERATE( dg::PER, dg::NEU, dg::DIR, dg::DIR_NEU, dg::NEU_DIR);
        double x = -10.*M_PI;
        for( int i =0; i<11; i++)
        {
            double x0 = x + i*2*M_PI;
            bool mirrored = false;
            INFO( "Point: "<<x0<<" bc: "<<dg::bc2str( bcx));
            dg::create::detail::shift( mirrored, x0, bcx, g1d.x0(), g1d.x1());
            INFO( "shifted "<< mirrored<<" "<<x0);
            if( bcx == dg::PER)
            {
                CHECK( ( x0-2.*M_PI)<1e-15 );
                CHECK( not mirrored);
            }
            else
            {
                if( i%2 != 0)
                    CHECK( ( x0-2.)<1e-15 );
                else
                    CHECK( ( x0-2.*M_PI)<1e-15 );
            }
            if ( bcx == dg::NEU)
            {
                CHECK( not mirrored);
            }
            else if( bcx == dg::DIR)
            {
                if( i%2 == 0)
                    CHECK( not mirrored);
                else
                    CHECK( mirrored);
            }
            else if( bcx == dg::DIR_NEU)
            {
                if( (i/2)%2 == 0)
                    CHECK( mirrored);
                else
                    CHECK( not mirrored);
            }
            else if( bcx == dg::NEU_DIR)
            {
                if( ((i+1)/2)%2 == 0)
                    CHECK( mirrored);
                else
                    CHECK( not mirrored);
            }
        }
    }
    SECTION( "Test 2d and 3d shift")
    {
        dg::Grid2d g2d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10,
                dg::DIR, dg::DIR);
        dg::Grid2d g2d_test{ g2d.gx(),g2d.gy()};
        dg::Grid3d g3d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0., 2.*M_PI, 3, 10, 10, 10,
            dg::DIR, dg::DIR, dg::PER);
        double y=0;
        for( int i=0; i<2; i++)
        {
            std::array<double,2> p = {0,y+i*2.*M_PI};
            INFO( "Point "<< p[1]);
            bool mirrored = false;
            dg::create::detail::shift( mirrored, p[0], g2d.bcx(), g2d.x0(),
                    g2d.x1());
            dg::create::detail::shift( mirrored, p[1], g2d.bcy(), g2d.y0(),
                    g2d.y1());
            CHECK( ( i==0 ? not mirrored : mirrored));
            CHECK( p[0] == 2);
            CHECK( ( i==0 ? p[1] == 2 : p[1] == 2.*M_PI));
            std::array<double,3> q = {0,y+i*2.*M_PI, 2.*M_PI};
            mirrored = false;
            dg::create::detail::shift( mirrored, q[0], g3d.bcx(), g3d.x0(),
                    g3d.x1());
            dg::create::detail::shift( mirrored, q[1], g3d.bcy(), g3d.y0(),
                    g3d.y1());
            dg::create::detail::shift( mirrored, q[2], g3d.bcz(), g3d.z0(),
                    g3d.z1());
            CHECK( ( i==0 ? not mirrored : mirrored));
            CHECK( q[0] == 2);
            CHECK( ( i==0 ? q[1] == 2 : q[1] == 2.*M_PI));
            CHECK( q[2] == 0);
        }
    }
    SECTION( "Test 1d contains function")
    {
        dg::Grid1d g1d( 1., 1.+2.*M_PI, 3, 10, dg::PER);
        double x_in = M_PI, x_out = 3.*M_PI;
        CHECK( g1d.contains( x_in));
        CHECK( not g1d.contains( x_out));
        x_out = 0.;
        CHECK( not g1d.contains( x_out));
    }
    SECTION("Test 2d contains function\n")
    {
        dg::Grid2d g2d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10,
                dg::DIR, dg::DIR);
        double x_in = M_PI;
        CHECK( g2d.contains( std::array{x_in, x_in}));
        std::array<double,2> p_out[] = {{0, M_PI}, {0,0},
            {3*M_PI,3*M_PI},{M_PI,0}, {0,3*M_PI}};
        for( int i=0; i<4; i++)
            CHECK( !g2d.contains( p_out[i]));

        std::array<unsigned,2> zero={0,0};
        CHECK( g2d.start() == zero);
    }
}
