
#include <iostream>
#include "elliptic.h"
#include "catch2/catch_all.hpp"

double func1d ( double x) { return sin(2*M_PI*x);}
double lapM_func1d ( double x) { return 4*M_PI*M_PI*sin(2*M_PI*x);}

double func2d ( double x, double y) { return sin(2*M_PI*x)*sin(2*M_PI*y);}
double lapM_func2d ( double x, double y) { return 2*4*M_PI*M_PI*(sin(2*M_PI*x)*sin(2*M_PI*y));}

double func3d ( double x, double y, double z) { return sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z);}
double lapM_func3d ( double x, double y, double z) { return 3*4*M_PI*M_PI*(sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z));}

TEST_CASE( "Elliptic1d")
{
    unsigned n = 19, N = 4;
    dg::Grid1d g1d( 0,1, n, N, dg::PER);
    const dg::DVec x = dg::evaluate( func1d, g1d);
    const dg::DVec w1d = dg::create::weights( g1d);
    const dg::DVec ysol = dg::evaluate( lapM_func1d, g1d);
    SECTION( "Construction")
    {
        const dg::Elliptic1d<dg::Grid1d, dg::DMatrix, dg::DVec> elliptic(g1d);
        CHECK( elliptic.get_jfactor() == 1);
        const auto& sigma = elliptic.get_chi();
        CHECK( dg::blas1::dot( sigma, sigma) == g1d.size());
    }
    SECTION( "Set and get")
    {
        dg::Elliptic1d<dg::Grid1d, dg::DMatrix, dg::DVec> elliptic(g1d);
        elliptic.set_jfactor( 10);
        CHECK( elliptic.get_jfactor() == 10);
        elliptic.set_chi( 4.0);
        const auto& sigma = elliptic.get_chi();
        CHECK( dg::blas1::dot( sigma, sigma) == 16*g1d.size());
    }

    SECTION( "Convergence")
    {
        auto dir = GENERATE( as<dg::direction>{}, dg::forward, dg::backward, dg::centered);
        const dg::Elliptic1d<dg::Grid1d, dg::DMatrix, dg::DVec> elliptic(g1d, dir);
        dg::DVec y(x);
        elliptic.symv( 1., x, 1., 2., 0, y);
        dg::blas1::axpby( 1., ysol, -1./2., y);
        CHECK ( dg::blas2::dot( y, w1d, y)  < 1e-16*dg::blas2::dot( x, w1d, x));
    }
    SECTION( "Convergence with sigma")
    {
        auto dir = GENERATE( as<dg::direction>{}, dg::forward, dg::backward, dg::centered);
        dg::Elliptic1d<dg::Grid1d, dg::DMatrix, dg::DVec> elliptic(g1d, dir);
        elliptic.set_chi( 10);
        dg::DVec y(x);
        elliptic.symv( 1., x, 0, y);
        dg::blas1::axpby( 1., ysol, -1./10., y);
        CHECK ( dg::blas2::dot( y, w1d, y)  < 1e-16*dg::blas2::dot( x, w1d, x));
    }
}
TEST_CASE( "Elliptic2d")
{
    unsigned nx = 19, ny = 17, Nx = 4, Ny = 5;
    dg::CartesianGrid2d g2d( dg::Grid1d{ 0, 1, nx, Nx}, dg::Grid1d{ 0, 1, ny, Ny} );
    const dg::DVec x = dg::evaluate( func2d, g2d);
    const dg::DVec w2d = dg::create::weights( g2d);
    const dg::DVec ysol = dg::evaluate( lapM_func2d, g2d);
    SECTION( "Construction")
    {
        const dg::Elliptic2d<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> elliptic(g2d);
        CHECK( elliptic.get_jfactor() == 1);
        const auto& sigma = elliptic.get_sigma();
        CHECK( dg::blas1::dot( sigma, sigma) == g2d.size());
    }
    SECTION( "Set and get")
    {
        dg::Elliptic2d<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> elliptic(g2d);
        elliptic.set_jfactor( 10);
        CHECK( elliptic.get_jfactor() == 10);
        elliptic.set_chi( 4.0);
        const auto& sigma = elliptic.get_sigma();
        CHECK( dg::blas1::dot( sigma, sigma) == 16*g2d.size());
    }

    SECTION( "Convergence")
    {
        auto dir = GENERATE( as<dg::direction>{}, dg::forward, dg::backward, dg::centered);
        const dg::Elliptic2d<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> elliptic(g2d, dir);
        dg::DVec y(x);
        elliptic.symv( 1., x, 1., 2., 0, y);
        dg::blas1::axpby( 1., ysol, -1./2., y);
        CHECK ( dg::blas2::dot( y, w2d, y)  < 1e-16*dg::blas2::dot( x, w2d, x));
    }
    SECTION( "Convergence with sigma")
    {
        auto dir = GENERATE( as<dg::direction>{}, dg::forward, dg::backward, dg::centered);
        dg::Elliptic2d<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> elliptic(g2d, dir);
        elliptic.set_chi( 10);
        dg::DVec y(x);
        elliptic.symv( 1., x, 0, y);
        dg::blas1::axpby( 1., ysol, -1./10., y);
        CHECK ( dg::blas2::dot( y, w2d, y)  < 1e-16*dg::blas2::dot( x, w2d, x));
    }
}
TEST_CASE( "Elliptic3d")
{
    unsigned nx = 19, ny = 17, Nx = 4, Ny = 5, nz = 18, Nz = 6;
    dg::CartesianGrid3d g3d( dg::Grid1d{ 0, 1, nx, Nx}, dg::Grid1d{ 0, 1, ny, Ny}, dg::Grid1d{0,1,nz,Nz} );
    const dg::DVec x = dg::evaluate( func3d, g3d);
    const dg::DVec w3d = dg::create::weights( g3d);
    const dg::DVec ysol = dg::evaluate( lapM_func3d, g3d);
    SECTION( "Construction")
    {
        const dg::Elliptic3d<dg::CartesianGrid3d, dg::DMatrix, dg::DVec> elliptic(g3d);
        CHECK( elliptic.get_jfactor() == 1);
        const auto& sigma = elliptic.get_sigma();
        CHECK( dg::blas1::dot( sigma, sigma) == g3d.size());
    }
    SECTION( "Set and get")
    {
        dg::Elliptic3d<dg::CartesianGrid3d, dg::DMatrix, dg::DVec> elliptic(g3d);
        elliptic.set_jfactor( 10);
        CHECK( elliptic.get_jfactor() == 10);
        elliptic.set_chi( 4.0);
        const auto& sigma = elliptic.get_sigma();
        CHECK( dg::blas1::dot( sigma, sigma) == 16*g3d.size());
    }

    SECTION( "Convergence")
    {
        auto dir = GENERATE( as<dg::direction>{}, dg::forward, dg::backward, dg::centered);
        const dg::Elliptic3d<dg::CartesianGrid3d, dg::DMatrix, dg::DVec> elliptic(g3d, dir);
        dg::DVec y(x);
        elliptic.symv( 1., x, 1., 2., 0, y);
        dg::blas1::axpby( 1., ysol, -1./2., y);
        CHECK ( dg::blas2::dot( y, w3d, y)  < 1e-16*dg::blas2::dot( x, w3d, x));
    }
    SECTION( "Convergence with sigma")
    {
        auto dir = GENERATE( as<dg::direction>{}, dg::forward, dg::backward, dg::centered);
        dg::Elliptic3d<dg::CartesianGrid3d, dg::DMatrix, dg::DVec> elliptic(g3d, dir);
        elliptic.set_chi( 10);
        dg::DVec y(x);
        elliptic.symv( 1., x, 0, y);
        dg::blas1::axpby( 1., ysol, -1./10., y);
        CHECK ( dg::blas2::dot( y, w3d, y)  < 1e-16*dg::blas2::dot( x, w3d, x));
    }
}
