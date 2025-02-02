#include <iostream>
#include <cusp/print.h>
#include "dg/blas.h"
#include "derivatives.h"
#include "projection.h"
#include "evaluation.h"
#include "fast_interpolation.h"

#include "catch2/catch.hpp"

static double sine( double x){ return sin(x);}
static double sine( double x, double y){return sin(x)*sin(y);}
static double sine( double x, double y, double z){return sin(x)*sin(y);}
//Actually this file is a test for fast_interpolation

TEST_CASE( "Projection")
{
    using namespace Catch::Matchers;
    std::array<unsigned,2> ns = {3,9}, Ns = {20,40};

    SECTION( "1d projection")
    {
        auto i = GENERATE( 0,1);
        unsigned n_old = ns[i], n_new = 3, N_old = Ns[i], N_new = 20;
        INFO( "Fine   Grid "<< n_old << " x "<<N_old );
        INFO( "Coarse Grid "<< n_new << " x "<<N_new );
        dg::Grid1d go ( 0, M_PI/2., n_old, N_old);
        dg::Grid1d gn ( 0, M_PI/2., n_new, N_new);
        dg::DMatrix proj = dg::create::fast_projection1d( go, n_old/n_new,
                N_old/N_new);
        dg::DMatrix inte = dg::create::fast_interpolation1d( gn, n_old/n_new,
                N_old/N_new);
        dg::DVec v = dg::evaluate( sine, go);
        dg::DVec w1do = dg::create::weights( go);
        dg::DVec w1dn = dg::create::weights( gn);
        dg::DVec w( gn.size());
        dg::blas2::symv( proj, v, w);
        double orginal = dg::blas1::dot( w1do, v);
        double project = dg::blas1::dot( w1dn, w);
        INFO( "Original vector  "<<orginal);
        INFO( "Projected vector "<<project);
        INFO( "Difference       "<<orginal - project << " (Must be 0)");
        CHECK( fabs( orginal - project) < 1e-14);

        w = dg::evaluate( sine, gn);
        dg::blas2::symv( inte, w, v);
        orginal = dg::blas1::dot( w1dn, w);
        project = dg::blas1::dot( w1do, v);
        INFO( "Original vector  "<<orginal);
        INFO( "Interpolated vector "<<project);
        INFO( "Difference       "<<orginal - project << " (Must be 0)");
        CHECK( fabs( orginal - project) < 1e-14);

        dg::DVec wP( w);
        dg::blas2::symv( proj, v, wP);
        dg::blas1::axpby( 1., wP, -1., w);
        double PI = dg::blas2::dot( w, w1dn, w);
        INFO( "Difference PI    "<<PI << " (Must be 0)");
        CHECK( PI < 1e-14);

        INFO(" LINEAR interpolation");
        dg::HVec xvec = dg::evaluate( dg::cooX1d, go);
        dg::IDMatrix inte_m = dg::create::interpolation( xvec, gn, dg::DIR,
            "linear");
        w = dg::evaluate( sine, gn);
        dg::blas2::symv( inte_m, w, v);
        orginal = dg::blas1::dot( w1dn, w);
        project = dg::blas1::dot( w1do, v);
        INFO( "Original vector  "<<orginal);
        INFO( "Interpolated vector "<<project);
        INFO( "Difference       "<<orginal - project << " (Must be 0)");
        CHECK( fabs( orginal - project) < 2.7e-5);

        dg::blas2::symv( proj, v, wP);
        dg::blas1::axpby( 1., wP, -1., w);
        PI = dg::blas2::dot( w, w1dn, w);
        INFO( "Difference PI    "<<PI << " (Must be 0)");
        CHECK( PI < 1.4e-7);
    }

    SECTION( "3d projection")
    {
        auto i = GENERATE( 0,1);
        unsigned n_old = ns[i], n_new = 3, N_old = Ns[i], N_new = 20;
        dg::Grid3d g2o (0, M_PI, 0, M_PI, 0,1, n_old, N_old, N_old, 4);
        dg::Grid3d g2n (0, M_PI, 0, M_PI, 0,1, n_new, N_new, N_new, 4);
        cusp::coo_matrix<int, double, cusp::host_memory> inte2d =
            dg::create::interpolation( g2n, g2o);
        auto proj2d = dg::create::fast_projection( g2o, n_old/n_new,
                N_old/N_new, N_old/N_new);
        auto fast_inte2d = dg::create::fast_interpolation( g2n, n_old/n_new,
                N_old/N_new, N_old/N_new);
        auto forward = dg::create::fast_transform( dg::DLT<double>::forward(
                    n_old), dg::DLT<double>::forward( n_old), g2o);
        auto backward = dg::create::fast_transform( dg::DLT<double>::backward(
                    n_old), dg::DLT<double>::backward( n_old), g2o);
        const dg::HVec sinO( dg::evaluate( sine, g2o)),
                       sinN( dg::evaluate( sine, g2n));
        dg::HVec w2do = dg::create::weights( g2o);
        dg::HVec w2dn = dg::create::weights( g2n);
        dg::HVec sinP( sinN), sinI(sinO), sinF(sinO);
        dg::blas2::gemv( proj2d, sinO, sinP); //FAST PROJECTION
        double value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
        double value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
        INFO( "Original vector     "<<value0);
        INFO( "Projected vector    "<<value1);
        INFO( "Difference in Norms "<<value0-value1);
        CHECK( fabs(value0 - value1) < 2.5e-10);

        dg::blas1::axpby( 1., sinN, -1., sinP);
        double value2 = sqrt( dg::blas2::dot( sinP, w2dn, sinP) /
                dg::blas2::dot( sinN, w2dn, sinN));
        INFO( "Difference between projection and evaluation      "<<value2);
        CHECK( value2 < 1.8e-7);

        dg::blas2::gemv( inte2d, sinO, sinP);
        value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
        value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
        INFO( "Interpolated vec    "<<value1);
        INFO( "Difference in Norms "<<value0 - value1);
        CHECK( fabs(value0 - value1) < 1e-10);

        dg::blas2::gemv( fast_inte2d, sinN, sinI);
        value2 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
        double value3 = sqrt(dg::blas2::dot( sinN, w2dn, sinN));
        INFO( "Fast Interpolated vec "<< value2);
        INFO( "Difference in Norms   "<<value2 - value3);
        CHECK( fabs(value2 - value3) < 1e-10);

        dg::blas2::gemv( forward, sinO, sinF);
        dg::blas2::gemv( backward, sinF, sinI);
        dg::blas1::axpby( 1., sinO, -1., sinI);
        value2 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
        INFO( "Forward-Backward Error   "<<value2 << " (Must be zero)");
        CHECK( value2 < 1e-14);
    }

    SECTION( "backproject 1d")
    {
        unsigned n=3, N = 20;
        dg::Grid1d g1d( 0.0, M_PI+0.0, n, N, dg::DIR);
        dg::Grid1d g1dequi( 0.0, M_PI, 1, n*N, dg::DIR);
        auto w1d = dg::create::weights( g1d);
        auto w1dequi = dg::create::weights( g1dequi);
        auto proj = dg::create::backproject( g1d);
        auto inv_proj = dg::create::inv_backproject( g1d);
        auto v = dg::evaluate( sine, g1d), w(v), x(v);
        dg::blas2::symv( proj, v, w);
        double integral = dg::blas1::dot( v, w1d);
        double integralequi = dg::blas1::dot( w, w1dequi);
        INFO( "Error Integral is "<<(integral-integralequi)<<" (Must be zero)");
        CHECK( fabs( integral - integralequi) <1e-14);
        dg::blas2::symv( inv_proj, w, x);
        dg::blas1::axpby( 1., v, -1., x);
        double err = sqrt(dg::blas1::dot( x, x));
        INFO( "Error is "<<err<<" (Must be zero)");
        CHECK( err < 1e-14);
    }

    SECTION( "Test backproject 2d");
    {
        dg::Grid2d g2d(0., M_PI, 0., M_PI, 3, 10, 20);
        dg::Grid2d g2dequi = g2d;
        g2dequi.set( 1, g2d.shape(0), g2d.shape(1));
        auto w2d = dg::create::weights( g2d);
        auto w2dequi = dg::create::weights( g2dequi);
        auto proj = dg::create::backproject( g2d);
        auto inv_proj = dg::create::inv_backproject( g2d);
        dg::DVec v = dg::evaluate( sine, g2d), w=v, x=v;
        dg::blas2::symv( proj, v, w);
        double integral = dg::blas1::dot( v, w2d);
        double integralequi = dg::blas1::dot( w, w2dequi);
        INFO( "Error Integral 2d is "<<(integral-integralequi)<<" (Must be zero)");
        CHECK( fabs( integral - integralequi) <1e-14);
        dg::blas2::symv( inv_proj, w, x);
        dg::blas1::axpby( 1., v, -1., x);
        double err = sqrt(dg::blas1::dot( x, x));
        INFO( "Error 2d is "<<sqrt(err)<<" (Must be zero)");
        CHECK( err < 1e-14);
    }
}
