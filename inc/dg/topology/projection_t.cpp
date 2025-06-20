#include <iostream>

#ifdef WITH_MPI
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "mpi_weights.h"
#include "mpi_projection.h"
#include "mpi_evaluation.h"
#endif

#include "dg/blas.h"
#include "derivatives.h"
#include "projection.h"
#include "evaluation.h"
#include "fast_interpolation.h"

#include <catch2/catch_all.hpp>

static double sine( double x){ return sin(x);}
static double sine( double x, double y, double){return sin(x)*sin(y);}
#ifndef WITH_MPI
static double sine( double x, double y){return sin(x)*sin(y);}
#endif
//Here, we test interpolation/projection/transform and fast versions between grids

TEST_CASE( "Projection")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm1d = dg::mpi_cart_create( MPI_COMM_WORLD, {0}, {1});
    MPI_Comm comm3d = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {1,1,1});
#endif
    using namespace Catch::Matchers;
    const std::vector<unsigned> ns = {3,3,6,6}, Ns = {7,14,7,14};
    const unsigned n_new = 3, N_new = 7;

    SECTION( "1d projection")
    {
        auto i = GENERATE( 0,1,2,3);
        unsigned n_old = ns[i], N_old = Ns[i];
        INFO( "Fine   Grid "<< n_old << " x "<<N_old );
        INFO( "Coarse Grid "<< n_new << " x "<<N_new );
        dg::x::Grid1d go ( 0, M_PI/2., n_old, N_old
#ifdef WITH_MPI
        , comm1d
#endif
        );
        dg::x::Grid1d gn ( 0, M_PI/2., n_new, N_new
#ifdef WITH_MPI
        , comm1d
#endif
        );
        dg::x::DMatrix proj = dg::create::fast_projection( 0, go, n_old/n_new,
            N_old/N_new);
        dg::x::DMatrix inte = dg::create::fast_interpolation(0, gn,
        n_old/n_new, N_old/N_new);
        dg::x::DVec v = dg::evaluate( sine, go);
        const dg::x::DVec w1do = dg::create::weights( go);
        const dg::x::DVec w1dn = dg::create::weights( gn);
        dg::x::DVec w = dg::evaluate( sine, gn);
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

        dg::x::DVec wP( w);
        dg::blas2::symv( proj, v, wP);
        dg::blas1::axpby( 1., wP, -1., w);
        double PI = dg::blas2::dot( w, w1dn, w);
        INFO( "Difference PI    "<<PI << " (Must be 0)");
        CHECK( PI < 1e-14);

        INFO(" LINEAR interpolation");
        dg::x::HVec xvec = dg::evaluate( dg::cooX1d, go);
        dg::x::IDMatrix inte_m = dg::create::interpolation(
#ifdef WITH_MPI
            xvec.data(),
#else
            xvec,
#endif
            gn, dg::DIR, "linear");
        w = dg::evaluate( sine, gn);
        dg::blas2::symv( inte_m, w, v);
        orginal = dg::blas1::dot( w1dn, w);
        project = dg::blas1::dot( w1do, v);
        INFO( "Original vector  "<<orginal);
        INFO( "Interpolated vector "<<project);
        INFO( "Difference       "<<orginal - project << " (Must be 0)");
        CHECK( fabs( orginal - project) < 1e-3);

        dg::blas2::symv( proj, v, wP);
        dg::blas1::axpby( 1., wP, -1., w);
        PI = dg::blas2::dot( w, w1dn, w);
        INFO( "Difference PI    "<<PI << " (Must be 0)");
        CHECK( PI < 1.4e-5);
    }

    SECTION( "3d projection")
    {
        auto i = GENERATE( 0,1,2,3); // ns, Ns
        unsigned n_old = ns[i], N_old = Ns[i];
        dg::x::Grid3d g2o (0, M_PI, 0, M_PI, 0,1, n_old, N_old, N_old, 4
#ifdef WITH_MPI
            , comm3d
#endif
        );
        dg::x::Grid3d g2n (0, M_PI, 0, M_PI, 0,1, n_new, N_new, N_new, 4
#ifdef WITH_MPI
            , comm3d
#endif
        );
        const dg::x::HVec w2do = dg::create::weights( g2o);
        const dg::x::HVec w2dn = dg::create::weights( g2n);
        const dg::x::HVec sinO( dg::evaluate( sine, g2o)),
                          sinN( dg::evaluate( sine, g2n));
        // Idea we test the original and then test that fast does the same
        SECTION( "Original projection")
        {
            dg::x::HVec sinP( sinN);
            auto proj2d = dg::create::projection( g2n, g2o);
            dg::blas2::gemv( proj2d, sinO, sinP);
            double value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
            double value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
            INFO( "Original vector     "<<value0);
            INFO( "Projected vector    "<<value1);
            INFO( "Difference in Norms "<<value0-value1);
            CHECK( fabs(value0 - value1) < 1e-6);

            dg::blas1::axpby( 1., sinN, -1., sinP);
            double value2 = sqrt( dg::blas2::dot( sinP, w2dn, sinP) /
                    dg::blas2::dot( sinN, w2dn, sinN));
            INFO( "Difference between projection and evaluation      "<<value2);
            CHECK( value2 < 1e-4);
        }
        SECTION( "Original interpolation")
        {
            dg::x::HVec sinI(sinO);
            auto inte2d = dg::create::interpolation( g2o, g2n);
            dg::blas2::gemv( inte2d, sinN, sinI);
            double value0 = sqrt(dg::blas2::dot( sinN, w2dn, sinN));
            double value1 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
            INFO( "Original vector     "<<value0);
            INFO( "Interpolated vec    "<<value1);
            INFO( "Difference in Norms "<<value0 - value1);
            CHECK( fabs(value0 - value1) < 1e-10);

            dg::blas1::axpby( 1., sinO, -1., sinI);
            double value2 = sqrt( dg::blas2::dot( sinI, w2do, sinI)/
                           dg::blas2::dot( sinO, w2do, sinO));
            INFO( "Difference between interpolation and evaluation   " <<value2);
            CHECK( value2 < 1e-3);
        }
        // Check that fast version does the same as original
        SECTION( "Fast projection")
        {
            // Check that fast projection does the same as original
            dg::x::HVec sinP_fast( sinN), sinP(sinN);
            auto proj2d = dg::create::projection( g2n, g2o);
            auto fast_proj2d = dg::create::fast_projection( g2o, n_old/n_new,
                N_old/N_new, N_old/N_new);
            dg::blas2::gemv( proj2d, sinO, sinP);
            dg::blas2::gemv( fast_proj2d, sinO, sinP_fast);
            dg::blas1::axpby( 1., sinP, -1., sinP_fast);
            double value0 = sqrt(dg::blas2::dot( sinP_fast, w2dn, sinP_fast));
            INFO( "Difference between original and fast p   " <<value0);
            CHECK( value0 < 1e-14);
        }

        SECTION( "Fast interpolation")
        {
            dg::x::HVec sinI_fast( sinO), sinI(sinO);
            auto inte2d = dg::create::interpolation( g2o, g2n);
            auto fast_inte2d = dg::create::fast_interpolation( g2n,
                n_old/n_new, N_old/N_new, N_old/N_new);
            dg::blas2::gemv( inte2d, sinN, sinI);
            dg::blas2::gemv( fast_inte2d, sinN, sinI_fast);
            dg::blas1::axpby( 1., sinI, -1., sinI_fast);
            double value0 = sqrt(dg::blas2::dot( sinI_fast, w2do, sinI_fast));
            INFO( "Difference between original and fast i   " <<value0);
            CHECK( value0 < 1e-14);
        }

        SECTION( "fast Transform")
        {
            // Transform
            auto forward = dg::create::fast_transform( dg::DLT<double>::forward(
                        n_old), dg::DLT<double>::forward( n_old), g2o);
            auto backward = dg::create::fast_transform( dg::DLT<double>::backward(
                        n_old), dg::DLT<double>::backward( n_old), g2o);

            dg::x::HVec sinF(sinO), sinI(sinO);
            dg::blas2::gemv( forward, sinO, sinF);
            dg::blas2::gemv( backward, sinF, sinI);
            dg::blas1::axpby( 1., sinO, -1., sinI);
            double value2 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
            INFO( "Forward-Backward Error   "<<value2 << " (Must be zero)");
            CHECK( value2 < 1e-14);
        }
    }

#ifndef WITH_MPI
    SECTION( "backproject 1d")
    {
        unsigned n=3, N = 20;
        dg::x::Grid1d g1d( 0.0, M_PI+0.0, n, N, dg::DIR);
        dg::x::Grid1d g1dequi( 0.0, M_PI, 1, n*N, dg::DIR);
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
        dg::x::Grid2d g2d(0., M_PI, 0., M_PI, 3, 10, 20);
        dg::x::Grid2d g2dequi = g2d;
        g2dequi.set( 1, g2d.shape(0), g2d.shape(1));
        auto w2d = dg::create::weights( g2d);
        auto w2dequi = dg::create::weights( g2dequi);
        auto proj = dg::create::backproject( g2d);
        auto inv_proj = dg::create::inv_backproject( g2d);
        auto v = dg::evaluate( sine, g2d), w=v, x=v;
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
#else
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    SECTION( "Correct mpi matrix conversion")
    {
        auto i = GENERATE( 0,1); // ns, Ns
        unsigned n_old = ns[i], N_old = Ns[i];
        dg::x::Grid3d g2o (0, M_PI, 0, M_PI, 0,1, n_old, N_old, N_old, 4
            , comm3d
        );
        dg::x::Grid3d g2n (0, M_PI, 0, M_PI, 0,1, n_new, N_new, N_new, 4
            , comm3d
        );
        auto method = GENERATE("dg", "linear", "cubic");
        INFO( "Method "<<method);
        dg::MIHMatrix inte2d = dg::create::interpolation(
                g2o, g2n, method);
        dg::IHMatrix inte2dg = dg::create::interpolation(
                g2o.global(), g2n.global(), method);

        const dg::x::HVec w2do = dg::create::weights( g2o);
        const dg::x::HVec w2dn = dg::create::weights( g2n);
        const dg::x::HVec sinO( dg::evaluate( sine, g2o)),
                          sinN( dg::evaluate( sine, g2n));
        dg::x::HVec sinP(sinN);
        dg::x::HVec sinI(sinO);
        dg::blas2::gemv( inte2d, sinN, sinI);

        double value0 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
        dg::HVec sinNg = dg::evaluate( sine, g2n.global()),
                 sinOg = dg::evaluate( sine, g2o.global());
        const dg::HVec w2dog = dg::create::weights( g2o.global());
        dg::blas2::gemv( inte2dg, sinNg, sinOg);
        double value1 = sqrt( dg::blas2::dot( sinOg, w2dog, sinOg));
        INFO( "MPI Interpolation: difference in Norms "<<value0 - value1);
        CHECK( fabs( value0 - value1) < 1e-14);

        dg::MIHMatrix project2d = dg::create::projection( g2n, g2o, method);
        dg::IHMatrix project2dg = dg::create::projection( g2n.global(),
                g2o.global(), method);
        dg::blas2::gemv( project2d, sinO, sinP);
        value0 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
        sinOg = dg::evaluate( sine, g2o.global());
        const dg::HVec w2dng = dg::create::weights( g2n.global());
        dg::blas2::gemv( project2dg, sinOg, sinNg);
        value1 = sqrt( dg::blas2::dot( sinNg, w2dng, sinNg));
        INFO( "MPI Projection   : difference in Norms "<<value0 - value1);
        CHECK( fabs( value0 - value1) < 1e-14);
    }
#endif
}
