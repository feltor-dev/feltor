#include <iostream>

#include "dg/blas.h"
#include "dx.h"
#include "derivatives.h"
#include "derivativesT.h"
#include "evaluation.h"
#include "weights.h"

#include "catch2/catch.hpp"

static double function( double x) { return sin(x);}
static double derivative( double x) { return cos(x);}

using Vector = dg::HVec;
using Matrix = dg::EllSparseBlockMat<double>;

TEST_CASE( "Dx")
{
    unsigned n=7, N=33;
    auto bcx = GENERATE( dg::PER, dg::DIR, dg::NEU, dg::DIR_NEU, dg::NEU_DIR);
    auto dir = GENERATE( dg::centered, dg::forward, dg::backward);
    dg::Grid1d g1d( 0.1, 2*M_PI+0.1, n, N, bcx);
    if( bcx == dg::DIR)
        g1d = dg::Grid1d( 0, M_PI, n, N, bcx);
    else if( bcx == dg::NEU)
        g1d = dg::Grid1d( M_PI/2., 3*M_PI/2., n, N, bcx);
    else if( bcx == dg::DIR_NEU)
        g1d = dg::Grid1d( 0, M_PI/2., n, N, bcx);
    else if( bcx == dg::NEU_DIR)
        g1d = dg::Grid1d( M_PI/2., M_PI, n, N, bcx);

    const Vector func = dg::evaluate( function, g1d);
    const Vector w1d = dg::create::weights( g1d);
    const Vector deri = dg::evaluate( derivative, g1d);
    const Vector null = dg::evaluate( dg::zero, g1d);
    Matrix js = dg::create::jumpX( g1d, g1d.bcx());
    Vector error = func;
    dg::blas2::symv( js, func, error);
    dg::blas1::axpby( 1., null , -1., error);
    double dist = sqrt( dg::blas2::dot( w1d, error));
    INFO("Distance to true solution (jump     ): "<<dist);
    CHECK( dist < 1e-10);

    Matrix hs = dg::create::dx( g1d, g1d.bcx(), dir);

    dg::blas2::symv( hs, func, error);
    dg::blas1::axpby( 1., deri, -1., error);
    dist = sqrt( dg::blas2::dot( w1d, error));
    INFO( "Distance to true solution "<<dg::direction2str(dir )<<dist);
    CHECK( dist < 1e-10);
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5
}
