#include <iostream>
#include "dg/blas.h"
#include "derivativesX.h"
#include "evaluationX.h"
#include "derivativesT.h"

#include "catch2/catch_all.hpp"

static double zero( double, double) { return 0;}
static double zero( double, double, double) { return 0;}
static double sin( double x, double y ) {
    if( x < 0)
    {
        if( y < 0) return sin(x)*sin(y);
        else if( 0 <= y && y < 2*M_PI) return sin(x)*cos(y);
        else return sin(x)*sin(y - 2*M_PI);
    }
    return sin(x)*sin(y);
}
static double cosx( double x, double y) {
    if( x < 0)
    {
        if( y < 0) return cos(x)*sin(y);
        else if( 0 <= y && y < 2*M_PI) return cos(x)*cos(y);
        else return cos(x)*sin(y - 2*M_PI);
    } //has a discontinuity at x=0
    return cos(x)*sin(y);
}
static double cosy( double x, double y) {
    if( x < 0)
    {
        if( y < 0) return sin(x)*cos(y);
        else if( 0 <= y && y < 2*M_PI) return -sin(x)*sin(y);
        else return sin(x)*cos(y - 2*M_PI);
    }
    return sin(x)*cos(y);
}
static double sin(  double x, double y, double z) { return sin(x,y)*sin(z);}
static double cosx( double x, double y, double z) { return cosx(x,y)*sin(z);}
static double cosy( double x, double y, double z) { return cosy(x,y)*sin(z);}
static double cosz( double x, double y, double z) { return sin(x,y)*cos(z);}


using Vector = dg::DVec;
using Matrix = dg::Composite<dg::EllSparseBlockMat<double, thrust::device_vector> >;

TEST_CASE( "Derivatives X")
{
    unsigned n = 4, Nx = 33, Ny = 36, Nz = 90;
    dg::bc bcx=dg::DIR, bcy=dg::NEU, bcz = dg::PER;
    dg::GridX2d g2d( -2.*M_PI, M_PI, -M_PI/2., 2.*M_PI+M_PI/2., 1./3., 1./6.,
            n, Nx, Ny, bcx, bcy);
    SECTION( "Two dimensional")
    {
        const Vector w2d = dg::create::weights( g2d);

        Matrix dx2 = dg::create::dx( g2d, g2d.bcx(), dg::forward);
        Matrix dy2 = dg::create::dy( g2d, g2d.bcy(), dg::centered);
        //dy2.m1.trivial = false;
        //dy2.m2.trivial = false;
        //dy2.display( std::cout );
        Matrix jx2 = dg::create::jumpX( g2d, g2d.bcx());
        Matrix jy2 = dg::create::jumpY( g2d, g2d.bcy());
        Matrix m2[] = {dx2, dy2, jx2, jy2};
        const Vector f2d = dg::evaluate( sin, g2d);
        const Vector dx2d = dg::evaluate( cosx, g2d);
        const Vector dy2d = dg::evaluate( cosy, g2d);
        const Vector null2 = dg::evaluate( zero, g2d);
        Vector sol2[] = {dx2d, dy2d, null2, null2};

        //"WE EXPECT CONVERGENCE IN ALL QUANTITIES!!!";
        INFO( "TEST 2D: DX, DY, JX, JY");
        auto i = GENERATE( 0,1,2,3);
        Vector error = f2d;
        dg::blas2::symv( m2[i], f2d, error);
        dg::blas1::axpby( 1., sol2[i], -1., error);
        double dist = sqrt(dg::blas2::dot(error, w2d, error));
        INFO("Distance "<<i<<" to true solution: "<<dist);
        CHECK( dist <  1e-3);
    }
    SECTION( "Three dimensional")
    {
        dg::GridX3d g3d( -2.*M_PI, M_PI, 0., 2*M_PI, 0., 2.*M_PI, 1./3., 0., n,
                Nx, Ny, Nz, bcx, bcy, bcz);
        const Vector w3d = dg::create::weights( g3d);
        Matrix dx3 = dg::create::dx( g3d, g3d.bcx(), dg::forward);
        Matrix dy3 = dg::create::dy( g3d, g3d.bcy(), dg::backward);
        Matrix dz3 = dg::create::dz( g3d, g3d.bcz(), dg::centered);
        Matrix jx3 = dg::create::jumpX( g3d, g3d.bcx());
        Matrix jy3 = dg::create::jumpY( g3d, g3d.bcy());
        Matrix jz3 = dg::create::jumpZ( g3d, g3d.bcz());
        Matrix m3[] = {dx3, dy3, dz3, jx3, jy3, jz3};
        const Vector f3d = dg::evaluate( sin, g3d);
        const Vector dx3d = dg::evaluate( cosx, g3d);
        const Vector dy3d = dg::evaluate( cosy, g3d);
        const Vector dz3d = dg::evaluate( cosz, g3d);
        const Vector null3 = dg::evaluate( zero, g3d);
        Vector sol3[] = {dx3d, dy3d, dz3d, null3, null3, null3};

        auto i = GENERATE( 0,1,2,3,4,5);
        Vector error = f3d;
        dg::blas2::symv( m3[i], f3d, error);
        dg::blas1::axpby( 1., sol3[i], -1., error);
        double dist = sqrt(dg::blas2::dot(error, w3d, error));
        INFO("Distance "<<i<<" to true solution: "<<dist);
        if ( i==2)
            CHECK( dist < 1e-2); // DZ  is 2nd order
        else if( i==5)
            CHECK( dist < 0.5); // JZ  is awefully slow
        else
            CHECK( dist < 1e-3);
    }
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5

}
