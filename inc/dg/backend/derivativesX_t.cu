#include <iostream>
#include "blas.h"
#include "derivativesX.h"
#include "evaluationX.cuh"
#include "typedefs.cuh"
#include "sparseblockmat.cuh"

double zero( double x, double y) { return 0;}
double zero( double x, double y, double z) { return 0;}
double sin( double x, double y ) { 
    if( x < 0)  
    {
        if( y < 0) return sin(x)*sin(y);
        else if( 0 <= y && y < 2*M_PI) return sin(x)*cos(y);
        else return sin(x)*sin(y - 2*M_PI);
    }
    return sin(x)*sin(y);
}
double cosx( double x, double y) { 
    if( x < 0)
    {
        if( y < 0) return cos(x)*sin(y);
        else if( 0 <= y && y < 2*M_PI) return cos(x)*cos(y);
        else return cos(x)*sin(y - 2*M_PI);
    } //has a discontinuity at x=0
    return cos(x)*sin(y);
}
double cosy( double x, double y) { 
    if( x < 0)
    {
        if( y < 0) return sin(x)*cos(y);
        else if( 0 <= y && y < 2*M_PI) return -sin(x)*sin(y);
        else return sin(x)*cos(y - 2*M_PI);
    }
    return sin(x)*cos(y);
}
double sin(  double x, double y, double z) { return sin(x,y)*sin(z);}
double cosx( double x, double y, double z) { return cosx(x,y)*sin(z);}
double cosy( double x, double y, double z) { return cosy(x,y)*sin(z);}
double cosz( double x, double y, double z) { return sin(x,y)*cos(z);}


typedef dg::DVec Vector;
typedef dg::Composite<dg::EllSparseBlockMatDevice<double> > Matrix;
//typedef dg::HVec Vector;
//typedef dg::Composite<dg::EllSparseBlockMat<double> > Matrix;

int main()
{
    unsigned n, Nx, Ny, Nz;
    //std::cout << "Type in n, Nx (1./5.) and Ny (1./4.) and Nz!\n";
    //std::cin >> n >> Nx >> Ny >> Nz;
    //dg::bc bcx=dg::DIR_NEU, bcy=dg::DIR, bcz = dg::PER;
    //dg::GridX2d g2d( -2.*M_PI, M_PI/2., -M_PI, 2*M_PI+M_PI, 1./5., 1./4., n, Nx, Ny, bcx, bcy);
    std::cout << "Type in n, Nx (1./3.) and Ny (1./6.) and Nz!\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::bc bcx=dg::DIR, bcy=dg::NEU, bcz = dg::PER;
    dg::GridX2d g2d( -2.*M_PI, M_PI, -M_PI/2., 2.*M_PI+M_PI/2., 1./3., 1./6., n, Nx, Ny, bcx, bcy);
    g2d.display(std::cout);
    //dg::GridX2d g2d( -2.*M_PI, M_PI/2., 0., 2*M_PI, 1./5., 0., n, Nx, Ny, bcx, bcy);
    const Vector w2d = dg::create::weights( g2d);

    Matrix dx2 = dg::create::dx( g2d, dg::forward);
    Matrix dy2 = dg::create::dy( g2d, dg::centered);
    //dy2.m1.trivial = false;
    //dy2.m2.trivial = false;
    //dy2.display( std::cout );
    Matrix jx2 = dg::create::jumpX( g2d);
    Matrix jy2 = dg::create::jumpY( g2d);
    Matrix m2[] = {dx2, dy2, jx2, jy2};
    const Vector f2d = dg::evaluate( sin, g2d);
    const Vector dx2d = dg::evaluate( cosx, g2d);
    const Vector dy2d = dg::evaluate( cosy, g2d);
    const Vector null2 = dg::evaluate( zero, g2d);
    Vector sol2[] = {dx2d, dy2d, null2, null2};

    std::cout << "WE EXPECT CONVERGENCE IN ALL QUANTITIES!!!\n";
    std::cout << "TEST 2D: DX, DY, JX, JY, JXY\n";
    for( unsigned i=0; i<4; i++)
    {
        Vector error = f2d;
        dg::blas2::symv( m2[i], f2d, error);
        dg::blas1::axpby( 1., sol2[i], -1., error);
        std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(error, w2d, error))<<"\n";
    }
    Vector tempX = f2d, tempY(tempX);
    dg::blas2::symv( m2[2], f2d, tempX);
    dg::blas2::symv( m2[3], f2d, tempY);
    dg::blas1::axpby( 1., tempX, 1., tempY, tempY);
    dg::blas1::axpby( 1., null2, -1., tempY);
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(tempY, w2d, tempY))<<"\n";
    //dg::GridX3d g3d( -2.*M_PI, M_PI/2., -M_PI, 2*M_PI+M_PI, 0., 2.*M_PI, 1./5., 1./4., n, Nx, Ny, Nz, bcx, bcy, bcz);
    dg::GridX3d g3d( -2.*M_PI, M_PI, 0., 2*M_PI, 0., 2.*M_PI, 1./3., 0., n, Nx, Ny, Nz, bcx, bcy, bcz);
    const Vector w3d = dg::create::weights( g3d);
    Matrix dx3 = dg::create::dx( g3d, dg::forward);
    Matrix dy3 = dg::create::dy( g3d, dg::backward);
    Matrix dz3 = dg::create::dz( g3d, dg::centered);
    Matrix jx3 = dg::create::jumpX( g3d);
    Matrix jy3 = dg::create::jumpY( g3d);
    Matrix jz3 = dg::create::jumpZ( g3d);
    Matrix m3[] = {dx3, dy3, dz3, jx3, jy3, jz3};
    const Vector f3d = dg::evaluate( sin, g3d);
    const Vector dx3d = dg::evaluate( cosx, g3d);
    const Vector dy3d = dg::evaluate( cosy, g3d);
    const Vector dz3d = dg::evaluate( cosz, g3d);
    const Vector null3 = dg::evaluate( zero, g3d);
    Vector sol3[] = {dx3d, dy3d, dz3d, null3, null3, null3};

    std::cout << "TEST 3D: DX, DY, DZ, JX, JY, JZ, JXY\n";
    for( unsigned i=0; i<6; i++)
    {
        Vector error = f3d;
        dg::blas2::symv( m3[i], f3d, error);
        dg::blas1::axpby( 1., sol3[i], -1., error);
        std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(error, w3d, error))<<"\n";
    }
    Vector tX = f3d, tY(tX);
    dg::blas2::symv( m3[3], f3d, tX);
    dg::blas2::symv( m3[4], f3d, tY);
    dg::blas1::axpby( 1., tX, 1., tY, tY);
    dg::blas1::axpby( 1., null3, -1., tY);
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(tY, w3d, tY))<<"\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5

    return 0;
}
