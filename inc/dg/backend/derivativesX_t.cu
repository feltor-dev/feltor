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


typedef dg::DVec Vector;
typedef dg::Composite<dg::EllSparseBlockMatDevice<double> > Matrix;
//typedef dg::HVec Vector;
//typedef dg::Composite<dg::EllSparseBlockMat<double> > Matrix;

int main()
{
    unsigned n, Nx, Ny;
    //std::cout << "Type in n, Nx (1./5.) and Ny (1./4.)!\n";
    //std::cin >> n >> Nx >> Ny;
    //dg::bc bcx=dg::DIR_NEU, bcy=dg::DIR, bcz = dg::PER;
    //dg::GridX2d g2d( -2.*M_PI, M_PI/2., -M_PI, 2*M_PI+M_PI, 1./5., 1./4., n, Nx, Ny, bcx, bcy);
    std::cout << "Type in n, Nx (1./3.) and Ny (1./6.)!\n";
    std::cin >> n >> Nx >> Ny;
    dg::bc bcx=dg::DIR, bcy=dg::NEU;
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
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5

    return 0;
}
