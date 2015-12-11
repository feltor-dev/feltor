#include <iostream>
#include "blas.h"
#include "derivatives.h"
#include "evaluation.cuh"
#include "typedefs.cuh"
#include "sparseblockmat.cuh"

double zero( double x, double y) { return 0;}
double sin(  double x, double y) { return sin(x)*sin(y);}
double cosx( double x, double y) { return cos(x)*sin(y);}
double cosy( double x, double y) { return cos(y)*sin(x);}
double zero( double x, double y, double z) { return 0;}
double sin(  double x, double y, double z) { return sin(x)*sin(y)*sin(z);}
double cosx( double x, double y, double z) { return cos(x)*sin(y)*sin(z);}
double cosy( double x, double y, double z) { return cos(y)*sin(x)*sin(z);}
double cosz( double x, double y, double z) { return cos(z)*sin(x)*sin(y);}

typedef dg::DVec Vector;
typedef dg::EllSparseBlockMatDevice Matrix;

int main()
{
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type in n, Nx and Ny and Nz!\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::bc bcx=dg::DIR, bcz=dg::NEU_DIR, bcy=dg::PER;
    dg::Grid2d<double> g2d( 0, M_PI, 0.1, 2*M_PI+0.1, n, Nx, Ny, bcx, bcy);
    const Vector w2d = dg::create::weights( g2d);

    Matrix dx2 = dg::create::dx( g2d, dg::forward);
    Matrix dy2 = dg::create::dy( g2d, dg::centered);
    Matrix jx2 = dg::create::jumpX( g2d);
    Matrix jy2 = dg::create::jumpY( g2d);
    Matrix m2[] = {dx2, dy2, jx2, jy2};
    const Vector f2d = dg::evaluate( sin, g2d);
    const Vector dx2d = dg::evaluate( cosx, g2d);
    const Vector dy2d = dg::evaluate( cosy, g2d);
    const Vector null2 = dg::evaluate( zero, g2d);
    Vector sol2[] = {dx2d, dy2d, null2, null2};

    std::cout << "WE EXPECT CONVERGENCE IN ALL QUANTITIES!!!\n";
    std::cout << "TEST 2D: DX, DY, JX, JY\n";
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
    dg::CylindricalGrid g3d( 0,M_PI, 0.1, 2.*M_PI+0.1, M_PI/2.,M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz);
    const Vector w3d = dg::create::weights( g3d);
    Matrix dx3 = dg::create::dx( g3d.grid(), dg::forward);
    Matrix dy3 = dg::create::dy( g3d.grid(), dg::centered);
    Matrix dz3 = dg::create::dz( g3d.grid(), dg::backward);
    Matrix jx3 = dg::create::jumpX( g3d.grid());
    Matrix jy3 = dg::create::jumpY( g3d.grid());
    Matrix jz3 = dg::create::jumpZ( g3d.grid());
    Matrix m3[] = {dx3, dy3, dz3, jx3, jy3, jz3};
    const Vector f3d = dg::evaluate( sin, g3d.grid());
    const Vector dx3d = dg::evaluate( cosx, g3d.grid());
    const Vector dy3d = dg::evaluate( cosy, g3d.grid());
    const Vector dz3d = dg::evaluate( cosz, g3d.grid());
    const Vector null3 = dg::evaluate( zero, g3d.grid());
    Vector sol3[] = {dx3d, dy3d, dz3d, null3, null3, null3};

    std::cout << "TEST 3D: DX, DY, DZ, JX, JY, JZ\n";
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
