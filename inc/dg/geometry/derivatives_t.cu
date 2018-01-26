#include <iostream>
#include <cmath>
#include "dg/blas.h"
#include "derivatives.h"
#include "evaluation.cuh"

double zero( double x, double y) { return 0;}
double sine( double x, double y) { return sin(x)*sin(y);}
double cosx( double x, double y) { return cos(x)*sin(y);}
double cosy( double x, double y) { return cos(y)*sin(x);}
double zero( double x, double y, double z) { return 0;}
double sine( double x, double y, double z) { return sin(x)*sin(y)*sin(z);}
double cosx( double x, double y, double z) { return cos(x)*sin(y)*sin(z);}
double cosy( double x, double y, double z) { return cos(y)*sin(x)*sin(z);}
double cosz( double x, double y, double z) { return cos(z)*sin(x)*sin(y);}

typedef dg::DMatrix Matrix;
typedef dg::DVec Vector;

int main()
{
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type in n, Nx and Ny and Nz!\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::bc bcx=dg::DIR, bcy=dg::PER, bcz=dg::NEU_DIR;
    dg::Grid2d g2d( 0, M_PI, 0.1, 2*M_PI+0.1, n, Nx, Ny, bcx, bcy);
    const Vector w2d = dg::create::weights( g2d);

    Matrix dx2 = dg::create::dx( g2d, dg::forward);
    Matrix dy2 = dg::create::dy( g2d, dg::centered);
    Matrix jx2 = dg::create::jumpX( g2d);
    Matrix jy2 = dg::create::jumpY( g2d);
    Matrix m2[] = {dx2, dy2, jx2, jy2};
    const Vector f2d = dg::evaluate( sine, g2d);
    const Vector dx2d = dg::evaluate( cosx, g2d);
    const Vector dy2d = dg::evaluate( cosy, g2d);
    const Vector null2 = dg::evaluate( zero, g2d);
    Vector sol2[] = {dx2d, dy2d, null2, null2};

    exblas::udouble res;
    std::cout << "WE EXPECT CONVERGENCE IN ALL QUANTITIES!!!\n";
    std::cout << "TEST 2D: DX, DY, JX, JY\n";
    for( unsigned i=0; i<4; i++)
    {
        Vector error = sol2[i];
        dg::blas2::symv( -1., m2[i], f2d, 1., error);
        dg::blas1::pointwiseDot( error, error, error);
        double norm = sqrt(dg::blas1::dot( w2d, error)); res.d = norm;
        std::cout << "Distance to true solution: "<<norm<<"\t"<<res.i<<"\n";
    }
    dg::Grid3d g3d( 0,M_PI, 0.1, 2.*M_PI+0.1, M_PI/2.,M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz);
    const Vector w3d = dg::create::weights( g3d);
    Matrix dx3 = dg::create::dx( g3d, dg::forward);
    Matrix dy3 = dg::create::dy( g3d, dg::centered);
    Matrix dz3 = dg::create::dz( g3d, dg::backward);
    Matrix jx3 = dg::create::jumpX( g3d);
    Matrix jy3 = dg::create::jumpY( g3d);
    Matrix jz3 = dg::create::jumpZ( g3d);
    Matrix m3[] = {dx3, dy3, dz3, jx3, jy3, jz3};
    const Vector f3d = dg::evaluate( sine, g3d);
    const Vector dx3d = dg::evaluate( cosx, g3d);
    const Vector dy3d = dg::evaluate( cosy, g3d);
    const Vector dz3d = dg::evaluate( cosz, g3d);
    const Vector null3 = dg::evaluate( zero, g3d);
    Vector sol3[] = {dx3d, dy3d, dz3d, null3, null3, null3};

    std::cout << "TEST 3D: DX, DY, DZ, JX, JY, JZ\n";
    for( unsigned i=0; i<6; i++)
    {
        Vector error = sol3[i];
        dg::blas2::symv( -1., m3[i], f3d, 1., error);
        double norm = sqrt(dg::blas2::dot( error, w3d, error)); res.d = norm;
        std::cout << "Distance to true solution: "<<norm<<"\t"<<res.i<<"\n";
    }
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5

    return 0;
}
