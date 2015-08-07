#include <iostream>
#include "blas.h"
#include "derivatives.h"
#include "evaluation.cuh"
#include "typedefs.cuh"
#include "sparseblockmat.cuh"
#include "timer.cuh"

const double lx = 2*M_PI;
/*
double function( double x, double y, double z) { return sin(3./4.*z);}
double derivative( double x, double y, double z) { return 3./4.*cos(3./4.*z);}
dg::bc bcz = dg::DIR_NEU;
*/
double function(   double x, double y, double z) { return sin(x);}
double derivative( double x, double y, double z) { return cos(x);}
double siny(   double x, double y, double z) { return sin(y);}
double cosy(   double x, double y, double z) { return cos(y);}
double sinz(   double x, double y, double z) { return sin(z);}
double cosz(   double x, double y, double z) { return cos(z);}
dg::bc bcx = dg::PER;
dg::bc bcy = dg::PER;
dg::bc bcz = dg::PER;

typedef dg::SparseBlockMatGPU Matrix;

int main()
{
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type in n, Nx and Ny and Nz!\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d<double> g( 0, lx, 0, lx, 0., lx, n, Nx, Ny, Nz, bcx, bcy, bcz);
    const dg::DVec w3d = dg::create::weights( g);
    dg::Timer t;
    std::cout << "TEST DX \n";
    {
    Matrix dx = dg::create::dx( g, bcx, dg::normed, dg::forward);
    dg::DVec v = dg::evaluate( function, g);
    dg::DVec w = v;
    const dg::DVec u = dg::evaluate( derivative, g);

    t.tic();
    dg::blas2::symv( dx, v, w);
    t.toc();
    std::cout << "Dx took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., u, -1., w);
    std::cout << "DX: Distance to true solution: "<<sqrt(dg::blas2::dot(w, w3d, w))<<"\n";
    }
    std::cout << "TEST DY \n";
    {
    const dg::DVec func = dg::evaluate( siny, g);
    const dg::DVec deri = dg::evaluate( cosy, g);

    Matrix dy = dg::create::dy( g); 
    dg::DVec temp( func);
    t.tic();
    dg::blas2::gemv( dy, func, temp);
    t.toc();
    std::cout << "Dy took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., deri, -1., temp);
    std::cout << "DY(1):           Distance to true solution: "<<sqrt(dg::blas2::dot(temp, w3d, temp))<<"\n";
    }
    std::cout << "TEST DZ \n";
    {
    const dg::DVec func = dg::evaluate( sinz, g);
    const dg::DVec deri = dg::evaluate( cosz, g);

    Matrix dz = dg::create::dz( g); 
    dg::DVec temp( func);
    t.tic();
    dg::blas2::gemv( dz, func, temp);
    t.toc();
    std::cout << "Dz took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., deri, -1., temp);
    std::cout << "DZ(1):           Distance to true solution: "<<sqrt(dg::blas2::dot(temp, w3d, temp))<<"\n";
    }
    return 0;
}
