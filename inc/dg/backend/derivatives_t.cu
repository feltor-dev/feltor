#include <iostream>
#include "blas.h"
#include "derivatives.h"
#include "evaluation.cuh"
#include "typedefs.cuh"
#include "sparseblockmat.cuh"

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


int main()
{
    unsigned n, Nx, Ny, Nz;
    std::cout << "Note the supraconvergence!\n";
    std::cout << "Type in n, Nx and Ny and Nz!\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d<double> g( 0, lx, 0, lx, 0., lx, n, Nx, Ny, Nz, bcx, bcy, bcz);
    //dg::Grid2d<double> g( 0, lx, 0, lx, n, Nx, Ny, bcx, dg::PER);
    dg::SparseBlockMatDevice dx = dg::create::dx( g, bcx, dg::centered);
    dg::DVec v = dg::evaluate( function, g);
    dg::DVec w = v;
    const dg::DVec u = dg::evaluate( derivative, g);

    const dg::DVec w3d = dg::create::weights( g);
    dg::blas2::symv( dx, v, w);
    dg::blas1::axpby( 1., u, -1., w);
    std::cout << "DX(symm):  Distance to true solution: "<<sqrt(dg::blas2::dot(w, w3d, w))<<"\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5

    std::cout << "TEST DY and DZ\n";
    {
    const dg::DVec func = dg::evaluate( siny, g);
    const dg::DVec deri = dg::evaluate( cosy, g);

    dg::SparseBlockMatDevice dy = dg::create::dy( g); 
    dg::DVec temp( func);
    dg::blas2::gemv( dy, func, temp);
    dg::blas1::axpby( 1., deri, -1., temp);
    std::cout << "DY(symm):  Distance to true solution: "<<sqrt(dg::blas2::dot(temp, w3d, temp))<<"\n";
    }
    {
    const dg::DVec func = dg::evaluate( sinz, g);
    const dg::DVec deri = dg::evaluate( cosz, g);

    dg::SparseBlockMatDevice dz = dg::create::dz( g); 
    dg::DVec temp( func);
    dg::blas2::gemv( dz, func, temp);
    dg::blas1::axpby( 1., deri, -1., temp);
    std::cout << "DZ(symm):  Distance to true solution: "<<sqrt(dg::blas2::dot(temp, w3d, temp))<<"\n";
    }
    return 0;
}
