#include <iostream>
#include "blas.h"
#include "derivatives.cuh"
#include "evaluation.cuh"
#include "typedefs.cuh"

const double lx = 2*M_PI;
/*
double function( double x, double y, double z) { return sin(3./4.*z);}
double derivative( double x, double y, double z) { return 3./4.*cos(3./4.*z);}
dg::bc bcz = dg::DIR_NEU;
*/
double function( double x, double y, double z) { return sin(x);}
double derivative( double x, double y, double z) { return cos(x);}
dg::bc bcx = dg::PER;


int main()
{
    unsigned n, Nx, Ny, Nz;
    std::cout << "Note the supraconvergence!\n";
    std::cout << "Type in n, Nx and Ny and Nz!\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d<double> g( 0, lx, 0, lx, 0., lx, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER);
    dg::DMatrix dx = dg::create::dx<double>( g, bcx, dg::normed, dg::symmetric);
    dg::DMatrix lzM = dg::create::laplacianM_perp<double>( g, bcx, dg::PER, dg::normed, dg::symmetric);
    const dg::DVec hv = dg::evaluate( function, g);
    dg::DVec hw = hv;
    const dg::DVec hu = dg::evaluate( derivative, g);

    const dg::DVec w3d = dg::create::weights( g);
    dg::blas2::symv( dx, hv, hw);
    dg::blas1::axpby( 1., hu, -1., hw);
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(hw, w3d, hw))<<"\n";
    dg::blas2::symv( lzM, hv, hw);
    dg::blas1::axpby( 1., hv, -1., hw);
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(hw, w3d, hw))<<" (Note the supraconvergence!)\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5
    return 0;
}
