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
double function( double x, double y, double z) { return sin(z);}
double derivative( double x, double y, double z) { return cos(z);}
dg::bc bcz = dg::PER;


int main()
{
    unsigned N;
    std::cout << "Note the supraconvergence!\n";
    std::cout << "Type in Nz!\n";
    std::cin >> N;
    std::cout << "# of cells          " << N <<"\n";
    dg::Grid3d<double> g( 0, lx, 0, lx, 0., lx, 1, N, N, N, dg::PER, dg::PER, bcz);
    cusp::ell_matrix< int, double, cusp::host_memory> dz = dg::create::dz<double>( g, bcz, dg::symmetric);
    cusp::ell_matrix< int, double, cusp::host_memory> lzM = dg::create::laplacianM_parallel<double>( g, bcz, dg::symmetric);
    const dg::HVec hv = dg::evaluate( function, g);
    dg::HVec hw = hv;
    const dg::HVec hu = dg::evaluate( derivative, g);

    dg::blas2::symv( dz, hv, hw);
    dg::blas1::axpby( 1., hu, -1., hw);
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(hw, dg::create::w3d(g), hw))<<"\n";
    dg::blas2::symv( lzM, hv, hw);
    dg::blas1::axpby( 1., hv, -1., hw);
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot(hw, dg::create::w3d(g), hw))<<"\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5
    return 0;
}
