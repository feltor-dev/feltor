#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "weights.cuh"

#include "blas.h"

double function( double x)
{
    return exp(x);
}

double function( double x, double y)
{
        return exp(x)*exp(y);
}
double function( double x, double y, double z)
{
        return exp(x)*exp(y)*exp(z);
}

const double lx = 2;
const double ly = 2;
const double lz = 2;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

int main()
{
    //This file tests not only the evaluation functions but also the weights
    unsigned n;
    std::cout << "Type # of polynomial coefficients ( 1, 2,...,20)!\n";
    std::cin >> n;
    unsigned N, Nx, Ny, Nz;
    std::cout << "# of polynomial coefficients is: "<< n<<std::endl;
    std::cout << "Type # of grid cells (e.g. 10, 100)! ( Nx = N, Ny = 2N, Nz = 10*N)\n";
    std::cin >> N;
    std::cout << "# of grid cells is: "<< N<<std::endl;
    Nx = N; Ny = 2*N; Nz = 10*N;

    dg::Grid1d g1d( 0, lx, n, N);
    dg::Grid2d g2d( 0, lx,0, ly,n, Nx, Ny);
    dg::Grid3d g3d( 0, lx,0, ly,0, lz, n, Nx, Ny, Nz,dg::PER,dg::PER,dg::PER);

    //test evaluation functions
    const DVec h_x = dg::evaluate( exp, g1d);
    const DVec h_n = dg::evaluate( function, g2d);
    const DVec h_z = dg::evaluate( function, g3d);
    const DVec w1d = dg::create::weights( g1d);
    const DVec w2d = dg::create::weights( g2d);
    const DVec w3d = dg::create::weights( g3d);

    //test preconditioners
    std::cout << "Square normalized 1DXnorm "<<std::setprecision(16);
    double normX = dg::blas2::dot( h_x, w1d, h_x);
    std::cout << normX<<"\n";
    double solution = (exp(4.) -exp(0))/2.;
    std::cout << "Correct square norm is    "<<solution<<std::endl;
    std::cout << "Relative 1d error is      "<<(normX-solution)/solution<<"\n\n";

    double norm2X = dg::blas2::dot( w2d, h_n);
    std::cout << "Square normalized 2DXnorm "<< norm2X<<"\n";
    double solution2 = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    std::cout << "Correct square norm is    "<<solution2<<std::endl;
    std::cout << "Relative 2d error is      "<<(norm2X-solution2)/solution2<<"\n\n";

    double norm3X = dg::blas2::dot( h_z, w3d, h_z);
    std::cout << "Square normalized 3DXnorm "<< norm3X<<"\n";
    double solution3 = solution2*solution;
    std::cout << "Correct square norm is    "<<solution3<<std::endl;
    std::cout << "Relative 3d error is      "<<(norm3X-solution3)/solution3<<"\n";
    return 0;
} 
