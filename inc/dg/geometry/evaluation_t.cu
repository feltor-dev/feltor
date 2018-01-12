#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dg/blas.h"

#include "evaluation.cuh"
#include "weights.cuh"


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

//typedef std::vector< double>   DVec;
typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

union udouble{
    double d;
    int64_t i;
};

int main()
{
    //This file tests not only the evaluation functions but also the weights
    unsigned n = 3, Nx = 10, Ny = 20, Nz = 100; 
    //std::cout << "Type # of polynomial coefficients ( 1, 2,...,20)!\n";
    //std::cin >> n;
    //std::cout << "# of polynomial coefficients is: "<< n<<std::endl;
    //std::cout << "Type # of grid cells (e.g. 10, 100)! ( Nx = N, Ny = 2N, Nz = 10*N)\n";
    //std::cin >> N;
    std::cout << "On Grid "<<n<<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<"\n";

    dg::Grid1d g1d( 0, lx, n, Nx);
    dg::Grid2d g2d( 0, lx,0, ly,n, Nx, Ny);
    dg::Grid3d g3d( 0, lx,0, ly,0, lz, n, Nx, Ny, Nz,dg::PER,dg::PER,dg::PER);

    //test evaluation functions
    const DVec func1d = dg::transfer<DVec>( dg::evaluate( exp, g1d));
    const DVec func2d = dg::transfer<DVec>( dg::evaluate( function, g2d));
    const DVec func3d = dg::transfer<DVec>( dg::evaluate( function, g3d));
    const DVec w1d = dg::transfer<DVec>( dg::create::weights( g1d));
    const DVec w2d = dg::transfer<DVec>( dg::create::weights( g2d));
    const DVec w3d = dg::transfer<DVec>( dg::create::weights( g3d));
    udouble res; 

    double integral = dg::blas1::dot( w1d, func1d); res.d = integral;
    std::cout << "1D integral               "<<std::setw(6)<<integral <<"\t" << res.i << "\n";
    double sol = (exp(2.) -exp(0));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol<<std::endl;
    std::cout << "Absolute 1d error is      "<<(integral-sol)<<"\n\n";

    double integral2d = dg::blas1::dot( w2d, func2d); res.d = integral2d;
    std::cout << "2D integral               "<<std::setw(6)<<integral2d <<"\t" << res.i << "\n";
    double sol2d = (exp(2.)-exp(0))*(exp(2.)-exp(0));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    std::cout << "Absolute 2d error is      "<<(integral2d-sol2d)<<"\n\n";

    double integral3d = dg::blas1::dot( w3d, func3d); res.d = integral3d;
    std::cout << "3D integral               "<<std::setw(6)<<integral3d <<"\t" << res.i << "\n";
    double sol3d = sol2d*(exp(2.)-exp(0));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol3d<<std::endl;
    std::cout << "Absolute 3d error is      "<<(integral3d-sol3d)<<"\n\n";

    double norm = dg::blas2::dot( func1d, w1d, func1d); res.d = norm;
    std::cout << "Square normalized 1D norm "<<std::setw(6)<<norm<<"\t" << res.i <<"\n";
    double solution = (exp(4.) -exp(0))/2.;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution<<std::endl;
    std::cout << "Relative 1d error is      "<<(norm-solution)/solution<<"\n\n";

    double norm2d = dg::blas2::dot( w2d, func2d); res.d = norm2d;
    std::cout << "Square normalized 2D norm "<<std::setw(6)<<norm2d<<"\t" << res.i <<"\n";
    double solution2d = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution2d<<std::endl;
    std::cout << "Relative 2d error is      "<<(norm2d-solution2d)/solution2d<<"\n\n";

    double norm3d = dg::blas2::dot( func3d, w3d, func3d); res.d = norm3d;
    std::cout << "Square normalized 3D norm "<<std::setw(6)<<norm3d<<"\t" << res.i <<"\n";
    double solution3d = solution2d*solution;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution3d<<std::endl;
    std::cout << "Relative 3d error is      "<<(norm3d-solution3d)/solution3d<<"\n";
    return 0;
} 
