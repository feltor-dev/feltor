#include <iostream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "operator.cuh"
#include "preconditioner.cuh"

#include "blas.h"

double function( double x)
{
    return exp(x);
}

double function( double x, double y)
{
        return exp(x)*exp(y);
}

const unsigned n = 3;
const unsigned N = 10;

const unsigned Nx = 32;
const unsigned Ny = 8;

const double lx = 2;
const double ly = 2;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrMat;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;


using namespace std;
int main()
{
    cout << "# of polynomial coefficients is: "<< n<<endl;
    double h = lx/(double)N;
    double hx = lx/(double)Nx; 
    double hy = ly/(double)Ny; 
    HArrVec h_v = dg::expand< double(&) (double), n>( function, 0, lx, N);
    HArrMat h_m = dg::expand< double(&) (double, double), n>( function, 0, lx, 0, ly, Nx, Ny);
    DArrVec d_v( h_v.data());
    dg::blas2::symv( 1., dg::S1D<double, n>(h), h_v.data(), 0., h_v.data());
    //dg::blas2::symv( dg::S2D<double, n>(hx,hy), h_m.data(), h_m.data());
    double norm = dg::blas2::dot( h_v.data(), dg::T1D<double, n>(h), h_v.data());
    double norm2 = dg::blas2::dot( dg::S2D<double, n>(hx,hy), h_m.data());
    cout<< "Square normalized 1D norm "<< norm <<"\n";
    double solution = (exp(4.) -exp(0))/2.;
    cout << "Correct square norm of exp(x) is "<<solution<<endl;
    cout<< "Square normalized 2D norm "<< norm2 <<"\n";
    double solution2 = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    cout << "Correct square norm of exp(x)exp(y) is "<<solution2<<endl;
    return 0;
}

