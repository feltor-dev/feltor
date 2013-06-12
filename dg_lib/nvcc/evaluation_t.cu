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

const unsigned n = 2;
const unsigned N = 10;

const unsigned Nx = 5;
const unsigned Ny = 4;

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
    dg::Grid1d<double,n> g1d( 0, lx, N);
    dg::Grid<double,n> g2d( 0, lx,0, ly, Nx, Ny);

    //test evaluation and expand functions
    HVec h_v = dg::expand( function, g1d);
    HVec h_x = dg::evaluate( function, g1d);
    HVec h_m = dg::expand( function, g2d);
    DVec d_v( h_v);

    //test preconditioners
    dg::blas2::symv( 1., dg::S1D<double, n>(g1d.h()), h_v, 0., h_v);

    double norm = dg::blas2::dot( h_v, dg::T1D<double, n>(g1d.h()), h_v);
    double normX = dg::blas2::dot( h_x, dg::W1D<double, n>(g1d.h()), h_x);
    double norm2 = dg::blas2::dot( dg::S2D<double, n>(g2d.hx(),g2d.hy()), h_m);

    cout<< "Square normalized 1D norm "<< norm <<"\n";
    cout<< "Square normalized 1DXnorm "<< normX <<"\n";
    double solution = (exp(4.) -exp(0))/2.;
    cout << "Correct square norm of exp(x) is "<<solution<<endl;
    cout << "Square normalized 2D norm "<< norm2 <<"\n";
    double solution2 = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    cout << "Correct square norm of exp(x)exp(y) is "<<solution2<<endl;
    return 0;
}

