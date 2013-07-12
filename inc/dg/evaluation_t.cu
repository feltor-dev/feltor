#include <iostream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
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

const double lx = 2;
const double ly = 2;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

using namespace std;
int main()
{
    unsigned n;
    cout << "Type # of polynomial coefficients ( 1, 2,...,5)!\n";
    cin >> n;
    unsigned N, Nx, Ny;
    cout << "# of polynomial coefficients is: "<< n<<endl;
    cout << "Type # of grid cells (e.g. 10, 100)! ( Nx = N, Ny = 2N)\n";
    cin >> N;
    cout << "# of grid cells is: "<< N<<endl;
    Nx = N; Ny = 2*N;

    dg::Grid1d<double> g1d( 0, lx, n, N);
    dg::Grid<double> g2d( 0, lx,0, ly,n, Nx, Ny);

    //test evaluation and expand functions
    HVec h_v = dg::expand( function, g1d);
    HVec h_x = dg::evaluate( function, g1d);
    HVec h_m = dg::expand( function, g2d);
    HVec h_n = dg::evaluate( function, g2d);

    //test preconditioners
    dg::blas2::symv( 1., dg::S1D<double>(g1d), h_v, 0., h_v);

    double norm = dg::blas2::dot( h_v, dg::T1D<double>(g1d), h_v);
    double normX = dg::blas2::dot( h_x, dg::create::w1d(g1d), h_x);
    //double norm2 = dg::blas2::dot( dg::S2D<double>(g2d), h_m);
    double norm2 = dg::blas2::dot( dg::S2D<double>(g2d), h_m);
    double norm2X = dg::blas2::dot( dg::create::w2d(g2d), h_n);

    cout<< "Square normalized 1D norm "<< norm <<"\n";
    cout<< "Square normalized 1DXnorm "<< normX <<"\n";
    double solution = (exp(4.) -exp(0))/2.;
    cout << "Correct square norm of exp(x) is "<<solution<<endl;
    cout << "Square normalized 2D norm "<< norm2 <<"\n";
    cout << "Square normalized 2DXnorm "<< norm2X<<"\n";
    double solution2 = (exp(4.)-exp(0))/2.*(exp(4.) -exp(0))/2.;
    cout << "Correct square norm of exp(x)exp(y) is "<<solution2<<endl;
    return 0;
}

