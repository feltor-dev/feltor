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

const unsigned N = 200;

const unsigned Nx = 5;
const unsigned Ny = 4;

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
    cout << "# of polynomial coefficients is: "<< n<<endl;
    dg::Grid1d<double> g1d( 0, lx, n, N);
    dg::Grid<double> g2d( 0, lx,0, ly,n, Nx, Ny);

    //test evaluation and expand functions
    cout << "ping0\n";
    HVec h_v = dg::expand( function, g1d);
    cout << "ping1\n";
    HVec h_x = dg::evaluate( function, g1d);
    cout << "ping2\n";
    HVec h_m = dg::expand( function, g2d);
    cout << "ping3\n";
    HVec h_n = dg::evaluate( function, g2d);
    cout << "ping4\n";
    DVec d_v( h_v);

    //test preconditioners
    dg::blas2::symv( 1., dg::S1D<double>(g1d), h_v, 0., h_v);
    cout << "ping5\n";

    double norm = dg::blas2::dot( h_v, dg::T1D<double>(g1d), h_v);
    cout << "ping6\n";
    double normX = dg::blas2::dot( h_x, dg::create::w1d(g1d), h_x);
    cout << "ping7\n";
    //double norm2 = dg::blas2::dot( dg::S2D<double>(g2d), h_m);
    double norm2 = dg::blas2::dot( dg::create::s2d(g2d), h_m);
    cout << "ping8\n";
    double norm2X = dg::blas2::dot( dg::create::w2d(g2d), h_n);
    cout << "ping9\n";

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

