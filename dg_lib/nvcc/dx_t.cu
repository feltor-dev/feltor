#include <iostream>

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dx.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"


using namespace std;
using namespace dg;

const unsigned n = 5;
const unsigned N = 40;
const double lx = 2*M_PI;

double function( double x) { return sin(x);}
double derivative( double x) { return cos(x);}

int main ()
{
    cout << "Note the supraconvergence!\n";
    cout << "# of Legendre nodes " << n <<"\n";
    cout << "# of cells          " << N <<"\n";
    const double hx = lx/(double)N;
    cusp::ell_matrix< int, double, cusp::host_memory> hm = create::dx_per<double, n>( N, hx);
    ArrVec1d<double, n> hv = expand<double(&)(double), n>( function, 0., lx, N);
    ArrVec1d<double, n> hw = hv;
    const ArrVec1d<double, n> hu = expand<double(&)(double), n>( derivative, 0., lx, N);

    blas2::symv( hm, hv.data(), hw.data());
    blas1::axpby( 1., hu.data(), -1., hw.data());
    
    cout << "Distance to true solution: "<<sqrt(blas2::dot( S1D<double, n>(hx), hw.data()) )<<"\n";
    //n = 1 -> p = 2
    //n = 2 -> p = 1
    //n = 3 -> p = 3
    //n = 4 -> p = 3
    //n = 5 -> p = 5

    //cout << "hv \n" << hv <<"\n";
    //cout << "hw \n" << hw <<"\n";


    
    return 0;
}
