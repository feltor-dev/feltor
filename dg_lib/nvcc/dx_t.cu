#include <iostream>

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dx.cuh"
#include "evaluation.cuh"


using namespace std;
using namespace dg;

const unsigned n = 4;
const unsigned N = 10;
const double lx = 1.;

double function( double x) { return (x);}

int main ()
{
    cout << "Note the supraconvergence!\n";
    cout << "# of Legendre nodes " << n <<"\n";
    cout << "# of cells          " << N <<"\n";
    const double hx = lx/(double)N;
    cusp::ell_matrix< int, double, cusp::host_memory> hm = create::dx_per<n>( N, hx);
    ArrVec1d<double, n> hv = expand<double(&)(double), n>( function, 0., lx, N);
    ArrVec1d<double, n> hw = hv;

    blas2::symv( hm, hv.data(), hw.data());

    cout << "hv \n" << hv <<"\n";
    cout << "hw \n" << hw <<"\n";


    
    return 0;
}
