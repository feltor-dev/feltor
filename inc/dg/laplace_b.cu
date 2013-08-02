#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/csr_matrix.h>

//#include "../gcc/timer.h"
#include "timer.cuh"
#include "laplace.cuh"
#include "arrvec1d.cuh"
#include "blas.h"

#include "typedefs.cuh"

const unsigned P = 3;
const unsigned N = 1e6;

using namespace dg;
using namespace std;

int main()
{
    Timer t;
    cout << "# of polynomial coefficients P is: "<< P <<endl;
    cout << "# of 1d intervals is:  "<<N<<"\n";
    ArrVec1d<HVec> hv( P, N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<P; i++)
            hv( k, i) = i;

    DVec dv = hv.data(), dw( dv);
    t.tic();
    DMatrix laplace1d = create::laplace1d_per<double>( P, N, 2.);
    t.toc();
    cout << "Laplace1d matrix creation took     "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( laplace1d, dv, dw);
    cudaThreadSynchronize();
    t.toc();
    cout << "Multiplication with laplace1d took "<<t.diff()<<"s\n";
    return 0;
}

