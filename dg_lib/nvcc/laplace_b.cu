#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/csr_matrix.h>

#include "../gcc/timer.h"
#include "laplace.cuh"
#include "laplace2d.cuh"
#include "dgvec.cuh"
#include "dgmat.cuh"
#include "blas.h"

const unsigned P = 3;
const unsigned N = 3e4;
const unsigned Nx = 1e2;
const unsigned Ny = 1e2;

using namespace dg;
using namespace std;
typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

//ell and hyb matrices are fastest for 1d transforms
typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

int main()
{
    Timer t;
    cout << "# of polynomial coefficients P is: "<< P <<endl;
    cout << "# of 1d intervals is:  "<<N<<"\n";
    cout << "# of 2d cells is:      "<<Nx*Ny<<"\n";
    ArrVec1d<double, P> hv( N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<P; i++)
            hv( k, i) = i;

    ArrVec2d<double, P> hw2d( Nx, Ny, 0.);
    DVec dv = hv.data(), dw( dv);
    DVec dw2d = hw2d.data(), dv2d( dw2d);
    DMatrix laplace1d = create::laplace1d_per<P>( N, 2.);
    DMatrix laplace2d = create::tensorSum<P>( create::laplace1d_per<P>(Ny, 2.),
                                              S1D<double, P>( 2.),
                                              S1D<double, P>( 2.),
                                              create::laplace1d_per<P>(Nx, 2.) );
    t.tic();
    blas2::symv( laplace1d, dv, dw);
    cudaThreadSynchronize();
    t.toc();
    cout << "Multiplication with laplace1d took "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( laplace2d, dv2d, dw2d);
    cudaThreadSynchronize();
    t.toc();
    cout << "Multiplication with laplace2d took "<<t.diff()<<"s\n";
    return 0;
}

