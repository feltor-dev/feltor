#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>

#include "laplace.cuh"
#include "laplace2d.cuh"
#include "dgvec.cuh"
#include "blas.h"

const unsigned n = 2;
const unsigned N = 3; //minimum 3

using namespace dg;
using namespace std;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

int main()
{
    HArrVec hv( N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<n; i++)
            hv( k, i) = i;

    HArrVec hw( N);
    DVec dv( hv.data()), dw( hw.data());
    DMatrix laplace1d = create::laplace1d_per<n>(N, 2);
    DMatrix laplace2d = create::laplace2d_per<n>(N,N, 2., 2.);

    cout << "The DG Laplacian: \n";
    cusp::print( laplace1d);
    cout << "The 2D DG Laplacian: \n";
    cusp::print( laplace2d);
    blas2::symv( laplace1d, dv, dw);
    cusp::array1d_view<DVec::iterator> dv_view( dv.begin(), dv.end());
    cusp::array1d_view<DVec::iterator> dw_view( dw.begin(), dw.end());
    cusp::print( dw_view);
    return 0;
}

