#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>

#include "laplace.cuh"
#include "laplace2d.cuh"
#include "dgvec.cuh"
#include "blas.h"

const unsigned n = 2;
const unsigned N = 5; //minimum 3

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
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells per dim: "<<N<<"\n";
    HArrVec hv( N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<n; i++)
            hv( k, i) = i;

    HArrVec hw( N);
    DVec dv( hv.data()), dw( hw.data());
    DMatrix laplace1d = create::laplace1d_per<n>(N, 2);
    HMatrix laplace1d_host (laplace1d);
    cout << "ping outside\n";
    HMatrix laplace2d = create::tensor<n>(laplace1d_host, laplace1d_host);
    cout << "ping outside\n";

    cout << "The DG Laplacian: \n";
    //cusp::print( laplace1d);
    cout << "The 2D DG Laplacian: \n";
    cusp::print( laplace2d);
    blas2::symv( laplace1d, dv, dw);
    cusp::array1d_view<DVec::iterator> dv_view( dv.begin(), dv.end());
    cusp::array1d_view<DVec::iterator> dw_view( dw.begin(), dw.end());
    cusp::print( dw_view);
    return 0;
}

