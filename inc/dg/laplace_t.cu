#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>

#include "laplace.cuh"
#include "arrvec1d.cuh"
#include "blas.h"

#include "typedefs.cuh"

const unsigned n = 2;
const unsigned N = 5; //minimum 3

using namespace dg;
using namespace std;

typedef dg::ArrVec1d< double, n, dg::HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, dg::DVec>  DArrVec;

int main()
{
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells per dim: "<<N<<"\n";
    HArrVec hv( N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<n; i++)
            hv( k, i) = i;

    HArrVec hw( N);
    dg::DVec dv( hv.data()), dw( hw.data());
    DMatrix laplace1d = create::laplace1d_per<double, n>(N, 2);

    cout << "The DG Laplacian: \n";
    cusp::print( laplace1d);
    blas2::symv( laplace1d, dv, dw);
    cusp::array1d_view<DVec::iterator> dv_view( dv.begin(), dv.end());
    cusp::array1d_view<DVec::iterator> dw_view( dw.begin(), dw.end());
    cusp::print( dw_view);
    return 0;
}

