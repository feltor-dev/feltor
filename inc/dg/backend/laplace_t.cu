#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>

#include "laplace.cuh"
#include "blas.h"

#include "typedefs.cuh"

const unsigned n = 1;
const unsigned N = 4; //minimum 3

using namespace dg;
using namespace std;


int main()
{
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells per dim: "<<N<<"\n";
    //HArrVec hv( n, N,  1);
    //for( unsigned k=0; k<N; k++)
    //    for( unsigned i=0; i<n; i++)
    //        hv( k, i) = i;

    //dg::HVec hw( n, N);
    //dg::DVec dv( hv.data()), dw( hw.data());
    double h = 1./N;
    DMatrix laplace1d = create::laplace1d_per<double>(n, N, h);
    dg::Grid1d<double> g( 0, 1., n, N, DIR);
    DMatrix laplace1dp = create::laplace1d<double>(g, dg::normed, dg::symmetric);

    cout << "The DG Laplacian: \n";
    //cusp::print( laplace1d);
    cusp::print( laplace1dp);
    //blas2::symv( laplace1d, dv, dw);
    //cusp::array1d_view<DVec::iterator> dv_view( dv.begin(), dv.end());
    //cusp::array1d_view<DVec::iterator> dw_view( dw.begin(), dw.end());
    //cusp::print( dw_view);
    return 0;
}

