#include <iostream>

#include <cusp/print.h>

#include "laplace.cuh"
#include "dgvec.cuh"

const unsigned P = 3;
const unsigned N = 3;

using namespace dg;
using namespace std;

typedef thrust::device_vector<double> DVector;
int main()
{
    ArrVec1d<P> hv( N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<P; i++)
            hv( k, i) = i;

    ArrVec1d<P> hw( N);
    DVector dv = hv.data();
    DVector dw = hw.data();
    Laplace<P> lap( N);
    cout << "The DG Laplacian: \n";
    cusp::print( lap.get_m());
    BLAS2< Laplace<P>, DVector>::dsymv( lap, dv, dw);
    cusp::array1d_view<DVector::iterator> dv_view( dv.begin(), dv.end());
    cusp::array1d_view<DVector::iterator> dw_view( dw.begin(), dw.end());
    cusp::print( dw_view);
    return 0;
}

