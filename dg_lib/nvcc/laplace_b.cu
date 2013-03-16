#include <iostream>

#include <cusp/print.h>

#include "timer.cuh"
#include "laplace.cuh"
#include "dgvec.cuh"

const unsigned P = 3;
const unsigned N = 1e5;

using namespace dg;
using namespace std;

typedef thrust::device_vector<double> DVector;
int main()
{
    Timer t;
    cout << "Order is (P-1): "<< P <<endl;
    cout << "# of intervals is: "<<N<<"\n";
    ArrVec1d<P> hv( N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<P; i++)
            hv( k, i) = i;

    ArrVec1d<P> hw( N);
    DVector dv = hv.data();
    DVector dw = hw.data();
    Laplace<P> lap( N);
    t.tic();
    BLAS2< Laplace<P>, DVector>::dsymv( lap, dv, dw);
    t.toc();
    cout << "Multiplication with laplace took "<<t.diff()<<"s\n";
    return 0;
}

