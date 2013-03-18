#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/csr_matrix.h>

#include "timer.cuh"
#include "laplace.cuh"
#include "dgvec.cuh"

const unsigned P = 3;
const unsigned N = 1e6;

using namespace dg;
using namespace std;
double sine(double x){ return /*x*x*x*/sin(2*M_PI*x);}
double secondsine(double x){ return /*-6*x*/4.*M_PI*M_PI*sin(2*M_PI*x);}

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, P, HVec>   HArrVec;
typedef dg::ArrVec1d< double, P, DVec>   DArrVec;

//ell and hyb matrices are fastest for 1d transforms
typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

int main()
{
    Timer t;
    cout << "Order is P: "<< P <<endl;
    cout << "# of intervals is: "<<N<<"\n";
    ArrVec1d<double, P> hv( N,  1);
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<P; i++)
            hv( k, i) = i;

    ArrVec1d<double, P> hw( N);
    DVec dv = hv.data();
    DVec dw = hw.data();
    DMatrix laplace1d = create::laplace1d_per<P>( N, 2.);
    t.tic();
    BLAS2< DMatrix, DVec>::dsymv( laplace1d, dv, dw);
    t.toc();
    cout << "Multiplication with laplace took "<<t.diff()<<"s\n";
    return 0;
}

