
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>

#include "blas.h"
#include "laplace.cuh"
#include "tensor.cuh"
#include "timer.cuh"
#include "array.cuh"
#include "dlt.h"
#include "arrvec1d.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"
#include "operator.cuh"
#include "operator_matrix.cuh"
#include "tensor.cuh"


using namespace std;
using namespace dg;

const unsigned n = 4; //thrust is faster for 2, equal for 3 and slower for 4 
const unsigned Nx = 3e2;
const unsigned Ny = 3e2;

typedef thrust::device_vector<double>   DVec;
typedef thrust::host_vector<double>     HVec;

typedef ArrVec2d< double, n, HVec> HArrVec;
typedef ArrVec2d< double, n, DVec> DArrVec;
typedef cusp::ell_matrix<int, double, cusp::host_memory>   HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

double function( double x, double y ) { return sin(x)*sin(y);}

int main()
{
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    Timer t;
    HArrVec hv = evaluate<double(&)(double, double), n>( function, 0, 2.*M_PI,0, 2.*M_PI, Nx, Ny );
    HArrVec hv2( hv);
    DArrVec  dv( hv);
    DArrVec  dv2( hv2);
    Operator<double, n> forward( DLT<n>::forward);
    HMatrix hmx = tensor<double, n>( tensor(Ny, forward), tensor<double, n>(Nx, delta));

    cout << "Transferring to device!\n";
    DMatrix dm( hmx);
    Operator<double,n*n > forward2d = dg::tensor( forward, forward);
    t.tic();
    dg::blas2::symv(1., forward2d, dv.data(),0., dv.data());
    t.toc();
    cout << "Forward thrust transform took      "<<t.diff()<<"s\n";
    t.tic();
    dg::blas2::symv( forward2d, dv2.data(), dv2.data());
    t.toc();
    cout << "Forward thrust transform 2nd took  "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( dm, dv2.data(), dv2.data());
    blas2::symv( forward, dv2.data(), dv2.data());
    t.toc();
    cout << "Foward cusp-thrust transform took  "<<t.diff()<<"s\n";
    
    return 0;
}
