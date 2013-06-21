#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>

#include "blas.h"
#include "laplace.cuh"
#include "array.cuh"
#include "dlt.h"
#include "arrvec1d.cuh"
#include "evaluation.cuh"
#include "operator.cuh"
#include "operator_matrix.cuh"
#include "tensor.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3; 
const unsigned Nx = 2e0;
const unsigned Ny = 2e0;

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
    HArrVec hv = evaluate<double(&)(double, double), n>( function, 0,2.*M_PI,0, 2.*M_PI, Nx, Ny );
    DArrVec  dv( hv);
    DArrVec  dv2( hv), dv3( hv);
    //cout << "Evaluated\n";
    //cout <<hv<<endl;

    Operator<double, n> forward( DLT<n>::forward);
    Operator<double, n*n> forward2d = tensor( forward, forward);
    dg::blas2::symv( 1., forward2d, dv.data(),0., dv.data());
    dg::blas2::symv( forward2d, dv2.data(), dv2.data());

    //HMatrix hm = tensor<double, n>( tensor(Ny, forward), tensor(Nx, forward));
    HMatrix hm = tensor( Nx*Ny, forward2d);
    DMatrix dm(hm);
    dg::blas2::symv( dm, dv3.data(), dv3.data());

    HArrVec hv2 = expand<double(&)(double, double), n>( function, 0,2.*M_PI, 0,2.*M_PI, Nx, Ny);

    //test for equality...
    hv = dv;
    cout << "Multiplied 1st version\n";
    cout << hv <<endl;
    cout << "Multiplied 2nd version\n";
    hv = dv2; 
    cout << hv<<endl;
    cout << "Multiplied 3rd version\n";
    hv = dv3; 
    cout << hv<<endl;
    cout << "Correct is: "<<endl;
    cout << hv2 <<endl;
    
    return 0;
}
