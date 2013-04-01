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

using namespace std;
using namespace dg;

const unsigned n = 3; //thrust transform is always faster
const unsigned Nx = 2e0;
const unsigned Ny = 2e0;

typedef thrust::device_vector<double>   DVec;
typedef thrust::host_vector<double>     HVec;

typedef ArrVec2d< double, n, HVec> HArrVec;
typedef ArrVec2d< double, n, DVec> DArrVec;

double function( double x, double y ) { return sin(x)*sin(y);}

int main()
{
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    HArrVec hv = evaluate<double(&)(double, double), n>( function, 0,2.*M_PI,0, 2.*M_PI, Nx, Ny );
    DArrVec  dv( hv);
    DArrVec  dv2( hv);
    cout << "Evaluated\n";
    cout <<hv<<endl;

    Operator<double, n> forward( DLT<n>::forward);
    dg::blas2::symv(1., thrust::make_tuple(forward, forward), dv.data(),0., dv.data());
    dg::blas2::symv(thrust::make_tuple(forward, forward), dv2.data(), dv2.data());

    HArrVec hv2 = expand<double(&)(double, double), n>( function, 0,2.*M_PI, 0,2.*M_PI, Nx, Ny);

    //test for equality...
    hv = dv;
    cout << "Multiplied\n";
    cout << hv <<endl;
    cout << "Multiplied 2nd version\n";
    hv = dv2; 
    cout << hv<<endl;
    cout << "Expanded: "<<endl;
    cout << hv2 <<endl;
    
    return 0;
}
