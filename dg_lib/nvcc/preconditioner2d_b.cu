#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>

#include "timer.cuh"
#include "operator_matrix.cuh"
#include "tensor.cuh"
#include "functions.h"
#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"
#include "blas.h"


const unsigned n = 3; //dg is slightly faster than cusp
const unsigned Nx = 3e2; //minimum 3
const unsigned Ny = 3e2; //minimum 3

const double lx = 2.*M_PI;
const double ly = 1.;//M_PI;
using namespace dg;
using namespace std;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrMat;
typedef dg::ArrVec2d< double, n, DVec>  DArrMat;

typedef cusp::dia_matrix<int, double, cusp::host_memory>   HMatrix;
typedef cusp::dia_matrix<int, double, cusp::device_memory> DMatrix;

double function( double x, double y) { return sin(x);}
double function( double x) { return sin(x);}

int main()
{
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells : "<<Nx*Ny<<"\n";
    HArrMat hv2d = expand< double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny), hw2d( hv2d);
    DArrMat dv2d( hv2d), dw2d( hw2d);
    Timer t;

    t.tic();
    blas2::symv( S2D<double,n >(2., 2.), dw2d.data(), dw2d.data());
    t.toc();
    cout << "dg symv took   "<<t.diff()<<"s\n";
    Operator<double, n> s2d( pipj);
    HMatrix hm = tensor<double, n>( tensor(Ny, s2d), tensor(Nx, s2d));
    DMatrix dm( hm);

    t.tic();
    blas2::symv( dm , dv2d.data(), dv2d.data());
    t.toc();
    cout << "cusp symv took "<<t.diff()<<"s\n";

    return 0;
}

