
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>

#include "blas.h"
#include "grid.h"
#include "dxx.cuh"
#include "tensor.cuh"
#include "dlt.h"
#include "evaluation.cuh"
#include "operator.h"
#include "operator_tensor.cuh"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3; //thrust is faster for 2, equal for 3 and slower for 4 
const unsigned Nx = 1e2;
const unsigned Ny = 1e2;

typedef thrust::device_vector<double>   DVec;
typedef thrust::host_vector<double>     HVec;

typedef cusp::ell_matrix<int, double, cusp::host_memory>   HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

double function( double x, double y ) { return sin(x)*sin(y);}

int main()
{
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    Timer t;
    Grid2d<double> g( 0, 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny);
    HVec hv = evaluate( function, g );
    HVec hv2( hv);
    DVec dv( hv);
    DVec dv2( hv2);
    Operator<double> forward( g.dlt().forward());
    Operator<double> forward2d = dg::tensor( forward, forward);
    //t.tic();
    //dg::blas2::symv(1., forward2d, dv.data(),0., dv.data());
    //t.toc();
    //cout << "Forward thrust transform took      "<<t.diff()<<"s\n";
    //t.tic();
    //dg::blas2::symv( forward2d, dv2.data(), dv2.data());
    //t.toc();
    //cout << "Forward thrust transform 2nd took  "<<t.diff()<<"s\n";

    HMatrix hforwardy = dgtensor<double>(n, tensor(Ny, forward), tensor(Nx, create::delta(n)));
    DMatrix dforwardy( hforwardy);
    HMatrix hforwardx = dgtensor<double>(n, tensor(Ny, create::delta(n)), tensor(Nx, forward));
    DMatrix dforwardx( hforwardx);
    t.tic();
    blas2::symv( dforwardy, dv2, dv2);
    t.toc();
    cout << "Foward - y cusp transform took     "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( dforwardx, dv2, dv2);
    t.toc();
    cout << "Foward - x cusp transform took     "<<t.diff()<<"s\n";
    //t.tic();
    //blas2::symv( dforwardy, dv2.data(), dv2.data());
    //blas2::symv( forward, dv2.data(), dv2.data());
    //t.toc();
    //cout << "Foward cusp-thrust transform took  "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( dforwardy, dv2, dv2);
    blas2::symv( dforwardx, dv2, dv2);
    t.toc();
    cout << "Foward cusp-cusp transform took    "<<t.diff()<<"s\n";

    HMatrix hforwardxy = dgtensor<double>(n, tensor(Ny, forward), tensor(Nx, forward));
    DMatrix dforwardxy( hforwardxy);
    t.tic();
    blas2::symv( dforwardxy, dv2, dv2);
    t.toc();
    cout << "Foward cusp transform took         "<<t.diff()<<"s\n";
    
    return 0;
}
