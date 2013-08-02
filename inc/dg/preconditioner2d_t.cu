#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>

#include "laplace.cuh"
#include "arrvec1d.cuh"
#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"
#include "blas.h"


const unsigned n = 2;
const unsigned Nx = 4; //minimum 3
const unsigned Ny = 4; //minimum 3

const double lx = 2.*M_PI;
const double ly = 1.;//M_PI;

using namespace dg;
using namespace std;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

double function( double x, double y) { return sin(x);}
double function( double x) { return sin(x);}

int main()
{
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells in x: "<<Nx<<"\n";
    cout<< "# of cells in y: "<<Ny<<"\n";
    Grid<double> g( 0, lx, 0, ly, n, Nx, Ny);
    Grid1d<double> g1d( 0, lx, n, Nx);
    T2D<double> t2d(g); 
    HVec hv2d = expand( function, g), hw2d( hv2d);
    HVec hx2d = evaluate( function, g);
    HVec hv1d = expand( function, g1d), hw1d( hv1d);
    cout << "Before multiplication: \n";
    double norm2 = blas2::dot( S1D<double>( g1d), hw1d);
    cout << "Norm2 1D is : "<<norm2<<endl;
    cout << "yields in 2D: "<< norm2*ly<<endl;
    double norm2_= blas2::dot( S2D<double>( g), hw2d);
    cout << "Norm2 2D is : "<<norm2_<<endl;
    double norm2X= blas2::dot( dg::create::w2d( g), hx2d);
    cout << "Norm2X2D is : "<<norm2X<<endl;

    cout << "Preconditioned\n";
    blas2::symv( S1D<double >(g1d), hw1d, hw1d);
    blas2::symv( S2D<double>(g), hw2d, hw2d);
    cout << "t2d " << t2d( 0) << " real: " << 8./M_PI<< endl;
    cout << "t2d " << t2d( 1) << " real: " << 24./M_PI<< endl;
    cout << "t2d " << t2d( 2) << " real: " << 3.*8./M_PI<< endl;
    cout << "t2d " << t2d( 3) << " real: " << 3.*24./M_PI<< endl;
    cout << "t2d " << t2d( 4) << " real: " << 8./M_PI<< endl;
    cout << "t2d " << t2d( 5) << " real: " << 24./M_PI<< endl;
    cout << "t2d " << t2d( 6) << " real: " << 3.*8./M_PI<< endl;
    cout << "t2d " << t2d( 7) << " real: " << 3.*24./M_PI<< endl;

    return 0;
}

