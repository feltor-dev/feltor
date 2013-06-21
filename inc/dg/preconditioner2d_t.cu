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
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrMat;
typedef dg::ArrVec2d< double, n, DVec>  DArrMat;

double function( double x, double y) { return sin(x);}
double function( double x) { return sin(x);}

int main()
{
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells in x: "<<Nx<<"\n";
    cout<< "# of cells in y: "<<Ny<<"\n";
    const double hx = lx/(double)Nx;
    const double hy = ly/(double)Ny;
    T2D<double, n> t2d(hx, hy); 
    HArrMat hv2d = expand< double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny), hw2d( hv2d);
    HArrMat hx2d = evaluate< double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny);
    HArrVec hv1d = expand< double(&)(double), n>( function, 0, lx, Nx), hw1d( hv1d);
    cout << "Before multiplication: \n";
    double norm2 = blas2::dot( S1D<double, n>( hx), hw1d.data());
    cout << "Norm2 1D is : "<<norm2<<endl;
    cout << "yields in 2D: "<< norm2*ly<<endl;
    double norm2_= blas2::dot( S2D<double, n>( hx, hy), hw2d.data());
    cout << "Norm2 2D is : "<<norm2_<<endl;
    double norm2X= blas2::dot( W2D<double, n>( hx, hy), hx2d.data());
    cout << "Norm2X2D is : "<<norm2X<<endl;
    cout << "Unpreconditioned\n";
    cout << hw1d<<endl;
    cout << hw2d<<endl;

    cout << "Preconditioned\n";
    blas2::symv( S1D<double,n >(hx), hw1d.data(), hw1d.data());
    cout << hw1d<<endl;
    blas2::symv( S2D<double,n >(hx, hy), hw2d.data(), hw2d.data());
    cout << hw2d<<endl;
    cout << "t2d " << t2d( 0) << " real: " << 8./M_PI<< endl;
    cout << "t2d " << t2d( 1) << " real: " << 24./M_PI<< endl;
    cout << "t2d " << t2d( 2) << " real: " << 3.*8./M_PI<< endl;
    cout << "t2d " << t2d( 3) << " real: " << 3.*24./M_PI<< endl;
    cout << "t2d " << t2d( 4) << " real: " << 8./M_PI<< endl;
    cout << "t2d " << t2d( 5) << " real: " << 24./M_PI<< endl;
    cout << "t2d " << t2d( 6) << " real: " << 3.*8./M_PI<< endl;
    cout << "t2d " << t2d( 7) << " real: " << 3.*24./M_PI<< endl;

    /*
    HMatrix laplace1d = create::laplace1d_per<n>(Nx, hx);
    HMatrix laplace2d = create::tensor<n>(create::laplace1d_per<n>( Ny, hy),
                                          create::laplace1d_per<n>( Nx, hx));
    //HMatrix laplace2d_= create::laplace2d_per<n>(Nx, Ny,hx, hy);
    blas2::symv( laplace1d, hv1d.data(), hw1d.data() );
    blas2::symv( laplace2d, hv2d.data(), hw2d.data() );

    //cout << "hw1d: \n"<<hw1d<<endl;
    //cout << "hw2d: \n"<<hw2d<<endl;
    //blas2::symv( laplace2d_, hv2d.data(), hw2d.data() );
    //cout << "hw2d_: \n"<<hw2d<<endl;

    cout << "After multiplication: \n";
    norm2 = blas2::dot( S1D<double, n>( hx), hw1d.data());
    cout << "Norm2 1D is : "<<norm2<<endl;
    cout << "yields in 2D: "<< norm2*ly<<endl;
    norm2_= blas2::dot( S2D<double, n>( hx, hy), hw2d.data());
    cout << "Norm2 2D is : "<<norm2_<<endl;
    */

    return 0;
}

