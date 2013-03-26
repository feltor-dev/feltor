#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>

#include "laplace.cuh"
#include "laplace2d.cuh"
#include "dgvec.cuh"
#include "dgmat.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"
#include "blas.h"


const unsigned n = 3;
const unsigned Nx = 10; //minimum 3
const unsigned Ny = 10; //minimum 3

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

using namespace dg;
using namespace std;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrMat;
typedef dg::ArrVec2d< double, n, DVec>  DArrMat;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

double function( double x, double y) { return sin(y);}
double function( double x) { return sin(x);}

int main()
{
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells in x: "<<Nx<<"\n";
    cout<< "# of cells in y: "<<Ny<<"\n";
    const double hx = lx/(double)Nx;
    const double hy = ly/(double)Ny;
    HArrMat hv2d = expand< double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny), hw2d( hv2d);
    HArrVec hv1d = expand< double(&)(double), n>( function, 0, lx, Nx), hw1d( hv1d);
    cout << "Before multiplication: \n";
    double norm2 = blas2::dot( S1D<double, n>( hx), hw1d.data());
    cout << "Norm2 1D is : "<<norm2<<endl;
    cout << "yields in 2D: "<< norm2*ly<<endl;
    double norm2_= blas2::dot( S2D<double, n>( hx, hy), hw2d.data());
    cout << "Norm2 2D is : "<<norm2_<<endl;

    HMatrix laplace1d = create::laplace1d_per<n>(Nx, hx);
    HMatrix laplace2d = create::tensor<n>(create::laplace1d_per<n>( Ny, hy),
                                          create::laplace1d_per<n>( Nx, hx));
    HMatrix laplace2d_= create::laplace2d_per<n>(Nx, Ny,hx, hy);
    blas2::symv( laplace1d, hv1d.data(), hw1d.data() );
    blas2::symv( laplace2d, hv2d.data(), hw2d.data() );

    //cout << "hw1d: \n"<<hw1d<<endl;
    //cout << "hw2d: \n"<<hw2d<<endl;
    blas2::symv( laplace2d_, hv2d.data(), hw2d.data() );
    //cout << "hw2d_: \n"<<hw2d<<endl;

    cout << "After multiplication: \n";
    norm2 = blas2::dot( S1D<double, n>( hx), hw1d.data());
    cout << "Norm2 1D is : "<<norm2<<endl;
    cout << "yields in 2D: "<< norm2*ly<<endl;
    norm2_= blas2::dot( S2D<double, n>( hx, hy), hw2d.data());
    cout << "Norm2 2D is : "<<norm2_<<endl;

    return 0;
}

