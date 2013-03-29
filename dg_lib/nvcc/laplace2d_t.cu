#include <iostream>

#include <cusp/print.h>
#include <cusp/ell_matrix.h>

#include "laplace.cuh"
#include "laplace2d.cuh"
#include "operator_matrix.cuh"
#include "dgvec.cuh"
#include "dgmat.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"
#include "blas.h"


const unsigned n = 2;
const unsigned Nx = 4; //minimum 3
const unsigned Ny = 4; //minimum 3

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

double function( double x, double y) { return sin(x);}
double function( double x) { return sin(x);}

int main()
{
    cout<< "# of polynomial coeff per dim: "<<n<<"\n";
    cout<< "# of cells in x: "<<Nx<<"\n";
    cout<< "# of cells in y: "<<Ny<<"\n";
    const double hx = lx/(double)Nx;
    const double hy = ly/(double)Ny;
    dg::S1D<double, n> s1dx( hx);
    dg::S1D<double, n> s1dy( hy);
    HArrMat hv2d = expand< double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny), hw2d( hv2d);
    HArrVec hv1d = expand< double(&)(double), n>( function, 0, lx, Nx), hw1d( hv1d);
    cout << "Before multiplication: \n";
    double norm2 = blas2::dot( S1D<double, n>( hx), hw1d.data());
    cout << "Norm2 1D is : "<<norm2<<endl;
    cout << "yields in 2D: "<< norm2*ly<<endl;
    double norm2_= blas2::dot( S2D<double, n>( hx, hy), hw2d.data());
    cout << "Norm2 2D is : "<<norm2_<<endl;

    HMatrix laplace1d = create::laplace1d_per<n>(Nx, hx);
    Operator<double, n> Sop( dg::create::detail::pipj);
    Operator<double, n> Id( dg::create::detail::delta);
    Sop *= hy/2.;
    HMatrix s1y = create::operatorMatrix( Sop, Ny);
    HMatrix s1x = create::operatorMatrix( Id, Nx);
    HMatrix ddxx = create::tensorProduct<double>( s1y, s1x);
    cout << endl;
    cout << endl;
    cusp::print( s1x); 
    cout << endl;
    cusp::print( s1y);
    cout << endl;
    cusp::print( ddxx);
    HMatrix laplace2d = create::tensorSum<n>(create::laplace1d_per<n>( Ny, hy), 
                                              s1dx, 
                                              s1dy,
                                              create::laplace1d_per<n>( Nx, hx));
    blas2::symv( laplace1d, hv1d.data(), hw1d.data() );
    blas2::symv( laplace2d, hv2d.data(), hw2d.data() );

    cout << "hw1d: \n"<<hw1d<<endl;
    cout << "hw2d: \n"<<hw2d<<endl;
    blas2::symv( ddxx, hv2d.data(), hw2d.data() );
    cout << "hw2d_: \n"<<hw2d<<endl;

    cout << "After multiplication: \n";
    norm2 = blas2::dot( S1D<double, n>( hx), hw1d.data());
    cout << "Norm2 1D is : "<<norm2<<endl;
    cout << "yields in 2D: "<< norm2*ly<<endl;
    norm2_= blas2::dot( S2D<double, n>( hx, hy), hw2d.data());
    cout << "Norm2 2D is : "<<norm2_<<endl;

    return 0;
}

