#include <iostream>

#include <cusp/print.h>
#include <cusp/elementwise.h>
#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/csr_matrix.h>

//#include "../gcc/timer.h"
#include "timer.cuh"
#include "laplace.cuh"
#include "tensor.cuh"
#include "operator_matrix.cuh"
#include "arrvec1d.cuh"
#include "arrvec2d.cuh"
#include "blas.h"

const unsigned n = 3;

using namespace dg;
using namespace std;
typedef thrust::device_vector< double>   DVec;
//typedef thrust::host_vector< double>     HVec;

//ell and hyb matrices are fastest for 1d transforms
//typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;
typedef cusp::csr_matrix<int, double, cusp::host_memory> HCMatrix;

int main()
{
    Timer t;
    cout << "Type Nx and Ny! \n";
    unsigned Nx, Ny;
    cin >> Nx; 
    cin >> Ny; //more N means less iterations for same error
    cout << "# of Legendre coefficients n is: "<< n <<endl;
    cout << "# of 2d cells is:                "<<Nx*Ny<<"\n";

    ArrVec2d<double, n> hv( Nx, Ny, 0.);
    DVec dv = hv.data(), dw( dv);
    t.tic();
    DMatrix laplace2d = dgtensor<double,n>(  create::laplace1d_per<double, n>(Ny, 2.),
                                    S1D<double, n>( 2.),
                                    S1D<double, n>( 2.),
                                    create::laplace1d_dir<double, n>(Nx, 2.) );
    t.toc();
    cout <<"\n";
    cout << "Laplace matrix creation took       "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( laplace2d, dv, dw);
    t.toc();
    cout << "Multiplication with laplace2d took "<<t.diff()<<"s\n";

    t.tic();
    HCMatrix ddyy_ = dgtensor<double, n>( 
                        create::laplace1d_per<double, n>(Ny, 2.),
                        tensor<double, n>( Nx, pipj));
    HCMatrix ddxx_ = dgtensor<double, n>( tensor<double, n>( Ny, pipj),
                                      create::laplace1d_dir<double, n>(Nx, 2.));
    HCMatrix laplace_;
    cusp::add( ddxx_, ddyy_, laplace_);
    DMatrix laplace( laplace_), ddxx( ddxx_), ddyy( ddyy_);
    
    t.toc();
    cout <<"\n";
    cout << "Laplace Product matrix creation took "<<t.diff()<<"s\n";
    //cout << "sorted "<<laplace_.is_sorted_by_row_and_column()<<"\n";
    t.tic();
    //blas2::symv( ddxx, dv, dw);
    //blas2::symv( ddyy, dw, dv);
    blas2::symv( laplace, dv, dw);
    t.toc();
    cout << "Multiplication with laplace2dp took "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( ddxx, dv, dw);
    t.toc();
    cout << "Multiplication with laplace_x took  "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( ddyy, dv, dw);
    t.toc();
    cout << "Multiplication with laplace_y took  "<<t.diff()<<"s\n";
    
    return 0;
}

