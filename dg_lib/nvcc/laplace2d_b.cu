#include <iostream>

#include <cusp/print.h>
#include <cusp/elementwise.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/csr_matrix.h>

//#include "../gcc/timer.h"
#include "timer.cuh"
#include "laplace.cuh"
#include "laplace2d.cuh"
#include "operator_matrix.cuh"
#include "dgvec.cuh"
#include "dgmat.cuh"
#include "blas.h"

const unsigned P = 3;
const unsigned Nx = 1e2;
const unsigned Ny = 1e2;

using namespace dg;
using namespace std;
typedef thrust::device_vector< double>   DVec;
//typedef thrust::host_vector< double>     HVec;

//ell and hyb matrices are fastest for 1d transforms
//typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

int main()
{
    Timer t;
    cout << "# of Legendre coefficients P is: "<< P <<endl;
    cout << "# of 2d cells is:                "<<Nx*Ny<<"\n";

    ArrVec2d<double, P> hv( Nx, Ny, 0.);
    DVec dv = hv.data(), dw( dv);
    t.tic();
    DMatrix laplace2d = create::tensorSum<P>( create::laplace1d_per<P>(Ny, 2.),
                                              S1D<double, P>( 2.),
                                              S1D<double, P>( 2.),
                                              create::laplace1d_per<P>(Nx, 2.) );
    t.toc();
    cout <<"\n";
    cout << "Laplace matrix creation took       "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( laplace2d, dv, dw);
    t.toc();
    cout << "Multiplication with laplace2d took "<<t.diff()<<"s\n";

    t.tic();
    DMatrix ddxx = create::tensorProduct<double, P>( 
                            create::laplace1d_per<P>(Ny, 2.),
                            create::operatorMatrix( Nx, Operator<double, P>( create::detail::pipj)));
    DMatrix ddyy = create::tensorProduct<double, P>( 
                            create::operatorMatrix( Ny, Operator<double, P>( create::detail::pipj)),
                            create::laplace1d_per<P>(Nx, 2.));
    DMatrix laplace( ddxx);
    cusp::add( ddxx, ddyy, laplace);
    t.toc();
    cout <<"\n";
    cout << "Laplace Product matrix creation took "<<t.diff()<<"s\n";
    t.tic();
    //blas2::symv( ddxx, dv, dw);
    //blas2::symv( ddyy, dw, dv);
    blas2::symv( laplace, dv, dw);
    t.toc();
    cout << "Multiplication with laplace2dp took "<<t.diff()<<"s\n";
    
    return 0;
}
