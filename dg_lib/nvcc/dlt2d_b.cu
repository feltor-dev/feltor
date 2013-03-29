
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>

#include "blas.h"
#include "laplace.cuh"
#include "laplace2d.cuh"
#include "timer.cuh"
#include "array.cuh"
#include "dlt.h"
#include "dgvec.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"
#include "operators.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3; //thrust transform is always faster
const unsigned Nx = 3e2;
const unsigned Ny = 3e2;

typedef thrust::device_vector<double>   DVec;
typedef thrust::host_vector<double>     HVec;

typedef ArrVec2d< double, n, HVec> HArrVec;
typedef ArrVec2d< double, n, DVec> DArrVec;
typedef cusp::ell_matrix<int, double, cusp::host_memory>   HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;


template< size_t n>
cusp::coo_matrix<int, double, cusp::host_memory> createForward( unsigned N)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A( n*N, n*N, n*n*N);
    Operator< double, n> forward(DLT<n>::forward);
    //std::cout << a << "\n"<<b <<std::endl;
    //assemble the matrix
    int number = 0;
    for( unsigned i=0; i<N; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned l=0; l<n; l++)
                create::detail::add_index<n>(A, number, i, i, k, l, forward(k,l));
    return A;
};

double function( double x, double y ) { return sin(x)*sin(y);}

int main()
{
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    Timer t;
    HArrVec hv = evaluate<double(&)(double, double), n>( function, 0, 2.*M_PI,0, 2.*M_PI, Nx, Ny );
    HArrVec hv2( hv);
    DArrVec  dv( hv);
    DArrVec  dv2( hv2);
    HMatrix hm = create::tensorSum( createForward<n>(Ny), S1D<double, n>(2.), S1D<double, n>(2.), createForward<n>(Nx));
    cout << "Transferring to device!\n";
    DMatrix dm(hm);
    Operator<double, n> forward( DLT<n>::forward);
    t.tic();
    dg::blas2::symv(1., thrust::make_tuple(forward, forward), dv.data(),0., dv.data());
    t.toc();
    cout << "Forward thrust transform took      "<<t.diff()<<"s\n";
    t.tic();
    dg::blas2::symv( thrust::make_tuple(forward, forward), dv.data(), dv.data());
    t.toc();
    cout << "Forward thrust transform 2nd took  "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( dm, dv2.data(), dv2.data());
    t.toc();
    cout << "Foward cusp transform took         "<<t.diff()<<"s\n";
    cout << "Note: Cusp transform is not the same !\n";
    
    return 0;
}
