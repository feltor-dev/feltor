#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>


#include "preconditioner.cuh"
#include "blas.h"
#include "laplace.cuh"
#include "timer.cuh"
#include "array.cuh"
#include "dlt.h"
#include "arrvec1d.cuh"
#include "evaluation.cuh"
#include "operator.cuh"


using namespace std;
using namespace dg;

const unsigned n = 5; // in 1d DG transform is fastest
const unsigned N = 1e6;

typedef thrust::device_vector<double>   DVec;
typedef thrust::host_vector<double>     HVec;

typedef thrust::device_vector< Array<double, n> > DArrVec;
typedef thrust::host_vector< Array<double, n> >   HArrVec;

typedef ArrVec1d< double, n, HVec> HArrVec_;
typedef ArrVec1d< double, n, DVec> DArrVec_;
typedef cusp::dia_matrix<int, double, cusp::host_memory>   HMatrix;
typedef cusp::dia_matrix<int, double, cusp::device_memory> DMatrix;

template< size_t n>
struct Diagonal{
    Diagonal( const double h = 2.): h_( h){}
    
    __host__ __device__
    Array<double, n> operator() (const Array<double, n>& arr)
    {
        Array<double,n > temp(0.);
        for( unsigned i=0; i<n; i++)
            temp[i] = (2.*i+1.)/h_*arr[i];
        return temp;
    }
    private:
    double h_; 
};

template< size_t n>
cusp::coo_matrix<int, double, cusp::host_memory> createDiagonal( unsigned N)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A( n*N, n*N, n*N);
    T1D<double, n> t1d(2.);
    //std::cout << a << "\n"<<b <<std::endl;
    //assemble the matrix
    int number = 0;
    for( unsigned i=0; i<N; i++)
        for( unsigned k=0; k<n; k++)
            create::detail::add_index<double, n>(A, number, i, i, k, k, t1d(i*n+k));
    return A;
};

double function( double x) { return sin(x);}


int main()
{
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << N<<endl;
    Timer t;
    HArrVec_ hv_ = evaluate<double(&)(double), n>( function, 0, 2.*M_PI, N);
    HArrVec  hv( N);
    for( unsigned i=0; i<N; i++)
        for( unsigned j=0; j<n; j++)
            hv[i][j] = hv_(i,j);
    DArrVec  dv( hv);
    DArrVec  dv2( N);
    DArrVec_ dv_( hv_);
    DArrVec_ dv_2( dv_);
    DMatrix dm = createDiagonal<n>( N);

    t.tic();
    thrust::transform( dv.begin(), dv.end(), dv.begin(), Diagonal<n>());
    //blas1::axpby( 1., dv2.data(), 0., dv.data());
    t.toc();
    cout << "Forward thrust transform took "<<t.diff()<<"s\n";

    t.tic();
    blas2::symv( dm, dv_.data(), dv_.data());
    //blas1::axpby( 1., dv_2.data(), 2., dv_.data());
    t.toc();
    cout << "Foward cusp transform took    "<<t.diff()<<"s\n";
    t.tic();
    T1D<double, n> t1d( 2.);
    blas2::symv( t1d , dv_2.data(), dv_2.data());
    //blas2::symv( 1., T1D<double, n>( 2.), dv_2.data(), 0., dv_2.data());
    //slower than single symv( T, v, v) but faster than symv plus axpby
    t.toc();
    cout << "Foward dg transform took      "<<t.diff()<<"s\n";
    W1D<double, n> w1d( 2.);
    t.tic();
    blas2::symv( w1d, dv_2.data(), dv_2.data());
    t.toc();
    cout << "Foward W  transform took      "<<t.diff()<<"s\n";

    //test for equality...
    /*
    hv = dv;
    hv_ = dv_;
    HArrVec_ hv_2= dv_2;
    for( unsigned i=0; i<N; i++)
    {
        for( unsigned j=0; j<n; j++)
            cout << hv[i][j]  << " ";
        cout << "\n";
    }
    cout << endl;
    cout << hv_ <<endl;
    cout << endl << hv_2<<endl;
    */

    
    return 0;
}
