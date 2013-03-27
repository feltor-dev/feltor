#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/ell_matrix.h>
#include <cusp/dia_matrix.h>

#include "blas.h"
#include "laplace.cuh"
#include "timer.cuh"
#include "array.cuh"
#include "dlt.h"
#include "dgvec.cuh"
#include "evaluation.cuh"
#include "operators.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned N = 1e5;

typedef thrust::device_vector<double>   DVec;
typedef thrust::host_vector<double>     HVec;

typedef thrust::device_vector< Array<double, n> > DArrVec;
typedef thrust::host_vector< Array<double, n> >   HArrVec;

typedef ArrVec1d< double, n, HVec> HArrVec_;
typedef ArrVec1d< double, n, DVec> DArrVec_;
typedef cusp::ell_matrix<int, double, cusp::host_memory>   HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

template< size_t n>
struct Forward{
    Forward(): forward( DLT<n>::forward){}
    
    __host__ __device__
    Array<double, n> operator() (const Array<double, n>& arr)
    {
        Array<double,n > temp(0.);
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
                temp[i] += forward(i,j)*arr[j];
        return temp;
               
    }
    private:
    Operator<double, n> forward;
};

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
    DArrVec_ dv_2( N);
    DMatrix dm = createForward<n>( N);

    Array<double, n>* begin = reinterpret_cast<Array<double, n>* >( thrust::raw_pointer_cast(&dv_.data()[0]));
    Array<double, n>* end = begin + N-1;
    t.tic();
    thrust::transform( thrust::device_pointer_cast(begin), thrust::device_pointer_cast(end), thrust::device_pointer_cast(begin), Forward<n>());
    t.toc();
    cout << "Forward thrust transform took "<<t.diff()<<"s\n";
    t.tic();
    blas2::symv( dm, dv_.data(), dv_.data());
    t.toc();
    cout << "Foward cusp transform took    "<<t.diff()<<"s\n";

    //test for equality...
    /*
    hv_ = dv_;
    hv = dv;
    for( unsigned i=0; i<N; i++)
    {
        for( unsigned j=0; j<n; j++)
            cout << hv[i][j]  << " ";
        cout << "\n";
    }
    cout << endl;
    cout << hv_ <<endl;
    */

    
    return 0;
}
