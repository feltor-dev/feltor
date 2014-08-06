#include <iostream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "timer.cuh"
#include "grid.h"
#include "evaluation.cuh"
#include "blas.h"


double function( double x)
{
    return exp(x);
}


const unsigned n = 3;
const unsigned N = 1e6;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;


using namespace std;
int main()
{
    cout << "Array size is:             "<< n<<endl;
    cout << "Number of intervals is:    "<< N <<endl;
    dg::Grid1d<double> g( 0, 1, n, N);
    double h = 1./(double)N;
    dg::Timer t;

    t.tic();
    HVec h_v = dg::expand( function, g);
    t.toc(); 
    cout << "Expansion on host took         "<< t.diff()<<"s\n";

    t.tic();
    DVec d_v( h_v);
    t.toc(); 
    cout << "Copy of data host2device took  "<< t.diff()<<"s\n\n";
    t.tic();
    dg::blas2::symv(  dg::S1D<double>(g), d_v, d_v);
    t.toc(); 
    cout << "symv(S,v) took on device       "<< t.diff()<<"s\n";
    t.tic();
    dg::blas2::symv(  dg::S1D<double>(g), h_v, h_v);
    t.toc(); 
    cout << "symv(S,v) took on host         "<< t.diff()<<"s\n";

    double norm;
    t.tic();
    norm = dg::blas2::dot( dg::T1D<double>(g), d_v);
    t.toc(); 
    cout << "ddot(v,T, v) took on device    "<< t.diff()<<"s\n";

    t.tic();
    norm = dg::blas2::dot( dg::T1D<double>(g), h_v);
    t.toc(); 
    cout << "ddot(v,T,v) took on host       "<< t.diff()<<"s\n\n";
    cout<< "Square normalized norm "<< norm <<"\n";
    double solution = (exp(2.) -exp(0))/2.;
    cout << "Correct square norm of exp(x) is "<<solution<<endl;
    return 0;
}
