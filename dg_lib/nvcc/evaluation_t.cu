#include <iostream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "operators.cuh"
#include "preconditioner.cuh"


double function( double x)
{
    return exp(x);
}


const unsigned n = 3;
const unsigned N = 20;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;


using namespace std;
int main()
{
    cout << "Polynomial order is (n-1): "<< n-1<<endl;
    double h = 1./(double)N;
    HArrVec h_v = dg::expand< double(&) (double), n>( function, 0, 1, N);
    DArrVec d_v( h_v.data());
    dg::BLAS2< dg::S<n>, HVec>::dsymv( 1., dg::S<n>(h), h_v.data(), 0., h_v.data());
    double norm = dg::BLAS2< dg::T<n>, DVec>::ddot( h_v.data(), dg::T<n>(h), h_v.data());
    cout<< "Square normalized norm "<< norm <<"\n";
    double solution = (exp(2.) -exp(0))/2.;
    cout << "Correct square norm of exp(x) is "<<solution<<endl;
    return 0;
}

