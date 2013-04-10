#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "blas1.h"
#include "arrvec1d.cuh"

const unsigned n = 3;
const unsigned N = 5;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;

using namespace std;
int main()
{
    HArrVec h_v( N,3);
    DArrVec d_v;
    //d_v.data() = h_v.data(); //this will trigger cryptic warnings from thrust
    h_v( 2,2) = 4.;
    DArrVec d_v2( h_v);
    d_v = d_v2;
    std::cout << h_v <<std::endl;
    //d_v( 1,2) = 7.; //doesn't work

    HArrVec h_v2( d_v);
    std::cout << h_v2<<std::endl;

    dg::blas1::pointwiseDot( d_v.data(), d_v2.data(), d_v2.data());
    cout << d_v2 << endl;


    return 0;
}
