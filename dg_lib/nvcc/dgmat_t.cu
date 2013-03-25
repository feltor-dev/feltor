#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dgmat.cuh"

const unsigned n = 3;
const unsigned N = 2;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrMat;
typedef dg::ArrVec2d< double, n, DVec>  DArrMat;

int main()
{
    HArrMat h_v( N,N,2);
    /*
    DArrVec d_v;
    d_v.data() = h_v.data(); //this will trigger cryptic warnings from thrust
    */
    h_v( 0, 1, 1,1) = 7;
    h_v( 0, 1, 1,2) = 0;
    h_v( 0, 1, 2,2) = 9;
    DArrMat d_v( h_v.data(), N);
    std::cout << h_v <<std::endl;
    return 0;
}
