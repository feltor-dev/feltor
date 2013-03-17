#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dgvec.cuh"

const unsigned n = 3;
const unsigned N = 5;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;

int main()
{
    HArrVec h_v( N,2);
    /*
    DArrVec d_v;
    d_v.data() = h_v.data(); //this will trigger cryptic warnings from thrust
    */
    DArrVec d_v( h_v.data());
    std::cout << h_v <<std::endl;
    return 0;
}
