#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "timer.cuh"
#include "array.cuh"
#include "blas/thrust_vector.cuh"

using namespace dg;
using namespace std;

typedef double real; //daxpby is memory bound so float will only increase by 2

const unsigned n = 3;
const unsigned N = 1e6; //GPU is 2-5 times faster
typedef dg::Array< real, n> Array_t;
typedef thrust::device_vector< Array_t > Vector;

int main()
{
    std::cout << "Hello\n";
    Array_t x; 
    Array_t y; 
    for( size_t i=0; i<n; i++)
    {
        x( i) = i;
        y( i) = (double)i-4;
    }
    cout << x << y<<"\n";

    Vector dx( N, x), dy( N, y);
    //thrust::device_vector<dg::Array<real,3> > dn(100, n);
    //thrust::device_vector<dg::Array<real,3> > dm(100, m);
    std::cout << "Hello\n";
    Timer t;
    t.tic();
    dg::BLAS1<Vector>::daxpby( 3,dx,7,dy);
    t.toc();
    std::cout << "GPU Transformation took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << dy[dy.size()-1]<< "\n";

    thrust::host_vector<Array_t> hx(dx), hy(dy);
    t.tic();
    for( unsigned i=0; i<N; i++)
        for( unsigned j=0; j<n; j++)
            hy[i](j) = 3.*hx[i](j) + 7.*hy[i](j);
    t.toc();
    std::cout << "CPU Transformation took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << hy[hy.size()-1]<< "\n";



    return 0;

}
