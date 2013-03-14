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
    cout << "Array size is: "<<n<<"\n";
    cout << "Vector size (n*N) is: "<<n*N<<"\n";
    Array_t x; 
    Array_t y; 
    for( size_t i=0; i<n; i++)
    {
        x[ i] = i;
        y[ i] = 2 - (double)i;
    }
    cout << x << y<<"\n";

    Vector dx( N, x), dy( N, y);
    thrust::host_vector<Array_t> hx(dx), hy(dy);
    Timer t;

    //Test dot product
    double dot;
    t.tic();
    dot = dg::BLAS1<Vector>::ddot( dx,dy);
    t.toc();
    std::cout << "GPU dot took "<<t.diff()<<"s\n";
    std::cout << "Result: " << dot << "\n";
    t.tic();
    dot = 0;
    for( unsigned i=0; i<N; i++)
        for( unsigned j=0; j<n; j++)
            dot += hx[i][j]*hy[i][j];
    t.toc();
    std::cout << "CPU dot took "<<t.diff()<<"s\n";
    std::cout << "Result: " << dot << "\n";

    t.tic();
    dg::BLAS1<Vector>::daxpby( 3,dx,7,dy);
    t.toc();
    std::cout << "GPU Transformation took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << dy[dy.size()-1]<< "\n";

    t.tic();
    for( unsigned i=0; i<N; i++)
        for( unsigned j=0; j<n; j++)
            hy[i][j] = 3.*hx[i][j] + 7.*hy[i][j];
    t.toc();
    std::cout << "CPU Transformation took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << hy[hy.size()-1]<< "\n";

    return 0;
}
