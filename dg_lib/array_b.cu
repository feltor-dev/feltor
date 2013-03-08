#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "timer.h"
#include "array.h"
#include "blas.h"

using namespace dg;
using namespace std;

typedef double real; //daxpby is memory bound so float will only increase by 2
struct daxpby_functor
{
    daxpby_functor( real alpha, real beta): alpha(alpha), beta(beta) {}
    template<size_t n>
    __host__ __device__
        Array<real, n>& operator()( const Array<real,n >& x, Array<real, n>& y)
        {
            //daxpby( alpha, x, beta, y);
            for( unsigned i=0; i<n; i++)
                y(i) = alpha*x(i)+beta*y(i);
            return y;
        }
    __host__ __device__
        real operator()( const real& x, const real& y)
        {
            return alpha*x+beta*y;
        }
  private:
    real alpha, beta;
};

const unsigned n = 3;
const unsigned N = 1e6; //GPU is 2-5 times faster
typedef dg::Array< real, n> Array_t;
typedef thrust::device_vector< Array_t > Vector;

template<>
struct dg::BLAS1< Vector>
{
    static void daxpby( real alpha, const Vector& x, real beta, Vector& y)
    {
        //thrust::transform( x.begin(), x.end(), y.begin(), y.begin(), daxpby_functor( alpha,beta));
        real const * raw_ptr_x = reinterpret_cast<real const*>( thrust::raw_pointer_cast(x.data()));
        thrust::device_ptr<real const> d_x = thrust::device_pointer_cast( raw_ptr_x);
        real * raw_ptr_y = reinterpret_cast<real *>( thrust::raw_pointer_cast(y.data()));
        thrust::device_ptr<real> d_y = thrust::device_pointer_cast( raw_ptr_y);
        thrust::transform( d_x, d_x+n*N, d_y, d_y, daxpby_functor( alpha,beta));
    }
};


int main()
{
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
    cudaEvent_t start, stop;
    float time;
    // create two events 
    cudaEventCreate( &start); 
    cudaEventCreate( &stop);

    cudaEventRecord( start, 0); // put event start in default stream 0 ( 0 can be omitted)
    dg::BLAS1<Vector>::daxpby( 3,dx,7,dy);
    cudaEventRecord( stop, 0); // put event stop in stream 0
    cudaEventSynchronize(stop); //since eventrecord is an asynchronous call, one has to wait before calling the next function
    cudaEventElapsedTime( &time, start, stop); // measure elapsed time in ms 
    std::cout << "GPU Transformation took "<<time/1000.<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << dy[dy.size()-1]<< "\n";

    thrust::host_vector<Array_t> hx(dx), hy(dy);
    Timer t;
    t.tic();
    for( unsigned i=0; i<N; i++)
        for( unsigned j=0; j<n; j++)
            hy[i](j) = 3.*hx[i](j) + 7.*hy[i](j);
    t.toc();
    std::cout << "CPU Transformation took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << hy[hy.size()-1]<< "\n";



    return 0;

}
