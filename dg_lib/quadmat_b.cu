#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "quadmat.cuh"
#include "timer.h"

typedef double real; //daxpby is memory bound so float will only increase by 2
struct daxpby_functor
{
    daxpby_functor( real alpha, real beta): alpha(alpha), beta(beta) {}
    template<size_t n>
    __host__ __device__
        dg::QuadMat<real, n>& operator()( const dg::QuadMat<real,n >& x, dg::QuadMat<real, n>& y)
        {
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    y(i,j) = alpha*x(i,j) + beta*y(i,j);
            return y;
        }
    __host__ __device__
    real operator()( const real& x, const real& y){ return alpha*x+ beta*y;}

  private:
    real alpha, beta;
};

const unsigned n = 3;
const unsigned N = 1e6; //GPU is 2-5 times faster
typedef dg::QuadMat< real, n> Matrix;
typedef thrust::device_vector< real > Vector;
int main()
{
    
    dg::Timer t;
    Matrix x, y;
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
        {
            x(i,j) = (real)(i+j);
            y(i,j) = (real)i-(real)j;
        }
    std::cout << " x and y are\n"<<x<<"\n"<<y<<"\n";
    //Vector xx( N, x), yy( N, y);
    //make std vectors 
    std::vector<Matrix> a(N, x), b(N,y);
    //convert to real vectors
    std::vector<real> aa(N*n*n), bb(aa);
    for( unsigned i=0; i<N; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<n; k++)
            {
                aa[i*n*n + j*n + k] = a[i](j,k);
                bb[i*n*n + j*n + k] = b[i](j,k);
            }
    thrust::host_vector<real> ha( aa), hb(bb);
    Vector da( ha), db(hb);
    cudaThreadSynchronize();

    cudaEvent_t start, stop;
    float time;
    // create two events 
    cudaEventCreate( &start); 
    cudaEventCreate( &stop);

    cudaEventRecord( start, 0); // put event start in default stream 0 ( 0 can be omitted)
    thrust::transform( da.begin(), da.end(), db.begin(), db.begin(), daxpby_functor( 3,7));
    cudaEventRecord( stop, 0); // put event stop in stream 0
    cudaEventSynchronize(stop); //since eventrecord is an asynchronous call, one has to wait before calling the next function
    cudaEventElapsedTime( &time, start, stop); // measure elapsed time in ms 
    std::cout << "GPU Transformation took "<<time/1000.<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << db[db.size()-1]<< "\n";
    t.tic();
    for( unsigned i=0; i<N; i++)
        //b[i] = 3.*a[i] + 7.*b[i];
        daxpby( 3., a[i], 7, b[i]);
    t.toc();
    std::cout << "CPU Transformation took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << b[b.size()-1]<< "\n";

    return 0;
}
