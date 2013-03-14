#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "quadmat.cuh"
#include "timer.cuh"
#include "blas/thrust_vector.cuh"

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
const unsigned N = 1e5; //GPU is 2-5 times faster
typedef dg::QuadMat< real, n> Matrix;
typedef thrust::device_vector< Matrix > Vector;
int main()
{
    
    cout << "QuadMat size is: "<<n<<"\n";
    cout << "Vector size (n*n*N) is: "<<n*N<<"\n";
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
    Vector da( a), db( b);
    cudaThreadSynchronize();

    //Test dot product
    double dot;
    t.tic();
    dot = dg::BLAS1< Vector>::ddot( da, db);
    t.toc();
    std::cout << "GPU ddot took "<<t.diff()<<"s\n";
    std::cout << "Result: " << dot<< "\n";

    //Test daxpy on device and host
    t.tic();
    dg::BLAS1< Vector>::daxpby( 3., da, 7., db); 
    t.toc();
    std::cout << "GPU daxpby took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << db[db.size()-1]<< "\n";
    t.tic();
    for( unsigned i=0; i<N; i++)
        //b[i] = 3.*a[i] + 7.*b[i];
        daxpby( 3., a[i], 7, b[i]);
    t.toc();
    std::cout << "CPU daxpby took "<<t.diff()<<"s\n";
    std::cout << "Result (should be 3*x+7*y): \n" << b[b.size()-1]<< "\n";

    return 0;
}
