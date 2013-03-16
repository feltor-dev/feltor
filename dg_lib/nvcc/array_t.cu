#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "array.cuh"
#include "blas.h"

using namespace dg;
using namespace std;
struct daxpby_functor
{
    daxpby_functor( double alpha, double beta): alpha(alpha), beta(beta) {}
    template<size_t n>
    __host__ __device__
        Array<double, n>& operator()( const Array<double,n >& x, Array<double, n>& y)
        {
            daxpby( alpha, x, beta, y);
            return y;
        }
  private:
    double alpha, beta;
};

int main()
{
    Array<double, 3> m; 
    Array<double, 3> n; 
    for( size_t i=0; i<3; i++)
        n( i) = i;
    n( 2 ) = 17;
    m=n;

    cout << "Test of Array class\n";
    cout << "Output operations m and n:\n";
    cout << m <<endl;
    cout << n <<endl;
    Array<double, 3> k(n);
    cout << "k(n)\n" << k<<endl;
    k = n;
    cout << "k = n\n" << k<<endl;
    n(1) = 0;

    thrust::device_vector<dg::Array<double,3> > dn(100, n);
    thrust::device_vector<dg::Array<double,3> > dm(100, m);
    thrust::transform( dn.begin(), dn.end(), dm.begin(), dm.begin(), daxpby_functor( 3,7));
    cudaThreadSynchronize();
    std::cout << "Test of device transform daxpby(3,n,7,m)\n" << dm[0]<<endl;


    return 0;

}

