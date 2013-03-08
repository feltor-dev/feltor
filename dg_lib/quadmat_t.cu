#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "quadmat.cuh"

using namespace dg;
using namespace std;

struct daxpby_functor
{
    daxpby_functor( double alpha, double beta): alpha(alpha), beta(beta) {}
    template<size_t n>
    __host__ __device__
        QuadMat<double, n>& operator()( const QuadMat<double,n >& x, QuadMat<double, n>& y)
        {
            daxpby( alpha, x, beta, y);
            return y;
        }
  private:
    double alpha, beta;
};

int main()
{
    QuadMat<double, 3> m; 
    QuadMat<double, 3> n; 
    for( size_t i=0; i<3; i++)
        for( size_t j=0; j<3; j++)
            n( i, j) = i + j;
    n( 2,2 ) = 17;
    m=n;

    cout << "Test of QuadMat class\n";
    cout << "Output operations m and n:\n";
    cout << m <<endl;
    cout << n <<endl;
    QuadMat<double, 3> k(n);
    cout << "k(n)\n" << k<<endl;
    
    k = n;
    cout << "k = n\n" << k<<endl;
    cout << "comparison: \n k == n: " << (k==n) << " and k !=n " << (k!=n) <<endl;
    n(1,1) = 0;
    cout << "comparison with changed n: \n k == n: " << (k==n) << " and k !=n " << (k!=n) <<endl;

    thrust::device_vector<dg::QuadMat<double,3> > dn(100, n);
    thrust::device_vector<dg::QuadMat<double,3> > dm(100, m);
    thrust::transform( dn.begin(), dn.end(), dm.begin(), dm.begin(), daxpby_functor( 3,7));
    cudaThreadSynchronize();
    std::cout << "Test of device transform daxpby(3,n,7,m)\n" << dm[0]<<endl;


    return 0;

}


