#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/inner_product.h>

#include "../blas.h"


namespace dg
{
    //TODO check for alpha = 0 and beta = 1
struct daxpby_functor
{
    daxpby_functor( double alpha, double beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        double operator()( const double& x, const double& y)
        {
            return alpha*x+beta*y;
        }
  private:
    double alpha, beta;
};

template< class ThrustVector>
double BLAS1<ThrustVector>::ddot( const Vector& x, const Vector& y)
{
    return thrust::inner_product( x.begin(), x.end(),  y.begin(), 0.0);
}

template< class ThrustVector>
void BLAS1<ThrustVector>::daxpby( double alpha, const Vector& x, double beta, Vector& y)
{
    thrust::transform( x.begin(), x.end(), y.begin(), y.begin(), daxpby_functor( alpha, beta));
}

/*
template<>
struct BLAS1<thrust::device_vector<double> >
{
    typedef thrust::device_vector<double> Vector;
    static double ddot( const Vector& x, const Vector& y)
    {
        return thrust::inner_product( x.begin(), x.end(),  y.begin(), 0.0);
    }
    
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        thrust::transform( x.begin(), x.end(), y.begin(), y.begin(), daxpby_functor( alpha, beta));
    }
};
*/



    
    
} //namespace dg



#endif //_DG_BLAS_VECTOR_
