#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#include <thrust/inner_product.h>

#include "../blas.h"


namespace dg
{
namespace detail
{
struct daxpby_functor
{
    daxpby_functor( double alpha, double beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        double operator()( const double& x, const double& y)
        {
            return alpha*x+beta*y;
        }
    __host__ __device__
        double operator()( const double& y)
        {
            return beta*y;
        }
  private:
    double alpha, beta;
};

//not faster
/*
struct dot_functor
{
    __host__ __device__
    double operator()( const double& x)
    {
        return x*x;
    }
    __host__ __device__
    double operator()( const double& x, const double& y)
    {
        return x+y;
    }
};
*/ 
} //namespace detail

template< class ThrustVector>
double BLAS1<ThrustVector>::ddot( const Vector& x, const Vector& y)
{
    /*
    if( &x == &y) 
        return thrust::transform_reduce( x.begin(), x.end(), detail::dot_functor(), 0.0, detail::dot_functor());
        */
    return thrust::inner_product( x.begin(), x.end(),  y.begin(), 0.0);
}

template< class ThrustVector>
void BLAS1<ThrustVector>::daxpby( double alpha, const Vector& x, double beta, Vector& y)
{
    if( alpha == 0)
    {
        if( beta == 1) 
            return;
        thrust::transform( y.begin(), y.end(), y.begin(), detail::daxpby_functor( 0, beta));
        return;
    }
    thrust::transform( x.begin(), x.end(), y.begin(), y.begin(), detail::daxpby_functor( alpha, beta) );
}




    
    
} //namespace dg



#endif //_DG_BLAS_VECTOR_
