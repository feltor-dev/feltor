#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../vector_categories.h"
#include "../vector_traits.h"


namespace dg
{
namespace blas1
{
namespace detail
{

template< typename value_type>
struct Axpby_Functor
{
    Axpby_Functor( value_type alpha, value_type beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        value_type operator()( const value_type& x, const value_type& y)
        {
            return alpha*x+beta*y;
        }
    __host__ __device__
        value_type operator()( const value_type& y)
        {
            return beta*y;
        }
  private:
    value_type alpha, beta;
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

template< class Vector>
typename Vector::value_type doDot( const Vector& x, const Vector& y, ThrustVectorTag)
{
    /*
    if( &x == &y) 
        return thrust::transform_reduce( x.begin(), x.end(), detail::dot_functor(), 0.0, detail::dot_functor());
        */
    return thrust::inner_product( x.begin(), x.end(),  y.begin(), 0.0);
}

template< class Vector>
void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              Vector& y, 
              ThrustVectorTag)
{
    if( alpha == 0)
    {
        if( beta == 1) 
            return;
        thrust::transform( y.begin(), y.end(), y.begin(), 
                detail::Axpby_Functor<typename Vector::value_type>( 0, beta));
        return;
    }
    thrust::transform( x.begin(), x.end(), y.begin(), y.begin(), 
            detail::Axpby_Functor< typename Vector::value_type>( alpha, beta) );
}



} //namespace detail

} //namespace blas1
    
} //namespace dg

#include "../blas.h"


#endif //_DG_BLAS_VECTOR_
