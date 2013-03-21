#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../vector_categories.h"
#include "../vector_traits.h"


namespace dg
{

template<typename T>
struct VectorTraits< thrust::host_vector<T> >
{
    typedef T value_type;
    typedef ThrustVectorTag vector_category;
};

template<typename T>
struct VectorTraits< thrust::device_vector<T> >
{
    typedef T value_type;
    typedef ThrustVectorTag vector_category;
};

namespace blas1
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
        thrust::transform( y.begin(), y.end(), y.begin(), detail::daxpby_functor( 0, beta));
        return;
    }
    thrust::transform( x.begin(), x.end(), y.begin(), y.begin(), detail::daxpby_functor( alpha, beta) );
}



} //namespace detail

} //namespace blas1
    
} //namespace dg

#include "../blas.h"


#endif //_DG_BLAS_VECTOR_
