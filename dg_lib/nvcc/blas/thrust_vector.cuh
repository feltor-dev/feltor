#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

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

template< class Vector>
typename Vector::value_type doDot( const Vector& x, const Vector& y, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    return thrust::inner_product( x.begin(), x.end(),  y.begin(), 0.0);
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              Vector& y, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
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

template< class Vector>
inline void doPointwiseDot( const Vector& x1, const Vector& x2, Vector& y, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    thrust::transform( x1.begin(), x1.end(), x2.begin(), y.begin(), 
                        thrust::multiplies<typename Vector::value_type>());
}



} //namespace detail
} //namespace blas1
} //namespace dg

#endif //_DG_BLAS_VECTOR_
