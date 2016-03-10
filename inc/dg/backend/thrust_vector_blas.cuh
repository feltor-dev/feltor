#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vector_categories.h"
#include "vector_traits.h"


namespace dg
{
namespace blas1
{
    ///@cond
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

template< class Vector1, class Vector2>
void doTransfer( const Vector1& in, Vector2& out, ThrustVectorTag, ThrustVectorTag)
{
    out.resize(in.size());
    thrust::copy( in.begin(), in.end(), out.begin());
}

template< class Vector>
typename Vector::value_type doDot( const Vector& x, const Vector& y, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    typedef typename Vector::value_type value_type;
    return thrust::inner_product( x.begin(), x.end(),  y.begin(), value_type(0));
}
template< class Vector>
inline void doScal(  Vector& x, 
              typename Vector::value_type alpha, 
              ThrustVectorTag)
{
    thrust::transform( x.begin(), x.end(), x.begin(), 
            detail::Axpby_Functor<typename Vector::value_type>( 0, alpha));
}
template< class Vector, class UnaryOp>
inline void doTransform(  const Vector& x, Vector& y,
                          UnaryOp op,
                          ThrustVectorTag)
{
    thrust::transform( x.begin(), x.end(), y.begin(), op);
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
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              const Vector& y, 
              Vector& z, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( x.size() == z.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        thrust::transform( y.begin(), y.end(), z.begin(), 
                detail::Axpby_Functor<typename Vector::value_type>( 0, beta));
        return;
    }
    if( beta == 0)
    {
        thrust::transform( x.begin(), x.end(), z.begin(), 
                detail::Axpby_Functor<typename Vector::value_type>( 0, alpha));
        return;
    }
    thrust::transform( x.begin(), x.end(), y.begin(), z.begin(), 
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
                        thrust::multiplies<typename VectorTraits<Vector>::value_type>());
}

template < class Vector>
struct ThrustVectorDoSymv
{
    typedef typename Vector::value_type value_type;
    typedef thrust::tuple< value_type, value_type> Pair; 
    __host__ __device__
        ThrustVectorDoSymv( value_type alpha, value_type beta): alpha_(alpha), beta_(beta){}

    __host__ __device__
        value_type operator()( const value_type& y, const Pair& p) 
        {
            return alpha_*thrust::get<0>(p)*thrust::get<1>(p) + beta_*y;
        }
  private:
    value_type alpha_, beta_;
};

template<class Vector>
inline void doPointwiseDot(  
              typename Vector::value_type alpha, 
              const Vector& x1,
              const Vector& x2, 
              typename Vector::value_type beta, 
              Vector& y, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == y.size() && x2.size() == y.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        if( beta == 1) 
            return;
        dg::blas1::detail::doScal(y, beta, dg::ThrustVectorTag());
        return;
    }
    thrust::transform( 
        y.begin(), y.end(),
        thrust::make_zip_iterator( thrust::make_tuple( x1.begin(), x2.begin() )),  
        y.begin(),
        detail::ThrustVectorDoSymv<Vector>( alpha, beta)
    ); 
}

template< class Vector>
inline void doPointwiseDivide( const Vector& x1, const Vector& x2, Vector& y, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    thrust::transform( x1.begin(), x1.end(), x2.begin(), y.begin(), 
                        thrust::divides<typename VectorTraits<Vector>::value_type>());
}



} //namespace detail
///@endcond
} //namespace blas1
} //namespace dg

#endif //_DG_BLAS_VECTOR_
