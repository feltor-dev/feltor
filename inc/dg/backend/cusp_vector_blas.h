#pragma once

#include <cassert>
#include <cusp/array1d.h>
#include <cusp/blas/blas.h>

#include "vector_categories.h"
#include "vector_traits.h"


//this file specializes blas1 functions when it is better than the thrust versions
namespace dg
{

template<class T,class M>
struct VectorTraits<cusp::array1d<T,M> >
{
    typedef typename cusp::array1d<T,M>::value_type value_type;
    typedef CuspVectorTag vector_category; //default is a ThrustVector
};

namespace blas1
{
    ///@cond
namespace detail
{


template< class Vector>
typename Vector::value_type doDot( const Vector& x, const Vector& y, CuspVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    return cusp::blas::dot(x,y);
}

template< class Vector>
inline void doScal(  Vector& x, 
              typename Vector::value_type alpha, 
              CuspVectorTag)
{
    cusp::blas::scal( x, alpha);
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              Vector& y, 
              CuspVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    if( beta == 1. )
    {
        cusp::blas::axpy(x,y,alpha);
        return;
    }
    if( alpha == 0.)
    {
        cusp::blas::scal(y, beta);
        return;
    }
    cusp::blas::axpby(x,y,y, alpha, beta);
}


template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              const Vector& y, 
              Vector& z, 
              CuspVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( x.size() == z.size() );
#endif //DG_DEBUG
    if( &y == &z)
    {
        doAxpby( alpha,x,beta,z, CuspVectorTag());
        return;
    }
    cusp::blas::axpby(x,y,z, alpha, beta);
}

template< class Vector>
inline void doPointwiseDot( const Vector& x1, const Vector& x2, Vector& y, CuspVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    cusp::blas::xmy(x1,x2,y);
}


} //namespace detail
///@endcond
} //namespace blas1
} //namespace dg

