#ifndef _DG_BLAS_STD_VECTOR_
#define _DG_BLAS_STD_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

#include "thrust_vector.cuh"
#include "../vector_categories.h"
#include "../vector_traits.h"


namespace dg
{
namespace blas1
{
namespace detail
{

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const std::vector<Vector>& x, 
              typename Vector::value_type beta, 
              std::vector<Vector>& y, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doAxpby( alpha, x[i], beta, y[i], ThrustVectorTag());
        
}

template< class Vector>
inline void doPointwiseDot( const Vector& x1, const Vector& x2, Vector& y, StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( x1[i], x2[i], y[i], ThrustVectorTag() );
}



} //namespace detail
} //namespace blas1
} //namespace dg

#endif //_DG_BLAS_STD_VECTOR_
