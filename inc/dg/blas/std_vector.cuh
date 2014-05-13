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

//assume that each element of a std::vector is a vector itself

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
        axpby( alpha, x[i], beta, y[i]);
        
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const std::vector<Vector>& x, 
              typename Vector::value_type beta, 
              const std::vector<Vector>& y, 
              std::vector<Vector>& z, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        axpby( alpha, x[i], beta, y[i], z[i]);
        
}

template< class Vector>
inline void doScal( std::vector<Vector>& x, 
              typename Vector::value_type alpha, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        scal( x[i], alpha);
        
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
        pointwiseDot( x1[i], x2[i], y[i] );
}

template< class Vector>
inline void doPointwiseDivide( const Vector& x1, const Vector& x2, Vector& y, StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        pointwiseDivide( x1[i], x2[i], y[i] );
}

template< class Vector>
inline typename VectorTraits<Vector>::value_type doDot( const std::vector<Vector>& x1, const std::vector<Vector>& x2, StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
#endif //DG_DEBUG
    typename VectorTraits<Vector>::value_type sum =0;
    for( unsigned i=0; i<x1.size(); i++)
        sum += dot( x1[i], x2[i] );
    return sum;
}


} //namespace detail
} //namespace blas1
} //namespace dg

#endif //_DG_BLAS_STD_VECTOR_
