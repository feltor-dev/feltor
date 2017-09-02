#ifndef _DG_BLAS_STD_VECTOR_
#define _DG_BLAS_STD_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

#include "thrust_vector_blas.cuh"
#include "vector_categories.h"
#include "vector_traits.h"

//assume that each element of a std::vector is a vector itself

///@cond
namespace dg
{
namespace blas1
{
namespace detail
{

template< class Vector>
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha, 
              const std::vector<Vector>& x, 
              typename VectorTraits<Vector>::value_type beta, 
              std::vector<Vector>& y, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doAxpby( alpha, x[i], beta, y[i], typename VectorTraits<Vector>::vector_category());
        
}

template< class Vector>
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha, 
              const std::vector<Vector>& x, 
              typename VectorTraits<Vector>::value_type beta, 
              const std::vector<Vector>& y, 
              std::vector<Vector>& z, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doAxpby( alpha, x[i], beta, y[i], z[i], typename VectorTraits<Vector>::vector_category());
        
}

template< class Vector>
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha, 
              const std::vector<Vector>& x, 
              typename VectorTraits<Vector>::value_type beta, 
              const std::vector<Vector>& y, 
              typename VectorTraits<Vector>::value_type gamma, 
              std::vector<Vector>& z, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doAxpby( alpha, x[i], beta, y[i], gamma, z[i], typename VectorTraits<Vector>::vector_category());
        
}

template<class container>
inline void doCopy( const std::vector<container>& x, std::vector<container>& y, StdVectorTag)
{
    for( unsigned i=0; i<x.size(); i++)
        doCopy( x[i], y[i], typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doScal( std::vector<Vector>& x, 
              typename VectorTraits<Vector>::value_type alpha, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doScal( x[i], alpha, typename VectorTraits<Vector>::vector_category());
        
}

template< class Vector>
inline void doPlus( std::vector<Vector>& x, 
              typename VectorTraits<Vector>::value_type alpha, 
              StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doPlus( x[i], alpha, typename VectorTraits<Vector>::vector_category());
        
}

template<class container, class UnaryOp>
inline void doTransform( const std::vector<container>& x, std::vector<container>& y, UnaryOp op, StdVectorTag)
{
    for( unsigned i=0; i<x.size(); i++)
        doTransform( x[i], y[i], op, typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doPointwiseDot( const std::vector<Vector>& x1, const std::vector<Vector>& x2, std::vector<Vector>& y, StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( x1[i], x2[i], y[i], typename VectorTraits<Vector>::vector_category() );
}
template< class Vector>
inline void doPointwiseDot( typename VectorTraits<Vector>::value_type alpha, 
const std::vector<Vector>& x1, const std::vector<Vector>& x2, 
typename VectorTraits<Vector>::value_type beta, 
std::vector<Vector>& y, StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], x2[i], beta, y[i], typename VectorTraits<Vector>::vector_category() );
}
template< class Vector>
inline void doPointwiseDot( typename VectorTraits<Vector>::value_type alpha, 
const std::vector<Vector>& x1, const std::vector<Vector>& y1, 
typename VectorTraits<Vector>::value_type beta, 
const std::vector<Vector>& x2, const std::vector<Vector>& y2, 
typename VectorTraits<Vector>::value_type gamma, 
std::vector<Vector>& z, StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == y1.size() );
    assert( x1.size() == x2.size() );
    assert( x1.size() == y2.size() );
    assert( x1.size() == z.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], y1[i], beta, x2[i], y2[i], gamma,z[i], typename VectorTraits<Vector>::vector_category() );
}

template< class Vector>
inline void doPointwiseDivide( const std::vector<Vector>& x1, const std::vector<Vector>& x2, std::vector<Vector>& y, StdVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDivide( x1[i], x2[i], y[i], typename VectorTraits<Vector>::vector_category());
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
        sum += doDot( x1[i], x2[i], typename VectorTraits<Vector>::vector_category());
    return sum;
}


} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_STD_VECTOR_
