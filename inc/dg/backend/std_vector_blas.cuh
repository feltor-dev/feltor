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
inline void doAxpby( get_value_type<Vector> alpha, 
              const Vector& x, 
              get_value_type<Vector> beta, 
              Vector& y, 
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doAxpby( alpha, x[i], beta, y[i], get_vector_category<typename Vector::value_type>());
        
}

template< class Vector>
inline void doAxpbypgz( get_value_type<Vector> alpha, 
              const Vector& x, 
              get_value_type<Vector> beta, 
              const Vector& y, 
              get_value_type<Vector> gamma, 
              Vector& z, 
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doAxpbypgz( alpha, x[i], beta, y[i], gamma, z[i], get_vector_category<typename Vector::value_type>());
        
}

template< class Vector>
inline void doScal( Vector& x, 
              get_value_type<Vector> alpha, 
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doScal( x[i], alpha, get_vector_category<typename Vector::value_type>());
        
}

template< class Vector>
inline void doPlus( Vector& x, 
              get_value_type<Vector> alpha, 
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
#endif //DG_DEBUG
    for( unsigned i=0; i<x.size(); i++)
        doPlus( x[i], alpha, get_vector_category<typename Vector::value_type>());
        
}

template<class Vector, class UnaryOp>
inline void doTransform( const Vector& x, Vector& y, UnaryOp op, VectorVectorTag)
{
    for( unsigned i=0; i<x.size(); i++)
        doTransform( x[i], y[i], op, get_vector_category<typename Vector::value_type>());
}

template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha, 
const Vector& x1, const Vector& x2, 
get_value_type<Vector> beta, 
Vector& y, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], x2[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
}
template< class Vector>
inline void doPointwiseDivide( get_value_type<Vector> alpha, 
const Vector& x1, const Vector& x2, 
get_value_type<Vector> beta, 
Vector& y, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDivide( alpha, x1[i], x2[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
}
template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha, 
    const Vector& x1, const Vector& x2, const Vector& x3,
    get_value_type<Vector> beta, 
    Vector& y, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
    assert( x1.size() == x3.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], x2[i],x3[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
}
template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha, 
    const Vector& x1, const Vector& y1, 
    get_value_type<Vector> beta, 
    const Vector& x2, const Vector& y2, 
    get_value_type<Vector> gamma, 
    Vector& z, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == y1.size() );
    assert( x1.size() == x2.size() );
    assert( x1.size() == y2.size() );
    assert( x1.size() == z.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], y1[i], beta, x2[i], y2[i], gamma,z[i], get_vector_category<typename Vector::value_type>() );
}

template< class Vector>
inline get_value_type<Vector> doDot( const Vector& x1, const Vector& x2, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
#endif //DG_DEBUG
    std::vector<exblas::Superaccumulator> acc( x1.size());
    for( unsigned i=0; i<x1.size(); i++)
        acc[i] = doDot_superacc( x1[i], x2[i], get_vector_category<typename Vector::value_type>());
    for( unsigned i=1; i<x1.size(); i++)
        acc[0].Accumulate( acc[i]);
    return acc[0].Round();
}


} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_STD_VECTOR_
