#ifndef _DG_BLAS_STD_VECTOR_
#define _DG_BLAS_STD_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <vector>
#include <array>
#include "thrust_vector_blas.cuh"
#include "vector_categories.h"
#include "vector_traits.h"
#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP

///@cond
namespace dg
{
namespace blas1
{
namespace detail
{

template<class To, class From>
To doTransfer( const From& src, ArrayVectorTag, AnyVectorTag)
{
    To t;
    for (unsigned i=0; i<t.size(); i++)
        t[i] = src;
    return t;
}


template< class Vector>
inline get_value_type<Vector> doDot( const Vector& x1, const Vector& x2, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
#endif //DG_DEBUG
    std::vector<std::vector<int64_t>> acc( x1.size());
    for( unsigned i=0; i<x1.size(); i++)
        acc[i] = doDot_superacc( x1[i], x2[i], get_vector_category<typename Vector::value_type>());
    for( unsigned i=1; i<x1.size(); i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return exblas::cpu::Round(&(acc[0][0]));
}
template<class Vector, class UnaryOp>
inline void doEvaluate( VectorVectorTag, Vector& y, get_value_type<Vector> alpha, UnaryOp op, const Vector& x)
{
    for( unsigned i=0; i<x.size(); i++)
        doEvaluate( get_vector_category<typename Vector::value_type>(), y[i], alpha, op, x[i]);
}
template<class Vector, class UnaryOp>
inline void doEvaluate( VectorVectorTag, Vector& z, get_value_type<Vector> alpha, UnaryOp op, const Vector& x, const Vector& y)
{
    for( unsigned i=0; i<x.size(); i++)
        doEvaluate( get_vector_category<typename Vector::value_type>(), z[i], alpha, op, x[i], y[i]);
}
#ifdef _OPENMP
template< class Vector>
inline void doScal( Vector& x, get_value_type<Vector> alpha, VectorVectorTag, OmpTag)
{
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x.size(); i++)
                doScal( x[i], alpha, get_vector_category<typename Vector::value_type>());
        }
    }
    else
        for( unsigned i=0; i<x.size(); i++)
            doScal( x[i], alpha, get_vector_category<typename Vector::value_type>());
}
template< class Vector>
inline void doPlus( Vector& x, get_value_type<Vector> alpha, VectorVectorTag, OmpTag)
{
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x.size(); i++)
                doPlus( x[i], alpha, get_vector_category<typename Vector::value_type>());
        }
    }
    else
        for( unsigned i=0; i<x.size(); i++)
            doPlus( x[i], alpha, get_vector_category<typename Vector::value_type>());
}
template< class Vector>
inline void doAxpby( get_value_type<Vector> alpha,
              const Vector& x,
              get_value_type<Vector> beta,
              Vector& y,
              VectorVectorTag, OmpTag)
{
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x.size(); i++)
                doAxpby( alpha, x[i], beta, y[i], get_vector_category<typename Vector::value_type>());
        }
    }
    else
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
              VectorVectorTag, OmpTag)
{
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x.size(); i++)
                doAxpbypgz( alpha, x[i], beta, y[i], gamma, z[i], get_vector_category<typename Vector::value_type>());
        }
    }
    else
        for( unsigned i=0; i<x.size(); i++)
            doAxpbypgz( alpha, x[i], beta, y[i], gamma, z[i], get_vector_category<typename Vector::value_type>());
}
template< class Vector>
inline void doPointwiseDivide( get_value_type<Vector> alpha,
    const Vector& x1, const Vector& x2,
    get_value_type<Vector> beta,
    Vector& y, VectorVectorTag, OmpTag)
{
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x1.size(); i++)
                doPointwiseDivide( alpha, x1[i], x2[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
        }
    }
    else
        for( unsigned i=0; i<x1.size(); i++)
            doPointwiseDivide( alpha, x1[i], x2[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
}
template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha,
    const Vector& x1, const Vector& x2, const Vector& x3,
    get_value_type<Vector> beta,
    Vector& y, VectorVectorTag, OmpTag)
{
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x1.size(); i++)
                doPointwiseDot( alpha, x1[i], x2[i],x3[i], beta, y[i], get_vector_category<typename Vector::value_type>());
        }
    }
    else
        for( unsigned i=0; i<x1.size(); i++)
            doPointwiseDot( alpha, x1[i], x2[i],x3[i], beta, y[i], get_vector_category<typename Vector::value_type>());

}
template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha,
    const Vector& x1, const Vector& y1,
    get_value_type<Vector> beta,
    const Vector& x2, const Vector& y2,
    get_value_type<Vector> gamma,
    Vector& z, VectorVectorTag, OmpTag)
{
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x1.size(); i++)
                doPointwiseDot( alpha, x1[i], y1[i], beta, x2[i], y2[i], gamma,z[i], get_vector_category<typename Vector::value_type>() );
        }
    }
    else
        for( unsigned i=0; i<x1.size(); i++)
            doPointwiseDot( alpha, x1[i], y1[i], beta, x2[i], y2[i], gamma,z[i], get_vector_category<typename Vector::value_type>() );
}
#endif //_OPENMP

template< class Vector>
inline void doScal( Vector& x, get_value_type<Vector> alpha, VectorVectorTag, AnyPolicyTag)
{
    for( unsigned i=0; i<x.size(); i++)
        doScal( x[i], alpha, get_vector_category<typename Vector::value_type>());
}
template< class Vector>
inline void doPlus( Vector& x, get_value_type<Vector> alpha, VectorVectorTag, AnyPolicyTag)
{
    for( unsigned i=0; i<x.size(); i++)
        doPlus( x[i], alpha, get_vector_category<typename Vector::value_type>());
}
template< class Vector>
inline void doAxpby( get_value_type<Vector> alpha,
              const Vector& x,
              get_value_type<Vector> beta,
              Vector& y,
              VectorVectorTag, AnyPolicyTag)
{
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
              VectorVectorTag, AnyPolicyTag)
{
    for( unsigned i=0; i<x.size(); i++)
        doAxpbypgz( alpha, x[i], beta, y[i], gamma, z[i], get_vector_category<typename Vector::value_type>());
}


template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha,
    const Vector& x1, const Vector& x2,
    get_value_type<Vector> beta,
    Vector& y, VectorVectorTag, AnyPolicyTag)
{
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], x2[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
}
template< class Vector>
inline void doPointwiseDivide( get_value_type<Vector> alpha,
    const Vector& x1, const Vector& x2,
    get_value_type<Vector> beta,
    Vector& y, VectorVectorTag, AnyPolicyTag)
{
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDivide( alpha, x1[i], x2[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
}
template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha,
    const Vector& x1, const Vector& x2, const Vector& x3,
    get_value_type<Vector> beta,
    Vector& y, VectorVectorTag, AnyPolicyTag)
{
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], x2[i],x3[i], beta, y[i], get_vector_category<typename Vector::value_type>() );
}
template< class Vector>
inline void doPointwiseDot( get_value_type<Vector> alpha,
    const Vector& x1, const Vector& y1,
    get_value_type<Vector> beta,
    const Vector& x2, const Vector& y2,
    get_value_type<Vector> gamma,
    Vector& z, VectorVectorTag, AnyPolicyTag)
{
    for( unsigned i=0; i<x1.size(); i++)
        doPointwiseDot( alpha, x1[i], y1[i], beta, x2[i], y2[i], gamma,z[i], get_vector_category<typename Vector::value_type>() );
}
/////////////////////dispatch////////////////////////
template< class Vector>
inline void doScal( Vector& x, get_value_type<Vector> alpha, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
#endif //DG_DEBUG
    doScal( x, alpha, VectorVectorTag(), get_execution_policy<Vector>());

}
template< class Vector>
inline void doPlus( Vector& x, get_value_type<Vector> alpha, VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( !x.empty());
#endif //DG_DEBUG
    doPlus( x, alpha, VectorVectorTag(), get_execution_policy<Vector>());
}

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
    doAxpby( alpha, x, beta, y, VectorVectorTag(), get_execution_policy<Vector>());
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
    doAxpbypgz( alpha, x, beta, y, gamma, z, VectorVectorTag(), get_execution_policy<Vector>());

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
    doPointwiseDot( alpha, x1, x2, beta, y, VectorVectorTag(), get_execution_policy<Vector>() );
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
    doPointwiseDivide( alpha, x1, x2, beta, y, VectorVectorTag(), get_execution_policy<Vector>() );
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
    doPointwiseDot( alpha, x1, x2,x3, beta, y, VectorVectorTag(), get_execution_policy<Vector>() );
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
    doPointwiseDot( alpha, x1, y1, beta, x2, y2, gamma,z, VectorVectorTag(), get_execution_policy<Vector>() );
}



} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_STD_VECTOR_
