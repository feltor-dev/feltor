#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "thrust_vector_blas.cuh" //load thrust_vector BLAS1 routines
#include "vector_categories.h"

namespace dg{
namespace blas2{
    ///@cond
namespace detail{

//thrust vector preconditioner
template< class Vector1, class Vector2>
void doTransfer( const Vector1& in, Vector2& out, ThrustMatrixTag, ThrustMatrixTag)
{
    out.resize(in.size());
    thrust::copy( in.begin(), in.end(), out.begin());
}

template < class Vector>
struct ThrustVectorDoDot
{
    typedef typename VectorTraits<Vector>::value_type value_type;
    typedef thrust::tuple< value_type, value_type> Pair; 
    __host__ __device__
        value_type operator()( const value_type & x, const Pair& p) {
            return thrust::get<0>(p)*thrust::get<1>(p)*x;
        }
    __host__ __device__
        value_type operator()( const value_type& x, const value_type& p) {
            return p*x*x;
        }
};

template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Vector& x, const Matrix& m, const Vector& y, ThrustMatrixTag, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
    typedef typename MatrixTraits<Matrix>::value_type value_type;
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    value_type sum = 0;
    unsigned size=x.size();
    #pragma omp parallel for simd reduction(+:sum) 
    for( unsigned i=0; i<size; i++)
        sum += x[i]*m[i]*y[i];
    return sum;
#else
    return thrust::inner_product(  x.begin(), x.end(), 
                            thrust::make_zip_iterator( thrust::make_tuple( y.begin(), m.begin())  ), 
                            value_type(0),
                            thrust::plus<value_type>(),
                            detail::ThrustVectorDoDot<Matrix>()
                            );
#endif
}

template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::ThrustMatrixTag, dg::ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( m.size() == x.size());
#endif //DG_DEBUG
    typedef typename MatrixTraits<Matrix>::value_type value_type;
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    value_type sum = 0;
    unsigned size=x.size();
    #pragma omp parallel for simd reduction(+:sum)
    for( unsigned i=0; i<size; i++)
        sum += x[i]*x[i]*m[i];
    return sum;
#else
    return thrust::inner_product( x.begin(), x.end(),
                                  m.begin(),
                                  value_type(0),
                                  thrust::plus<value_type>(),
                                  detail::ThrustVectorDoDot<Matrix>()
            ); //very fast
#endif
}

template< class Matrix, class Vector>
inline void doSymv(  
              typename MatrixTraits<Matrix>::value_type alpha, 
              const Matrix& m,
              const Vector& x, 
              typename MatrixTraits<Matrix>::value_type beta, 
              Vector& y, 
              ThrustMatrixTag,
              ThrustVectorTag)
{
    dg::blas1::detail::doPointwiseDot( alpha, m, x, beta, y, ThrustVectorTag());
}

template< class Matrix, class Vector>
inline void doSymv(  
              Matrix& m, 
              const Vector& x,
              Vector& y, 
              ThrustMatrixTag,
              ThrustVectorTag,
              ThrustVectorTag)
{
    dg::blas1::detail::doPointwiseDot( m,x,y, dg::ThrustVectorTag());
}


}//namespace detail
    ///@endcond
} //namespace blas2
} //namespace dg
#endif //_DG_BLAS_PRECONDITIONER_
