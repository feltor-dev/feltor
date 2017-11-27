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

template< class Matrix, class value_type>
inline exblas::Superacc doDot_dispatch( const thrust::host_vector<value_type>& x, const Matrix& m, const thrust::host_vector<value_type>& y, ThrustMatrixTag, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
    const double* x_ptr = thrust::raw_pointer_case( x.data());
    const double* m_ptr = thrust::raw_pointer_case( m.data());
    const double* y_ptr = thrust::raw_pointer_case( y.data());
    return exblas::Superaccumulator(  exblas::exdot_cpu( x.size(), x_ptr,m_ptr, y_ptr)) ;
}
template< class Matrix, class value_type>
inline exblas::Superacc doDot_dispatch( const thrust::device_vector<value_type>& x, const Matrix& m, const thrust::device_vector<value_type>& y, ThrustMatrixTag, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    const double* x_ptr = thrust::raw_pointer_case( x.data());
    const double* m_ptr = thrust::raw_pointer_case( m.data());
    const double* y_ptr = thrust::raw_pointer_case( y.data());
    return exblas::Superaccumulator(  exblas::exdot_omp( x.size(), x_ptr,m_ptr, y_ptr)) ;
#else
    const double* x_ptr = thrust::raw_pointer_case( x.data());
    const double* m_ptr = thrust::raw_pointer_case( m.data());
    const double* y_ptr = thrust::raw_pointer_case( y.data());
    return exblas::Superaccumulator(  exblas::exdot_gpu( x.size(), x_ptr,m_ptr, y_ptr)) ;
#endif
}

template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Vector& x, const Matrix& m, const Vector& y, dg::ThrustMatrixTag, dg::ThrustVectorTag)
{
    exblas::Superacc acc = doDot_dispatch( x,m,y,dg::ThrustMatrixTag(),dg::ThrustVectorTag());
}
template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::ThrustMatrixTag, dg::ThrustVectorTag)
{
    exblas::Superacc acc = doDot_dispatch( x,m,x,dg::ThrustMatrixTag(),dg::ThrustVectorTag());
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
