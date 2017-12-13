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
exblas::Superaccumulator doDot_dispatch( SerialTag, unsigned size, const double* x_ptr, const double * y_ptr, const double* z_ptr) {
    return exblas::Superaccumulator(  exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, 8,true)) ;
}
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
exblas::Superaccumulator doDot_dispatch( CudaTag, unsigned size, const double* x_ptr, const double * y_ptr, const double * z_ptr) {
    return exblas::Superaccumulator(  exblas::exdot_gpu( size, x_ptr,y_ptr)) ;
}
#else
exblas::Superaccumulator doDot_dispatch( OmpTag, unsigned size, const double* x_ptr, const double * y_ptr, const double* z_ptr) {
    return exblas::Superaccumulator(  exblas::exdot_omp( size, x_ptr,y_ptr,z_ptr, 8,true)) ;
}
#endif

template< class Matrix, class Vector>
exblas::Superaccumulator doDot_superacc( const Vector& x, const Matrix& m, const Vector& y, ThrustMatrixTag, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
    const double* x_ptr = thrust::raw_pointer_cast( x.data());
    const double* m_ptr = thrust::raw_pointer_cast( m.data());
    const double* y_ptr = thrust::raw_pointer_cast( y.data());
    return doDot_dispatch( get_execution_policy<Vector>(), x.size(), x_ptr, m_ptr, y_ptr);
}

template< class Matrix, class Vector>
inline get_value_type<Vector> doDot( const Vector& x, const Matrix& m, const Vector& y, dg::ThrustMatrixTag, dg::ThrustVectorTag)
{
    exblas::Superaccumulator acc = doDot_superacc( x,m,y,dg::ThrustMatrixTag(),dg::ThrustVectorTag());
    return acc.Round();
}
template< class Matrix, class Vector>
inline get_value_type<Vector> doDot( const Matrix& m, const Vector& x, dg::ThrustMatrixTag, dg::ThrustVectorTag)
{
    exblas::Superaccumulator acc = doDot_superacc( x,m,x,dg::ThrustMatrixTag(),dg::ThrustVectorTag());
    return acc.Round();
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
