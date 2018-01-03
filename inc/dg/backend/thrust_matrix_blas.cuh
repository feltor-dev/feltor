#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "matrix_categories.h"
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
std::vector<int64_t> doDot_dispatch( SerialTag, unsigned size, const double* x_ptr, const double * y_ptr, const double* z_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, &h_superacc[0]) ;
    return h_superacc;
}
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
std::vector<int64_t> doDot_dispatch( CudaTag, unsigned size, const double* x_ptr, const double * y_ptr, const double * z_ptr) {
    thrust::device_vector<int64_t> d_superacc(exblas::BIN_COUNT);
    int64_t * d_ptr = thrust::raw_pointer_cast( d_superacc.data());
    exblas::exdot_gpu( size, x_ptr,y_ptr,z_ptr, d_ptr);
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    cudaMemcpy( &h_superacc[0], d_ptr, exblas::BIN_COUNT*sizeof(int64_t), cudaMemcpyDeviceToHost);
    return h_superacc;
}
#else
std::vector<int64_t> doDot_dispatch( OmpTag, unsigned size, const double* x_ptr, const double * y_ptr, const double* z_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    if(size<MIN_SIZE) 
        exblas::exdot_cpu( size, x_ptr,y_ptr, &h_superacc[0]);
    else 
        exblas::exdot_omp( size, x_ptr,y_ptr, &h_superacc[0]);
    return h_superacc;
}
#endif

template< class Matrix, class Vector>
std::vector<int64_t> doDot_superacc( const Vector& x, const Matrix& m, const Vector& y, ThrustMatrixTag, ThrustVectorTag)
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
inline get_value_type<Vector> doDot( const Vector& x, const Matrix& m, const Vector& y, ThrustMatrixTag, ThrustVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,m,y,ThrustMatrixTag(),ThrustVectorTag());
    return exblas::Round(acc.data());
}
template< class Matrix, class Vector>
inline get_value_type<Vector> doDot( const Matrix& m, const Vector& x, ThrustMatrixTag, ThrustVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,m,x,ThrustMatrixTag(),ThrustVectorTag());
    return exblas::Round(acc.data());
}

template< class Matrix, class Vector>
inline void doSymv(  
              get_value_type<Vector> alpha, 
              const Matrix& m,
              const Vector& x, 
              get_value_type<Vector> beta, 
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
    dg::blas1::detail::doPointwiseDot( 1., m,x,0., y, ThrustVectorTag());
}


}//namespace detail
    ///@endcond
} //namespace blas2
} //namespace dg
#endif //_DG_BLAS_PRECONDITIONER_
