#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include "matrix_categories.h"
#include "blas1_dispatch_shared.h" //load thrust_vector BLAS1 routines
#include "vector_categories.h"

namespace dg{
namespace blas2{
    ///@cond
namespace detail{

//thrust vector preconditioner
template< class Vector1, class Vector2>
void doTransfer( const Vector1& in, Vector2& out, AnyVectorTag, AnyVectorTag)
{
    dg::blas1::transfer(in,out);
}

std::vector<int64_t> doDot_dispatch( SerialTag, unsigned size, const double* x_ptr, const double * y_ptr, const double* z_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, &h_superacc[0]) ;
    return h_superacc;
}

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
std::vector<int64_t> doDot_dispatch( CudaTag, unsigned size, const double* x_ptr, const double * y_ptr, const double * z_ptr) {
    static thrust::device_vector<int64_t> d_superacc(exblas::BIN_COUNT);
    int64_t * d_ptr = thrust::raw_pointer_cast( d_superacc.data());
    exblas::exdot_gpu( size, x_ptr,y_ptr,z_ptr, d_ptr);
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    cudaMemcpy( &h_superacc[0], d_ptr, exblas::BIN_COUNT*sizeof(int64_t), cudaMemcpyDeviceToHost);
    return h_superacc;
}
#else
std::vector<int64_t> doDot_dispatch( OmpTag, unsigned size, const double* x_ptr, const double * y_ptr, const double* z_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    if(size<dg::blas1::detail::MIN_SIZE)
        exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, &h_superacc[0]);
    else
        exblas::exdot_omp( size, x_ptr,y_ptr,z_ptr, &h_superacc[0]);
    return h_superacc;
}
#endif

template< class Vector1, class Matrix, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag, SharedVectorTag, SharedVectorTag)
{
    static_assert( std::is_base_of<SharedVectorTag,
        get_data_layout<Vector2>>::value,
        "All container types must share the same vector category (SharedVectorTag in this case)!");
    static_assert( std::is_same<get_execution_policy<Vector>,
        get_execution_policy<Vector2> >::value,
        "All container types must share the same execution policy!");
    static_assert( std::is_base_of<SharedVectorTag,
        get_data_layout<Matrix>>::value,
        "All container types must share the same vector category (SharedVectorTag in this case)!");
    static_assert( std::is_same<get_execution_policy<Vector1>,
        get_execution_policy<Matrix> >::value,
        "All container types must share the same execution policy!");
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
    const double* x_ptr = thrust::raw_pointer_cast( x.data());
    const double* m_ptr = thrust::raw_pointer_cast( m.data());
    const double* y_ptr = thrust::raw_pointer_cast( y.data());
    return doDot_dispatch( get_execution_policy<Vector1>(), x.size(), x_ptr, m_ptr, y_ptr);
}

template< class Matrix, class Vector>
inline get_value_type<Vector> doDot( const Vector& x, const Matrix& m, const Vector& y, SharedVectorTag, SharedVectorTag, SharedVectorTag)
{
    static_assert( std::is_same<get_value_type<Vector>, double>::value, "We only support double precision dot products at the moment!");
    std::vector<int64_t> acc = doDot_superacc( x,m,y,SharedVectorTag(),SharedVectorTag());
    return exblas::cpu::Round(acc.data());
}
template< class Matrix, class Vector>
inline get_value_type<Vector> doDot( const Matrix& m, const Vector& x, SharedVectorTag, SharedVectorTag, SharedVectorTag)
{
    return doDot( x,m,x,SharedVectorTag(), SharedVectorTag(), SharedVectorTag());
}

template< class Matrix, class Vector, class Vector2>
inline void doSymv(
              get_value_type<Vector> alpha,
              const Matrix& m,
              const Vector1& x,
              get_value_type<Vector> beta,
              Vector2& y,
              AnyVectorTag,
              AnyVectorTag,
              AnyVectorTag)
{
    dg::blas1::pointwiseDot( alpha, m, x, beta, y);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              AnyVectorTag,
              AnyVectorTag,
              AnyVectorTag)
{
    dg::blas1::pointwiseDot( 1., m,x,0., y);
}


}//namespace detail
    ///@endcond
} //namespace blas2
} //namespace dg
#endif //_DG_BLAS_PRECONDITIONER_
