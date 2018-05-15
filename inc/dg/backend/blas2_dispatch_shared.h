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

template< class Vector1, class Matrix, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag)
{
    static_assert( std::is_same<get_value_type<Vector1>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_same<get_value_type<Matrix>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_same<get_value_type<Vector2>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( all_true<
            std::is_base_of<SharedVectorTag, get_data_layout<Vector1>>::value,
            std::is_base_of<SharedVectorTag, get_data_layout<Vector2>>::value>::value,
        "All container types must share the same data layout (SharedVectorTag in this case)!");
    static_assert( all_true<
        std::is_same<get_execution_policy<Vector1>, get_execution_policy<Vector2> >::value,
        std::is_same<get_execution_policy<Vector1>, get_execution_policy<Matrix> >::value,
        "All container types must share the same execution policy!");
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
    const double* x_ptr = thrust::raw_pointer_cast( x.data());
    const double* m_ptr = thrust::raw_pointer_cast( m.data());
    const double* y_ptr = thrust::raw_pointer_cast( y.data());
    return dg::blas1::detail::doDot_dispatch( get_execution_policy<Vector1>(), x.size(), x_ptr, m_ptr, y_ptr);
}

template< class Vector1, class Matrix, class Vector2>
inline get_value_type<Vector> doDot( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,m,y,SharedVectorTag());
    return exblas::cpu::Round(acc.data());
}
template< class Matrix, class Vector>
inline get_value_type<Vector> doDot( const Matrix& m, const Vector& x, SharedVectorTag)
{
    return doDot( x,m,x,SharedVectorTag());
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              const Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              AnyVectorTag)
{
    dg::blas1::pointwiseDot( alpha, m, x, beta, y);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              AnyVectorTag)
{
    dg::blas1::pointwiseDot( 1., m,x,0., y);
}


}//namespace detail
    ///@endcond
} //namespace blas2
} //namespace dg
#endif //_DG_BLAS_PRECONDITIONER_
