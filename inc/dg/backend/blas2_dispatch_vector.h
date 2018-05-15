#pragma once

#include "blas1_dispatch_vector.h"
#include "matrix_categories.h"

///@cond
namespace dg
{
namespace blas2
{
namespace detail
{


template< class Vector1, class Matrix, class Vector2>
inline get_value_type<Matrix> doDot(
              const Vector1& x,
              const Matrix& m,
              const Vector2& y,
              AnyMatrixTag,
              VectorVectorTag)
{
    static_assert( std::is_base_of<VectorVectorTag,
        get_data_layout<Vector2>>::value,
        "All data layouts must derive from the same vector category (VectorVectorTag in this case)!");
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    using inner_container1 = typename std::decay<Vector1>::type::value_type;
    std::vector<std::vector<int64_t>> acc( x.size());
    for( unsigned i=0; i<x.size(); i++)
        acc[i] = doDot_superacc( x[i], m, y[i],
                       get_data_layout<Matrix>(),
                       get_data_layout<inner_container1>() );
    for( unsigned i=1; i<x.size(); i++)
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
template< class Matrix, class Vector>
inline get_value_type<Matrix>  doDot(
              const Matrix& m,
              const Vector& y,
              AnyMatrixTag,
              VectorVectorTag)
{
    return doDot( y,m,y,AnyMatrixTag(),VectorVectorTag());
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              AnyMatrixTag,
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    //assert( m.size() == y.size() );
#endif //DG_DEBUG
    using inner_container1 = typename std::decay<Vector1>::type::value_type;
    for( unsigned i=0; i<x.size(); i++)
        doSymv( m, x[i], y[i],
                       get_data_layout<Matrix>(),
                       get_data_layout<inner_container1>());

}

template< class Precon, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Precon& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              AnyMatrixTag,
              VectorVectorTag,
              VectorVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    //assert( m.size() == y.size() );
#endif //DG_DEBUG
    using inner_container1 = typename std::decay<Vector1>::type::value_type;
    for( unsigned i=0; i<x.size(); i++)
        doSymv( alpha, m, x[i], beta, y[i],
                       get_data_layout<Precon>(),
                       get_data_layout<inner_container1>());
}


} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
