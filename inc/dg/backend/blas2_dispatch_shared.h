#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include "matrix_categories.h"
#include "blas1_dispatch_shared.h" //load thrust_vector BLAS1 routines
#include "vector_categories.h"

///@cond
namespace dg{
namespace blas1{
//forward declare used blas1 functions
template<class from_ContainerType, class to_ContainerType>
inline void transfer( const from_ContainerType& source, to_ContainerType& target);

template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y);
}//namespace blas1
namespace blas2{
namespace detail{

//thrust vector preconditioner
template< class Vector1, class Vector2>
void doTransfer( const Vector1& in, Vector2& out, AnyVectorTag, AnyVectorTag)
{
    dg::blas1::transfer(in,out);
}

template< class Vector1, class Matrix, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag, SharedVectorTag)
{
    static_assert( std::is_same<get_value_type<Vector1>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_same<get_value_type<Matrix>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( std::is_same<get_value_type<Vector2>, double>::value, "We only support double precision dot products at the moment!");
    static_assert( all_true<
            std::is_base_of<SharedVectorTag, get_tensor_category<Vector1>>::value,
            std::is_base_of<SharedVectorTag, get_tensor_category<Vector2>>::value>::value,
        "All container types must share the same data layout (SharedVectorTag in this case)!");
    static_assert( std::is_same<get_execution_policy<Vector1>, get_execution_policy<Vector2> >::value &&
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
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag, RecursiveVectorTag)
{
    static_assert( std::is_base_of<RecursiveVectorTag,
        get_tensor_category<Vector2>>::value,
        "All data layouts must derive from the same vector category (RecursiveVectorTag in this case)!");
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    using inner_container1 = typename std::decay<Vector1>::type::value_type;
    std::vector<std::vector<int64_t>> acc( x.size());
    for( unsigned i=0; i<x.size(); i++)
        acc[i] = doDot_superacc( x[i], m, y[i],
                       get_tensor_category<Matrix>(),
                       get_tensor_category<inner_container1>() );
    for( unsigned i=1; i<x.size(); i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return acc[0];
}

template< class Vector1, class Matrix, class Vector2>
inline get_value_type<Vector1> doDot( const Vector1& x, const Matrix& m, const Vector2& y, SharedVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,m,y,SharedVectorTag(), get_tensor_category<Vector1>());
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
              SharedVectorTag, SharedVectorTag)
{
    dg::blas1::pointwiseDot( alpha, m, x, beta, y);
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              SharedVectorTag, SharedVectorTag)
{
    dg::blas1::pointwiseDot( 1, m, x, 0, y);
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              const Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SharedVectorTag, RecursiveVectorTag)
{
    for(unsigned i=0; i<x.size(); i++)
        dg::blas1::pointwiseDot( alpha, m, x[i], beta, y[i]);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              SharedVectorTag, RecursiveVectorTag)
{
    for(unsigned i=0; i<x.size(); i++)
        dg::blas1::pointwiseDot( 1, m, x[i], 0, y[i]);
}


}//namespace detail
} //namespace blas2
} //namespace dg
///@endcond
#endif //_DG_BLAS_PRECONDITIONER_
