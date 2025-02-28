#ifndef _DG_BLAS_SPARSEBLOCKMAT_
#define _DG_BLAS_SPARSEBLOCKMAT_
#include "exceptions.h"
#include "tensor_traits.h"
#include "tensor_traits.h"
#include "sparseblockmat.h"
//
///@cond
namespace dg{
namespace blas2{
namespace detail{


template<class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& x, Matrix2& y, AnyMatrixTag, SparseBlockMatrixTag)
{
    y = (Matrix2)x; //try to invoke the explicit conversion construction
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SparseBlockMatrixTag,
              SharedVectorTag,
              AnyPolicyTag)
{
    using value_type = get_value_type<Vector1>;

    int size_x = x.size();
    int size_y = y.size();
    if( size_x != m.total_num_cols()) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<x.size()<<" Number of columns is "<<m.total_num_cols());
    }
    if( size_y != m.total_num_rows()) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<y.size()<<" Number of rows is "<<m.total_num_rows());
    }
    const value_type * x_ptr = thrust::raw_pointer_cast(x.data());
          value_type * y_ptr = thrust::raw_pointer_cast(y.data());
    m.symv( SharedVectorTag(), get_execution_policy<Vector1>(), alpha, x_ptr, beta, y_ptr);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SparseBlockMatrixTag,
              RecursiveVectorTag,
              AnyPolicyTag)
{
    for(unsigned i=0; i<x.size(); i++)
        doSymv_dispatch( alpha, std::forward<Matrix>(m), x[i], beta, y[i],
                SparseBlockMatrixTag(),
                get_tensor_category<typename Vector1::value_type>(),
                get_execution_policy<Vector1>());
}
#ifdef _OPENMP
template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SparseBlockMatrixTag,
              RecursiveVectorTag,
              OmpTag)
{
    if( !omp_in_parallel())
    {
        #pragma omp parallel
        {
            for(unsigned i=0; i<x.size(); i++)
                doSymv_dispatch( alpha, std::forward<Matrix>(m), x[i], beta, y[i],
                        SparseBlockMatrixTag(),
                        get_tensor_category<typename Vector1::value_type>(),
                        OmpTag());
        }
    }
    else
        for(unsigned i=0; i<x.size(); i++)
            doSymv_dispatch( alpha, std::forward<Matrix>(m), x[i], beta, y[i],
                    SparseBlockMatrixTag(),
                    get_tensor_category<typename Vector1::value_type>(),
                    OmpTag());
}
#endif//_OPENMP


template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SparseBlockMatrixTag)
{
    doSymv_dispatch(alpha, std::forward<Matrix>(m), x, beta, y,
            SparseBlockMatrixTag(),
            get_tensor_category<Vector1>(),
            get_execution_policy<Vector1>()
            );
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix&& m,
              const Vector1& x,
              Vector2& y,
              SparseBlockMatrixTag)
{
    doSymv( 1., std::forward<Matrix>(m), x, 0., y, SparseBlockMatrixTag());
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
//
#endif//_DG_BLAS_SPARSEBLOCKMAT_
