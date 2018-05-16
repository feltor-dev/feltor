#ifndef _DG_BLAS_SELFMADE_
#define _DG_BLAS_SELFMADE_
#include "type_traits.h"
#include "type_traits.h"
#include "sparseblockmat.h"
#include "sparseblockmat.cuh"
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
              Matrix& m,
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
    if( size_x != m.num_cols()) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<x.size()<<" and not "<<m.num_cols());
    }
    if( size_y != m.num_rows()) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<y.size()<<" and not "<<m.num_rows());
    }
    const value_type * x_ptr = thrust::raw_pointer_cast(x.data());
          value_type * y_ptr = thrust::raw_pointer_cast(y.data());
    m.symv( SharedVectorTag(), get_execution_policy<Vector1>(), alpha, x_ptr, beta, y_ptr);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SparseBlockMatrixTag,
              VectorVectorTag,
              AnyPolicyTag)
{
    for(unsigned i=0; i<x.size(); i++)
        doSymv_dispatch( alpha, m, x[i], beta, y[i],
                SparseBlockMatrixTag(),
                get_data_layout<typename Vector1::value_type>(),
                get_execution_policy<Vector1>());
}
#ifdef _OPENMP
template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SparseBlockMatrixTag,
              VectorVectorTag,
              OmpTag)
{
    if( !omp_in_parallel())
    {
        #pragma omp parallel
        {
            for(unsigned i=0; i<x.size(); i++)
                doSymv_dispatch( alpha, m, x[i], beta, y[i],
                        SparseBlockMatrixTag(),
                        get_data_layout<typename Vector1::value_type>(),
                        OmpTag());
        }
    }
    else
        for(unsigned i=0; i<x.size(); i++)
            doSymv_dispatch( alpha, m, x[i], beta, y[i],
                    SparseBlockMatrixTag(),
                    get_data_layout<typename Vector1::value_type>(),
                    OmpTag());
}
#endif//_OPENMP


template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              SparseBlockMatrixTag)
{
    doSymv(alpha, x,beta,y
            SparseBlockMatrixTag(),
            get_data_layout<Vector1>(),
            get_execution_policy<Vector1>()
            );
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix& m,
              const Vector1& x,
              Vector2& y,
              SparseBlockMatrixTag)
{
    doSymv( 1., m, x, 0., y, SparseBlockMatrixTag());
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
//
#endif//_DG_BLAS_SELFMADE_
