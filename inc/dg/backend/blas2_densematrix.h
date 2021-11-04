#pragma once

#include "densematrix.h"
#include "densematrix_serial.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "densematrix_cuda.cuh"
#else
#include "densematrix_omp.h"
#endif

///@cond
namespace dg{
namespace blas2{
namespace detail{
//forward declare
template< class Matrix, class Vector1, class Vector2>
void doSymv(  get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              DenseMatrixTag);

template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              DenseMatrixTag,
              SharedVectorTag,
              AnyPolicyTag)
{
    using value_type = get_value_type<Vector1>;

    unsigned size_x = x.size();
    unsigned size_y = y.size();
    if( size_x != m.num_cols()) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<x.size()<<" Number of columns is "<<m.num_cols());
    }
    if( size_y != m[0]->size()) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<y.size()<<" Number of rows is "<<m[0]->size());
    }
          value_type * y_ptr = thrust::raw_pointer_cast(y.data());
    std::vector<const value_type *> m_ptr( m.num_cols(), nullptr);
    for( unsigned i=0; i<m.num_cols(); i++)
        m_ptr[i] = thrust::raw_pointer_cast( m[i]->data());
    doDenseSymv( get_execution_policy<Vector2>(), size_y, size_x,
            alpha, m_ptr, x, beta, y_ptr);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              DenseMatrixTag,
              RecursiveVectorTag,
              AnyPolicyTag)
{
    // Commute std::vector with DenseMatrix
    std::vector<DenseMatrix<typename
        std::decay_t<Matrix>::container_type::value_type>>
        mtx(m[0]->size(), m.num_cols());
    for( unsigned i=0; i<m.num_cols(); i++)
        for( unsigned k=0; k<m[0]->size(); k++)
            mtx[k][i] = &((*m[i])[k]);

    // x is a coefficients vector
    for(unsigned i=0; i<y.size(); i++)
        doSymv( alpha, mtx[i], x, beta, y[i],
                DenseMatrixTag());
}
#ifdef _OPENMP
template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              DenseMatrixTag,
              RecursiveVectorTag,
              OmpTag)
{
    // Commute std::vector with DenseMatrix
    std::vector<DenseMatrix<typename
        std::decay_t<Matrix>::container_type::value_type>>
        mtx(m[0]->size(), m.num_cols());
    for( unsigned i=0; i<m.num_cols(); i++)
        for( unsigned k=0; k<m[0]->size(); k++)
            mtx[k][i] = &((*m[i])[k]);
    if( !omp_in_parallel())
    {
        #pragma omp parallel
        {
            for(unsigned i=0; i<y.size(); i++)
            {
                doSymv( alpha, mtx[i], x, beta, y[i],
                        DenseMatrixTag());
            }
        }
    }
    else
        for(unsigned i=0; i<y.size(); i++)
            doSymv( alpha, mtx[i], x, beta, y[i],
                    DenseMatrixTag());
}
#endif//_OPENMP
#ifdef MPI_VERSION
template< class Matrix, class Vector1, class Vector2>
inline void doSymv_dispatch(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              DenseMatrixTag,
              MPIVectorTag,
              AnyPolicyTag)
{
    // Commute std::vector with DenseMatrix
    DenseMatrix<typename std::decay_t<Matrix>::container_type::container_type>
        mtx(m.num_cols());
    for( unsigned i=0; i<m.num_cols(); i++)
        mtx[i] = &(m[i]->data());

    doSymv( alpha, mtx, x, beta, y.data(), DenseMatrixTag());
}
#endif //MPI_VERSION


// symv will call this function first
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              DenseMatrixTag)
{
    static_assert( std::is_same<get_execution_policy<Matrix>,
                                get_execution_policy<Vector2>>::value,
                "Dense Matrix and output Vector type must have same execution policy");
    static_assert( std::is_same<get_tensor_category<
                            typename std::decay_t<Matrix>::container_type>,
                        get_tensor_category<Vector2>>::value,
                "Dense Matrix and output Vector type must have same data layout");
    static_assert( std::is_same<get_tensor_category<
                            typename std::decay_t<Matrix>::container_type>,
                        get_tensor_category<Vector2>>::value,
                "Dense Matrix and output Vector type must have same data layout");
    doSymv_dispatch(alpha, std::forward<Matrix>(m), x, beta, y,
            DenseMatrixTag(),
            get_tensor_category<Vector2>(),
            get_execution_policy<Vector2>()
            );
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix&& m,
              const Vector1& x,
              Vector2& y,
              DenseMatrixTag)
{
    doSymv( 1., std::forward<Matrix>(m), x, 0., y, SparseBlockMatrixTag());
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
