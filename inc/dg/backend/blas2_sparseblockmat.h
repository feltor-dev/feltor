#ifndef _DG_BLAS_SPARSEBLOCKMAT_
#define _DG_BLAS_SPARSEBLOCKMAT_
#include "exceptions.h"
#include "tensor_traits.h"
#include "tensor_traits.h"
#include "sparsematrix.h"
#include "sparseblockmat.h"
//
///@cond
namespace dg{
namespace blas2{
template< class Stencil, class ContainerType, class ...ContainerTypes>
inline void parallel_for( Stencil f, unsigned N, ContainerType&& x, ContainerTypes&&... xs);
namespace detail{

template<class T>
struct SCAL
{
    SCAL( T beta) : m_beta(beta){}
    DG_DEVICE void operator()( unsigned i, T* y) { y[i] *= m_beta;}
    private:
    T m_beta;
};


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
    if( (size_t)size_x != (size_t)m.total_num_cols()) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<x.size()<<" Number of columns is "<<m.total_num_cols());
    }
    if( (size_t)size_y != (size_t)m.total_num_rows()) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<y.size()<<" Number of rows is "<<m.total_num_rows());
    }
    // This happens sometimes in MPI
    if( m.total_num_rows() == 0) // no elements in y
        return;
    if( m.total_num_cols() == 0) // no elements in x
    {
        if ( beta == value_type(1))
            return;
        else
            dg::blas2::parallel_for( SCAL(beta), m.total_num_rows(),  y);

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
    static_assert( std::is_same_v<get_execution_policy<Vector1>,
                                  get_execution_policy<Vector2>>,
                                "Vector types must have same execution policy");
    static_assert( std::is_same_v<get_tensor_category<Vector1>,
                                  get_tensor_category<Vector2>>,
                                "Vector types must have same data layout");
    static_assert( std::is_same_v<get_execution_policy<Vector1>,
                                  get_execution_policy<Matrix>>,
                                "Vector types must have same execution policy");
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
/////////////////// STENCIL /////////////////////////////////////
// Only works for SparseMatrix, not SparseBlockMatrix
template<class Functor, class Matrix, class Vector1, class Vector2>
inline void doStencil(
                    Functor f,
                    Matrix&& m,
                    const Vector1&x,
                    Vector2& y,
                    SparseMatrixTag,
                    SharedVectorTag  )
{
    static_assert( std::is_base_of<SharedVectorTag, get_tensor_category<Vector2>>::value,
        "All data layouts must derive from the same vector category (SharedVectorTag in this case)!");
    static_assert( std::is_same< get_execution_policy<Vector1>, get_execution_policy<Vector2> >::value, "Execution policies must be equal!");
    using value_type = dg::get_value_type<Matrix>;
    static_assert( std::is_same< get_value_type<Vector1>, value_type >::value,
        "Value types must be equal"
    );
    static_assert( std::is_same< get_value_type<Vector2>, value_type >::value,
        "Value types must be equal"
    );

    if( x.size() != m.num_cols()) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<x.size()<<" Number of columns is "<<m.num_cols());
    }
    if( y.size() != m.num_rows()) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<y.size()<<" Number of rows is "<<m.num_rows());
    }
    dg::blas2::parallel_for( f, m.num_rows(), m.row_offsets(), m.column_indices(), m.values(), x, y);
}
template< class Functor, class Matrix, class Vector1, class Vector2>
inline void doStencil( get_value_type<Vector1> alpha,
                    Functor f,
                    Matrix&& m,
                    const Vector1&x,
                    get_value_type<Vector1> beta,
                    Vector2& y,
                    SparseMatrixTag,
                    RecursiveVectorTag  )
{
    static_assert( std::is_base_of<RecursiveVectorTag, get_tensor_category<Vector2>>::value,
        "All data layouts must derive from the same vector category (RecursiveVectorTag in this case)!");
    if( x.size() != m.num_cols()) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<x.size()<<" Number of columns is "<<m.num_cols());
    }
    if( y.size() != m.num_rows()) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<y.size()<<" Number of rows is "<<m.num_rows());
    }
    using inner_container = typename std::decay_t<Vector1>::value_type;
    for ( unsigned i=0; i<x.size(); i++)
        doStencil( f,std::forward<Matrix>(m),x[i],y[i], SparseMatrixTag(), get_tensor_category<inner_container>());
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
//
#endif//_DG_BLAS_SPARSEBLOCKMAT_
