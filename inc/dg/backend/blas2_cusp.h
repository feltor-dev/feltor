#ifndef _DG_BLAS_LAPLACE_CUH
#define _DG_BLAS_LAPLACE_CUH

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <typeinfo>
#include <limits.h>
#include <cusp/multiply.h>
#include <cusp/convert.h>
#include <cusp/array1d.h>

#include "config.h"
#include "type_traits.h"

///@cond
namespace dg{
namespace blas2{
namespace detail{

template<class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& x, Matrix2& y, CuspMatrixTag, CuspMatrixTag)
{
    cusp::convert(x,y);
}

//Dot not implemented for cusp (and not needed)


#ifdef _OPENMP
template< class Matrix, class value_type>
inline void doSymv_cusp_dispatch( Matrix& m,
                    const value_type* RESTRICT x_ptr,
                    value_type* RESTRICT y_ptr,
                    cusp::csr_format,
                    OmpTag)
{
    typedef typename Matrix::index_type index_type;
    const value_type* RESTRICT val_ptr = thrust::raw_pointer_cast( &m.values[0]);
    const index_type* RESTRICT row_ptr = thrust::raw_pointer_cast( &m.row_offsets[0]);
    const index_type* RESTRICT col_ptr = thrust::raw_pointer_cast( &m.column_indices[0]);
    int rows = m.num_rows;
    #pragma omp parallel for
    for(int i = 0; i < rows; i++)
    {
        value_type temp = 0.;
        for (index_type jj = row_ptr[i]; jj < row_ptr[i+1]; jj++)
        {
            index_type j = col_ptr[jj];
            temp += val_ptr[jj]*x_ptr[j];
        }

        y_ptr[i] = temp;
    }
}
#endif// _OPENMP

template< class Matrix, class value_type>
inline void doSymv_cusp_dispatch( Matrix& m,
                    const value_type* x_ptr,
                    value_type*  y_ptr,
                    cusp::sparse_format,
                    AnyPolicyTag)
{
    cusp::array1d_view< const value_type*> cx( x_ptr, x_ptr + m.num_cols);
    cusp::array1d_view<       value_type*> cy( y_ptr, y_ptr + m.num_rows);
    cusp::multiply( m, cx, cy);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix& m,
                    const Vector1&x,
                    Vector2& y,
                    CuspMatrixTag,
                    SharedVectorTag  )
{
    static_assert( std::is_base_of<SharedVectorTag, get_data_layout<Vector2>>::value,
        "All data layouts must derive from the same vector category (SharedVectorTag in this case)!");
    static_assert( std::is_same< get_execution_policy<Vector1>, get_execution_policy<Vector2> >::value, "Execution policies must be equal!");
    typedef typename Matrix::value_type value_type;
    static_assert( std::is_same< get_value_type<Vector1>, value_type >::value,
        "Value types must be equal"
    );
    static_assert( std::is_same< get_value_type<Vector2>, value_type >::value,
        "Value types must be equal"
    );

    const value_type* RESTRICT x_ptr = thrust::raw_pointer_cast( x.data());
    value_type* RESTRICT y_ptr = thrust::raw_pointer_cast( y.data());
#ifdef DG_DEBUG
    assert( m.num_rows == y.size() );
    assert( m.num_cols == x.size() );
#endif //DG_DEBUG
    doSymv_cusp_dispatch( m,x_ptr,y_ptr,
            typename Matrix::format(),
            get_execution_policy<Vector1>());
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix& m,
                    const Vector1&x,
                    Vector2& y,
                    CuspMatrixTag,
                    VectorVectorTag  )
{
    static_assert( std::is_base_of<VectorVectorTag, get_data_layout<Vector2>>::value,
        "All data layouts must derive from the same vector category (VectorVectorTag in this case)!");
#ifdef DG_DEBUG
    assert( m.num_rows == y.size() );
    assert( m.num_cols == x.size() );
#endif //DG_DEBUG
    using inner_container = typename std::decay<Vector1>::type::value_type;
    for ( unsigned i=0; i<x.size(); i++)
        doSymv( m, x[i], y[i], CuspMatrixTag(), get_data_layout<inner_container>());
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond

#endif //_DG_BLAS_LAPLACE_CUH
