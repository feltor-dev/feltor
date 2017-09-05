#ifndef _DG_BLAS_LAPLACE_CUH
#define _DG_BLAS_LAPLACE_CUH

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <typeinfo>
#include <cusp/multiply.h>
#include <cusp/convert.h>
#include <cusp/array1d.h>

#include "matrix_categories.h"
#include "vector_categories.h"

///@cond
namespace dg{

namespace blas2
{
namespace detail
{
template<class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& x, Matrix2& y, CuspMatrixTag, CuspMatrixTag)
{
    cusp::convert(x,y);
}
template< class Matrix, class Vector>
inline void doSymv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    ThrustVectorTag,
                    ThrustVectorTag, cusp::csr_format  )
{
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    typedef typename Vector::value_type value_type;
    typedef typename Matrix::index_type index_type;
    const value_type* RESTRICT val_ptr = thrust::raw_pointer_cast( &m.values[0]);
    const value_type* RESTRICT x_ptr = thrust::raw_pointer_cast( &x[0]);
    value_type* RESTRICT y_ptr = thrust::raw_pointer_cast( &y[0]);
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
#else
    cusp::array1d_view< typename Vector::const_iterator> cx( x.cbegin(), x.cend());
    cusp::array1d_view< typename Vector::iterator> cy( y.begin(), y.end());
    cusp::multiply( m, cx, cy);
#endif
}

template< class Matrix, class Vector>
inline void doSymv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    ThrustVectorTag,
                    ThrustVectorTag, 
                    cusp::sparse_format  )
{
    cusp::array1d_view< typename Vector::const_iterator> cx( x.cbegin(), x.cend());
    cusp::array1d_view< typename Vector::iterator> cy( y.begin(), y.end());
    cusp::multiply( m, cx, cy);
}

template< class Matrix, class Vector>
inline void doSymv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    ThrustVectorTag,
                    ThrustVectorTag  )
{
#ifdef DG_DEBUG
    assert( m.num_rows == y.size() );
    assert( m.num_cols == x.size() );
#endif //DG_DEBUG
    doSymv( m,x,y, CuspMatrixTag(), ThrustVectorTag(), ThrustVectorTag(), typename Matrix::format());
}

template< class Matrix, class Vector>
inline void doSymv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    CuspVectorTag,
                    CuspVectorTag  )
{
#ifdef DG_DEBUG
    assert( m.num_rows == y.size() );
    assert( m.num_cols == x.size() );
#endif //DG_DEBUG
    cusp::multiply( m, x, y);
}

template< class Matrix, class Vector>
inline void doGemv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    ThrustVectorTag,
                    ThrustVectorTag  )
{
    doSymv( m, x, y, CuspMatrixTag(), ThrustVectorTag(), ThrustVectorTag());
}

template< class Matrix, class Vector>
inline void doGemv( Matrix& m, 
                    const Vector&x, 
                    Vector& y, 
                    CuspMatrixTag, 
                    CuspVectorTag,
                    CuspVectorTag  )
{
    doGemv( m,x,y,CuspMatrixTag(), CuspVectorTag(), CuspVectorTag());
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond

#endif //_DG_BLAS_LAPLACE_CUH
