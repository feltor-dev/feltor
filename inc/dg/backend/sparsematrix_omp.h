#pragma once
namespace dg
{
namespace detail
{
// !!! Do not edit by hand:
// 1. copy spmv_cpu_kernel from sparsematrix_cpu.h
// 2. uncomment the pragma omp parallel
// 3. Rename cpu to omp
//y = alpha A*x + beta y
template<class I, class V, class value_type, class C1, class C2>
void spmv_omp_kernel(
    size_t A_num_rows,
    const I& A_pos , const I& A_idx, const V& A_val,
    value_type alpha, value_type beta, const C1* RESTRICT x_ptr, C2* RESTRICT y_ptr
)
{
    const auto* RESTRICT val_ptr = thrust::raw_pointer_cast( &A_val[0]);
    const auto* RESTRICT row_ptr = thrust::raw_pointer_cast( &A_pos[0]);
    const auto* RESTRICT col_ptr = thrust::raw_pointer_cast( &A_idx[0]);
    if( beta == value_type(1))
    {
    #pragma omp parallel for
        for(int i = 0; i < (int)A_num_rows; i++)
        {
            for (int jj = row_ptr[i]; jj < row_ptr[i+1]; jj++)
            {
                int j = col_ptr[jj];
                y_ptr[i] = DG_FMA( alpha*val_ptr[jj], x_ptr[j], y_ptr[i]);
            }
        }
    }
    else
    {
    #pragma omp parallel for
        for(int i = 0; i < (int)A_num_rows; i++)
        {
            value_type temp = beta*y_ptr[i];
            for (int jj = row_ptr[i]; jj < row_ptr[i+1]; jj++)
            {
                int j = col_ptr[jj];
                temp = DG_FMA( val_ptr[jj], x_ptr[j], temp);
            }

            y_ptr[i] = DG_FMA( alpha, temp, y_ptr[i]);
        }
    }
}
}//namespace detail
}//namespace dg
