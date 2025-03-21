#pragma once
namespace dg
{
namespace detail
{
// !!! Do not edit by hand:
// 1. copy CSRCache_cpu and spmv_cpu_kernel from sparsematrix_cpu.h
// 2. uncomment the pragma omp parallel
// 3. Rename cpu to omp

struct CSRCache_omp
{
    void forget(){}
};
//y = alpha A*x + beta y
template<class I, class V, class value_type, class C1, class C2>
void spmv_omp_kernel(
    CSRCache_omp& cache,
    size_t A_num_rows, size_t A_num_cols, size_t A_nnz,
    const I* RESTRICT A_pos , const I* RESTRICT A_idx, const V* RESTRICT  A_val,
    value_type alpha, value_type beta, const C1* RESTRICT x_ptr, C2* RESTRICT y_ptr
)
{
    if( beta == value_type(1))
    {
    #pragma omp for nowait
        for(int i = 0; i < (int)A_num_rows; i++)
        {
            for (int jj = A_pos[i]; jj < A_pos[i+1]; jj++)
            {
                int j = A_idx[jj];
                y_ptr[i] = DG_FMA( alpha*A_val[jj], x_ptr[j], y_ptr[i]);
            }
        }
    }
    else
    {
    #pragma omp for nowait
        for(int i = 0; i < (int)A_num_rows; i++)
        {
            value_type temp = 0;
            for (int jj = A_pos[i]; jj < A_pos[i+1]; jj++)
            {
                int j = A_idx[jj];
                temp = DG_FMA( alpha*A_val[jj], x_ptr[j], temp);
            }

            y_ptr[i] = DG_FMA( beta, y_ptr[i], temp);
        }
    }
}

}//namespace detail
}//namespace dg
