#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

namespace dg
{
namespace detail
{


// MW if ever needed the code for assembly of sparsity structure of A and the computation of A can be separated
// TODO Make gemm pointers restrict!?

// A = B*C
// We do not catch explicit zeros in A
// Entries in A are sorted (even if B and/or C are not)
template<class I, class V>
void spgemm_cpu_kernel(
    size_t B_num_rows, size_t B_num_cols, size_t C_num_cols,
    const I& B_pos , const I& B_idx, const V& B_val,
    const I& C_pos , const I& C_idx, const V& C_val,
          I& A_pos ,       I& A_idx,       V& A_val
)
{
    // We follow
    // https://commit.csail.mit.edu/papers/2018/kjolstad-18-workspaces.pdf
    // Sparse data structures do not support fast random inserts ...
    // That is why a dense workspace is inserted
    // ... a dense workspace turns sparse-sparse into sparse-dense iterations (fewer conditional checks)
    // Workspace have easier merge at the cost of managing the workspace and reduced temporal locality
    //
    // 1. Figure 10: assemble sparsity structure of A
    // (Theoretically this could be cached if we did the same multiplication multiple times)
    A_pos.resize( B_num_rows+1);
    A_pos[0] = 0;
    std::vector<bool> w( C_num_cols, false); // A dense work array
    // A_ij  = B_ik C_kj
    I wlist; // one row of A
    for( int i = 0; i<(int)B_num_rows; i++)
    {
        for( int pB = B_pos[i]; pB < B_pos[i+1]; pB++)
        {
            int k = B_idx[pB];
            for( int pC = C_pos[k]; pC < C_pos[k+1]; pC++)
            {
                int j = C_idx[pC];
                if( not w[j])
                {
                    wlist.push_back( j);
                    w[j] = true;
                }
            }
        }
        // Sort entries of A in each row
        std::sort( wlist.begin(), wlist.end());

        // Append wlist to A_idx
        A_idx.resize( A_pos[i] + wlist.size());
        A_pos[i+1] = A_pos[i] + wlist.size();
        for( int pwlist = 0; pwlist < (int)wlist.size(); pwlist ++)
        {
            int j = wlist[pwlist];
            A_idx[ A_pos[i] + pwlist ] = j;
            w[j] = false;
        }
        wlist.resize( 0);
    }
    A_val.resize( A_pos[B_num_rows]);
    // 2. Figure 1d) Do the actual multiplication

    // A dense workspace array
    V workspace( C_num_cols, 0);
    for (int i = 0; i < (int)B_num_rows; i++)
    {
        // Dense Workspace array
        for (int pB = B_pos[i]; pB < B_pos[i+1]; pB++)
        {
            int k = B_idx[pB];
            for (int pC = C_pos[k]; pC < C_pos[k+1]; pC++)
            {
                int j = C_idx[pC];
                workspace[j] += B_val[pB] * C_val[pC];
            }
        }
        // We have to know sparse structure of A...
        for (int pA = A_pos[i]; pA < A_pos[i+1]; pA++)
        {
            int j = A_idx[pA];
            A_val[pA] = workspace[j];
            workspace[j] = 0;
        }
    }
}

//A = B + C
// We do not catch explicit zeros in A
// Entries in A are sorted
template<class I, class V>
void spadd_cpu_kernel(
    size_t B_num_rows, size_t B_num_cols,
    const I& B_pos , const I& B_idx, const V& B_val,
    const I& C_pos , const I& C_idx, const V& C_val,
          I& A_pos ,       I& A_idx,       V& A_val // restrict !
)
{
    // We follow
    // https://commit.csail.mit.edu/papers/2018/kjolstad-18-workspaces.pdf
    //
    // 1. (No Figure) assemble sparsity structure of A
    // (Theoretically this could be cached if we did the same multiplication multiple times)
    A_pos.resize( B_num_rows+1);
    A_pos[0] = 0;
    std::vector<bool> w( B_num_cols, false); // A dense work array
    // A_ij  = B_ik C_kj
    I wlist; // one row of A
    for( int i = 0; i<(int)B_num_rows; i++)
    {
        // Dense Workspace array
        for (int pB = B_pos[i]; pB < B_pos[i+1]; pB++)
        {
            int ib = B_idx[pB];
            wlist.push_back(ib);
            w[ib] = true;
        }
        for (int pC = C_pos[i]; pC < C_pos[i+1]; pC++)
        {
            int ic = C_idx[pC];
            if( not w[ic])
            {
                wlist.push_back(ic);
                w[ic] = true;
            }
        }
        // Sort entries of A in each row
        std::sort( wlist.begin(), wlist.end());
        // Append wlist to A_idx
        A_idx.resize( A_pos[i] + wlist.size());
        A_pos[i+1] = A_pos[i] + wlist.size();
        for( int pwlist = 0; pwlist < (int)wlist.size(); pwlist ++)
        {
            int j = wlist[pwlist];
            A_idx[ A_pos[i] + pwlist ] = j;
            w[j] = false;
        }
        wlist.resize( 0);
    }
    A_val.resize( A_pos[B_num_rows]);
    //
    // 2. Figure 6) Sparse vector addition

    // A dense workspace array
    V workspace( B_num_cols, 0);
    for (int i = 0; i < (int)B_num_rows; i++)
    {
        // Dense Workspace array
        for (int pB = B_pos[i]; pB < B_pos[i+1]; pB++)
        {
            int ib = B_idx[pB];
            workspace[ib] = B_val[pB];
        }
        for (int pC = C_pos[i]; pC < C_pos[i+1]; pC++)
        {
            int ic = C_idx[pC];
            workspace[ic] += C_val[pC];
        }
        // We have to know sparse structure of A...
        for (int pA = A_pos[i]; pA < A_pos[i+1]; pA++)
        {
            int ia = A_idx[pA];
            A_val[pA] = workspace[ia];
            workspace[ia] = 0;
        }
    }
}

struct CSRCache_cpu
{
    void forget(){}
};
//y = alpha A*x + beta y
template<class I, class V, class value_type, class C1, class C2>
void spmv_cpu_kernel(
    CSRCache_cpu& cache,
    size_t A_num_rows, size_t A_num_cols, size_t A_nnz,
    const I* RESTRICT A_pos , const I* RESTRICT A_idx, const V* RESTRICT  A_val,
    value_type alpha, value_type beta, const C1* RESTRICT x_ptr, C2* RESTRICT y_ptr
)
{
    if( beta == value_type(1))
    {
//    #pragma omp for nowait
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
//    #pragma omp for nowait
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



}// namespace detail

} // namespace dg
