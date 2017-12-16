#include <omp.h>
#include "vector_categories.h"

namespace dg{

// general multiply kernel
template<class value_type>
void ell_multiply_kernel( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int n, 
         const int left_size, const int right_size, 
         const int * RESTRICT right_range,
         const value_type * RESTRICT x, value_type * RESTRICT y
         )
{
#pragma omp for collapse(2) nowait
    for( int s=0; s<left_size; s++)
    for( int i=0; i<num_rows; i++)
    {
        int J[blocks_per_line];
        for( int d=0; d<blocks_per_line; d++)
            J[d] = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
        for( int k=0; k<n; k++)
        {
            int B[blocks_per_line];
            for( int d=0; d<blocks_per_line; d++)
                B[d] = (data_idx[i*blocks_per_line+d]*n+k)*n;
            for( int j=right_range[0]; j<right_range[1]; j++)
            {
                value_type temp = 0;
                for( int d=0; d<blocks_per_line; d++)
                    for( int q=0; q<n; q++) //multiplication-loop
                        temp += data[ B[d]+q]*
                            x[(J[d]+q)*right_size+j];
                int I = ((s*num_rows + i)*n+k)*right_size+j;
                y[I] = alpha*temp + beta*y[I];
            }
        }
    }
}
//specialized multiply kernel, as fast as it will ever be :) 
template<class value_type, int n, int blocks_per_line>
void ell_multiply_kernel( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols,
         const int left_size, const int right_size, 
         const int * RESTRICT right_range,
         const value_type * RESTRICT x, value_type * RESTRICT y
         )
{
    if(right_size==1)
    {
    //trivial means that the data blocks do not change among rows
    bool trivial = true; 
    for( int i=1; i<num_rows-1; i++)
        for( int d=0; d<blocks_per_line; d++)
        {
            if( data_idx[i*blocks_per_line+d] 
                    != data_idx[blocks_per_line+d]) trivial = false;
        }
    if(trivial)
    {
    value_type xprivate[blocks_per_line*n];
    value_type dprivate[blocks_per_line*n*n];
    for( int d=0; d<blocks_per_line; d++)
    for( int k=0; k<n; k++)
    for( int q=0; q<n; q++)
    {
        int B = data_idx[blocks_per_line+d];
        dprivate[(k*blocks_per_line+d)*n+q] = data[(B*n+k)*n+q];
    }
    #pragma omp for nowait
    for( int s=0; s<left_size; s++)
    {
        for( int i=0; i<1; i++)
        {
            for( int d=0; d<blocks_per_line; d++)
            {
                int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
                for(int q=0; q<n; q++)
                    xprivate[d*n+q] = x[J+q];
            }
            for( int k=0; k<n; k++)
            {
                value_type temp = 0;
                for( int d=0; d<blocks_per_line; d++)
                {
                    int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
                    for( int q=0; q<n; q++) //multiplication-loop
                        temp += data[B+q]*xprivate[d*n+q];
                }
                int I = ((s*num_rows + i)*n+k);
                y[I] = alpha*temp + beta*y[I];
            }
        }
        #pragma omp SIMD //very important for KNL
        for( int i=1; i<num_rows-1; i++)
        {
            for( int d=0; d<blocks_per_line; d++)
            {
                int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
                for(int q=0; q<n; q++)
                    xprivate[d*n+q] = x[J+q];
            }
            for( int k=0; k<n; k++)
            {
                value_type temp = 0;
                int B = n*blocks_per_line*k;
                for( int d=0; d<blocks_per_line*n; d++)
                    temp += dprivate[B+d]*xprivate[d];
                int I = ((s*num_rows + i)*n+k);
                y[I] = alpha*temp + beta*y[I];
            }
        }
        for( int i=num_rows-1; i<num_rows; i++)
        {
            for( int d=0; d<blocks_per_line; d++)
            {
                int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
                for(int q=0; q<n; q++)
                    xprivate[d*n+q] = x[J+q];
            }
            for( int k=0; k<n; k++)
            {
                value_type temp = 0;
                for( int d=0; d<blocks_per_line; d++)
                {
                    int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
                    for( int q=0; q<n; q++) //multiplication-loop
                        temp += data[B+q]*xprivate[d*n+q];
                }
                int I = ((s*num_rows + i)*n+k);
                y[I] = alpha*temp + beta*y[I];
            }
        }
    }
    } //trivial
    else 
    {
    value_type xprivate[blocks_per_line*n];
    #pragma omp for nowait
    for( int s=0; s<left_size; s++)
    for( int i=0; i<num_rows; i++)
    {
        for( int d=0; d<blocks_per_line; d++)
        {
            int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
            for(int q=0; q<n; q++)
                xprivate[d*n+q] = x[J+q];
        }
        for( int k=0; k<n; k++)
        {
            value_type temp = 0;
            for( int d=0; d<blocks_per_line; d++)
            {
                int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
                for( int q=0; q<n; q++) //multiplication-loop
                    temp += data[B+q]*xprivate[d*n+q];
            }
            int I = ((s*num_rows + i)*n+k);
            y[I] = alpha*temp + beta*y[I];
        }
    }
    }// trivial
    }// right_size==1
    else
    {
    value_type dprivate[blocks_per_line*n];
#pragma omp for collapse(2) nowait
    for( int s=0; s<left_size; s++)
    for( int i=0; i<num_rows; i++)
    {
        int J[blocks_per_line];
        for( int d=0; d<blocks_per_line; d++)
            J[d] = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
        for( int k=0; k<n; k++)
        {
            for( int d=0; d<blocks_per_line; d++)
            {
                int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
                for(int q=0; q<n; q++)
                    dprivate[d*n+q] = data[B+q];
            }
#pragma omp SIMD //very important for KNL
            for( int j=right_range[0]; j<right_range[1]; j++)
            {
                value_type temp = 0;
                for( int d=0; d<blocks_per_line; d++)
                    for( int q=0; q<n; q++) //multiplication-loop
                        temp += dprivate[ d*n+q]*
                            x[(J[d]+q)*right_size+j];
                int I = ((s*num_rows + i)*n+k)*right_size+j;
                y[I] = alpha*temp + beta*y[I];
            }
        }
    }
    }
}

template<class value_type, int n>
void call_ell_multiply_kernel( value_type alpha, value_type beta,
         const value_type * RESTRICT data_ptr, const int * RESTRICT cols_ptr, const int * RESTRICT block_ptr, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int left_size, const int right_size, 
         const int * RESTRICT right_range_ptr,
         const value_type * RESTRICT x_ptr, value_type * RESTRICT y_ptr)
{
    if( blocks_per_line == 1)
        ell_multiply_kernel<value_type, n, 1>  (alpha, beta,  
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if (blocks_per_line == 2)
        ell_multiply_kernel<value_type, n, 2>  (alpha, beta,  
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if (blocks_per_line == 3)
        ell_multiply_kernel<value_type, n, 3>  (alpha, beta,  
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if (blocks_per_line == 4)
        ell_multiply_kernel<value_type, n, 4>  (alpha, beta,  
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else
        ell_multiply_kernel<value_type>  (alpha, beta,  
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
}


template<class value_type>
void EllSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const value_type* x_ptr, value_type beta, value_type* y_ptr) const
{
    const value_type* data_ptr = thrust::raw_pointer_cast( &data[0]);
    const int* cols_ptr = thrust::raw_pointer_cast( &cols_idx[0]);
    const int* block_ptr = thrust::raw_pointer_cast( &data_idx[0]);
    const int* right_range_ptr = thrust::raw_pointer_cast( &right_range[0]);
    if( n == 1)
        call_ell_multiply_kernel<value_type, 1>  (alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);

    else if( n == 2)
        call_ell_multiply_kernel<value_type, 2>  (alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if( n == 3)
        call_ell_multiply_kernel<value_type, 3>  (alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if( n == 4)
        call_ell_multiply_kernel<value_type, 4>  (alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if( n == 5)
        call_ell_multiply_kernel<value_type, 5>  (alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if( n == 6)
        call_ell_multiply_kernel<value_type, 6>  (alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else 
        ell_multiply_kernel<value_type> ( alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
}

template<class value_type>
void CooSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const value_type* x, value_type beta, value_type* y) const
{
    for( int i=0; i<num_entries; i++)
#pragma omp parallel for collapse(3)
    for( int s=0; s<left_size; s++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right_size; j++)
    {
        int I = ((s*num_rows + rows_idx[i])*n+k)*right_size+j;
        value_type temp=0;
        for( int q=0; q<n; q++) //multiplication-loop
            temp+= data[ (data_idx[i]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i])*n+q)*right_size+j];
        y[I] = alpha*temp + beta*y[I];
    }
}

}//namespace dg
