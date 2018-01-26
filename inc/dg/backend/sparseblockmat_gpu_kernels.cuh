#pragma once

namespace dg
{

// general multiply kernel
template<class value_type>
 __global__ void ell_multiply_kernel( value_type alpha, value_type beta,
         const value_type* __restrict__  data, const int* __restrict__  cols_idx, const int* __restrict__  data_idx,
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int n, const int size,
         const int right_size,
         const int* __restrict__  right_range,
         const value_type* __restrict__  x, value_type * __restrict__ y
         )
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int right_ = right_range[1]-right_range[0];
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<size; row += grid_size)
    {
        int rr = row/right_size, rrn = rr/n;
        int s=rrn/num_rows,
            i = (rrn)%num_rows,
            k = (rr)%n,
            j=right_range[0]+row%right_;
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
        {
            int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
            int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
            for( int q=0; q<n; q++) //multiplication-loop
                temp =fma( data[ B+q], x[(J+q)*right_size+j], temp);
        }
        y[row]*= beta;
        y[row] = fma( alpha, temp, y[row]);
    }

}


//specialized multiply kernel
template<class value_type, size_t n, size_t blocks_per_line>
 __global__ void ell_multiply_kernel(value_type alpha, value_type beta,
         const value_type* __restrict__  data, const int* __restrict__  cols_idx, const int* __restrict__  data_idx,
         const int num_rows, const int num_cols,
         const int size, const int right_size,
         const int* __restrict__  right_range,
         const value_type* __restrict__  x, value_type * __restrict__ y
         )
{
    //int size = left*num_rows*n*right;
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int right_ = right_range[1]-right_range[0];
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<size; row += grid_size)
    {
        if(right_size==1)
        {
            int rrn = row/n, k = row%n;
            int s=rrn/num_rows, i = (rrn)%num_rows;
            value_type temp=0;
            for( int d=0; d<blocks_per_line; d++)
            {
                int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
                int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
                for( int q=0; q<n; q++) //multiplication-loop
                    temp = fma( data[ B+q], x[(J+q)], temp);
            }
            y[row]*= beta;
            y[row] = fma( alpha, temp, y[row]);
        }
        else
        {
            int rr = row/right_size;
            int rrn = rr/n, k = rr%n;
            int s=rrn/num_rows, i = (rrn)%num_rows;
            int j=right_range[0]+row%right_;
            value_type temp = 0;
            for( int d=0; d<blocks_per_line; d++)
            {
                int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
                int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
                for( int q=0; q<n; q++) //multiplication-loop
                    temp = fma( data[ B+q], x[(J+q)*right_size+j], temp);
            }
            y[row]*= beta;
            y[row] = fma( alpha, temp, y[row]);
        }
    }
}

template<class value_type, size_t n>
void call_ell_multiply_kernel( value_type alpha, value_type beta,
         const value_type * RESTRICT data_ptr, const int * RESTRICT cols_ptr, const int * RESTRICT block_ptr,
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int left_size, const int right_size,
         const int * RESTRICT right_range_ptr,
         const value_type * RESTRICT x_ptr, value_type * RESTRICT y_ptr)
{
    //set up kernel parameters
    const size_t BLOCK_SIZE = 256;
    const size_t size = left_size*right_size*num_rows*n; //number of lines
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    //note that the following use size instead of left_size
    if( blocks_per_line == 1)
        ell_multiply_kernel<value_type, n, 1><<<NUM_BLOCKS, BLOCK_SIZE>>>  (alpha, beta,
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if (blocks_per_line == 2)
        ell_multiply_kernel<value_type, n, 2><<<NUM_BLOCKS, BLOCK_SIZE>>>  (alpha, beta,
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if (blocks_per_line == 3)
        ell_multiply_kernel<value_type, n, 3><<<NUM_BLOCKS, BLOCK_SIZE>>>  (alpha, beta,
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else if (blocks_per_line == 4)
        ell_multiply_kernel<value_type, n, 4><<<NUM_BLOCKS, BLOCK_SIZE>>>  (alpha, beta,
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, size, right_size, right_range_ptr,  x_ptr,y_ptr);
    else
        ell_multiply_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>  (alpha, beta,
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, size, right_size, right_range_ptr,  x_ptr,y_ptr);
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
    {
        //set up kernel parameters
        const size_t BLOCK_SIZE = 256;
        const size_t size = left_size*right_size*num_rows*n; //number of lines
        const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
        ell_multiply_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, beta,
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, size, right_size, right_range_ptr,  x_ptr,y_ptr);
    }
}

//////////////////// COO multiply kernel
template<class value_type>
 __global__ void coo_multiply_kernel(
         const value_type* __restrict__  data, const int* __restrict__  rows_idx, const int* __restrict__  cols_idx, const int* __restrict__  data_idx,
         const int num_rows, const int num_cols, const int entry,
         const int n,
         const int left, const int right,
         value_type alpha, const value_type* __restrict__  x, value_type beta, value_type * __restrict__ y
         )
{
    int size = left*n*right;
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int idx = thread_id; idx<size; idx += grid_size)
    {
        int s=idx/(n*right),
            k=(idx/right)%n,
            j=idx%right;
        int I = ((s*num_rows+rows_idx[entry])*n+k)*right+j;
        value_type temp = 0;
        int B = data_idx[entry];
        int J = cols_idx[entry];
        for( int q=0; q<n; q++) //multiplication-loop
            temp = fma( data[ (B*n + k)*n+q], x[((s*num_cols + J)*n+q)*right+j], temp);
        y[I] = fma( alpha, temp, y[I]);
    }

}

template<class value_type>
void CooSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const value_type* x_ptr, value_type beta, value_type* y_ptr) const
{
    //set up kernel parameters
    const size_t BLOCK_SIZE = 256;
    const size_t size = left_size*right_size*n;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);

    const value_type* data_ptr = thrust::raw_pointer_cast( data.data());
    const int* rows_ptr = thrust::raw_pointer_cast( rows_idx.data());
    const int* cols_ptr = thrust::raw_pointer_cast( cols_idx.data());
    const int* block_ptr = thrust::raw_pointer_cast( data_idx.data());
    for( int i=0; i<num_entries; i++)
    {
        coo_multiply_kernel<value_type> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
            data_ptr, rows_ptr, cols_ptr, block_ptr, num_rows, num_cols, i, n, left_size, right_size, alpha, x_ptr, beta, y_ptr);
    }
}

}//namespace dg


