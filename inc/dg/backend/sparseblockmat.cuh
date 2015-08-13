#pragma once

#include <thrust/device_vector.h>
//#include <cusp/system/cuda/utils.h>
#include "sparseblockmat.h"

namespace dg
{
//mixed derivatives for jump terms missing
struct SparseBlockMatDevice
{
    typedef double value_type;
    SparseBlockMatDevice( const SparseBlockMat& src)
    {
        data = src.data;
        cols_idx = src.cols_idx, data_idx = src.data_idx;
        num_rows = src.num_rows, num_cols = src.num_cols, blocks_per_line = src.blocks_per_line;
        n = src.n, left = src.left, right = src.right;
    }
    
    typedef thrust::device_vector<double> DVec;
    typedef thrust::device_vector<int> IVec;
    void symv(const DVec& x, DVec& y) const;
    void launch_multiply_kernel(const DVec& x, DVec& y) const;
    
    DVec data;
    IVec cols_idx, data_idx; 
    int num_rows, num_cols, blocks_per_line;
    int n;
    int left, right;
};

///@cond
inline void SparseBlockMatDevice::symv( const DVec& x, DVec& y) const
{
    launch_multiply_kernel( x,y);
}


template <>
struct MatrixTraits<SparseBlockMatDevice>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const SparseBlockMatDevice>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
void SparseBlockMatDevice::launch_multiply_kernel( const DVec& x, DVec& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);
    int offset[blocks_per_line];
    for( int d=0; d<blocks_per_line; d++)
        offset[d] = cols_idx[blocks_per_line+d]-1;
if(right==1) //alle dx Ableitungen
{
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<n; k++)
    {
        double temp=0;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                    x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)];
        y[(s*num_rows+i)*n+k]=temp;
    }
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    for( int k=0; k<n; k++)
    {
        double temp=0;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp+=data[(d*n + k)*n+q]*x[((s*num_cols + i+offset[d])*n+q)];
        y[(s*num_rows+i)*n+k]=temp;
    }
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<n; k++)
    {
        double temp=0;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                    x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)];
        y[(s*num_rows+i)*n+k]=temp;
    }
    return;
} //if right==1



#pragma omp parallel for
    for( unsigned  i=0; i<y.size(); i++)
    {
        y[i] =0;
    }
    //std::cout << "Difference is " <<t.diff()<<"s\n";
#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + i)*n+k)*right+j;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                y[I] += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                    x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
    }

if(left > 1)
{
    for( int d=0; d<blocks_per_line; d++)
    {
#pragma omp parallel for collapse(2)
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    {
        int J = i+offset[d];
        for( int k=0; k<n; k++)
        for( int j=0; j<right; j++)
        {
            int I = ((s*num_rows + i)*n+k)*right+j;
            {
                for( int q=0; q<n; q++) //multiplication-loop
                    y[I] += data[ (d*n+k)*n+q]*x[((s*num_cols + J)*n+q)*right+j];
            }
            //double temp=0;
            //for( int d=0; d<blocks_per_line; d++)
            //    for( int q=0; q<n; q++) //multiplication-loop
            //        temp+=data[(d*n + k)*n+q]*
            //        x[((s*num_cols + i + offset[d])*n+q)*right+j];
            //y[((s*num_rows+i)*n+k)*right+j]=temp;
        }
    }
    }
}
else
{
    for( int d=0; d<blocks_per_line; d++)
    {
#pragma omp parallel for 
    for( int i=1; i<num_rows-1; i++)
    {
        int J = i+offset[d];
        for( int k=0; k<n; k++)
        for( int j=0; j<right; j++)
        {
            int I = (i*n+k)*right+j;
            for( int q=0; q<n; q++) //multiplication-loop
                y[I] += data[ (d*n+k)*n+q]*x[(J*n+q)*right+j];
        }
    }
    }
} //endif
#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + i)*n+k)*right+j;
        for( int d=0; d<blocks_per_line; d++)
        for( int q=0; q<n; q++) //multiplication-loop
            y[I] += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
    }
}
#else

// multiply kernel
 __global__ void ell_multiply_kernel(
         const double* data, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int n, 
         const int left, const int right, 
         const double* x, double *y
         )
{
    int size = left*num_rows*n*right;
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<size; row += grid_size)
    {
        int s=row/(n*num_rows*right), 
            i = (row/(right*n))%num_rows, 
            k = (row/right)%n, 
            j=row%right;
        y[row] = 0;
        //if( i==0||i==num_rows-1)
            for( int d=0; d<blocks_per_line; d++)
            {
                int B = data_idx[i*blocks_per_line+d];
                int J = cols_idx[i*blocks_per_line+d];
                //int B = d;
                //int J = (i+d-1)%num_cols;
                for( int q=0; q<n; q++) //multiplication-loop
                    y[row] += 
                        data[ (B*n + k)*n+q]* x[((s*num_cols + J)*n+q)*right+j];
            }
            //wird nicht schneller!
        //else
        //    for( int d=0; d<blocks_per_line; d++)
        //    {
        //        int B = data_idx[blocks_per_line+d];
        //        int J = cols_idx[blocks_per_line+d]+i-1;
        //        //int B = d;
        //        //int J = (i+d-1)%num_cols;
        //        for( int q=0; q<n; q++) //multiplication-loop
        //            y[row] += 
        //                data[ (B*n + k)*n+q]* x[((s*num_cols + J)*n+q)*right+j];
        //    }
    }

}

void SparseBlockMatDevice::launch_multiply_kernel( const DVec& x, DVec& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);
    //set up kernel parameters
    const size_t BLOCK_SIZE = 256; 
    const size_t size = left*right*num_rows*n;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);

    const double* data_ptr = thrust::raw_pointer_cast( &data[0]);
    const int* cols_ptr = thrust::raw_pointer_cast( &cols_idx[0]);
    const int* block_ptr = thrust::raw_pointer_cast( &data_idx[0]);
    const double* x_ptr = thrust::raw_pointer_cast( &x[0]);
    double* y_ptr = thrust::raw_pointer_cast( &y[0]);
    ell_multiply_kernel <<<NUM_BLOCKS, BLOCK_SIZE>>> ( 
        data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, left, right, x_ptr,y_ptr);
}
#endif
///@endcond


} //namespace dg
