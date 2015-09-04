#pragma once

#include <thrust/device_vector.h>
//#include <cusp/system/cuda/utils.h>
#include "sparseblockmat.h"

namespace dg
{

/**
* @brief Ell Sparse Block Matrix format device version
*
* @ingroup lowlevel
* This class holds a copy of a EllSparseBlockMat on the device, which may 
be gpu or omp depending on the THRUST_DEVICE_SYSTEM macro. It can be applied
to device vectors and does the same thing as the host version
*/
struct EllSparseBlockMatDevice
{
    /**
    * @brief Allocate storage
    *
    * A device matrix has to be constructed from a host matrix. It simply
        copies all internal data of the host matrix to the device
        @param src  source on the host
    */
    EllSparseBlockMatDevice( const EllSparseBlockMat& src)
    {
        data = src.data;
        cols_idx = src.cols_idx, data_idx = src.data_idx;
        num_rows = src.num_rows, num_cols = src.num_cols, blocks_per_line = src.blocks_per_line;
        n = src.n, left = src.left, right = src.right;
    }
    
    /**
    * @brief Apply the matrix to a vector
    *
    * @param x input
    * @param y output may not equal input
    */
    void symv(const thrust::device_vector<double>& x, thrust::device_vector<double>& y) const;
    private:
    typedef thrust::device_vector<double> DVec;
    typedef thrust::device_vector<int> IVec;
    void launch_multiply_kernel(const DVec& x, DVec& y) const;
    
    DVec data;
    IVec cols_idx, data_idx; 
    int num_rows, num_cols, blocks_per_line;
    int n;
    int left, right;
};

/**
* @brief Coo Sparse Block Matrix format device version
*
* @ingroup lowlevel
* This class holds a copy of a CooSparseBlockMat on the device, which may 
be gpu or omp depending on the THRUST_DEVICE_SYSTEM macro. It does the same thing as the host version with the difference that it applies to device vectors.
*/
struct CooSparseBlockMatDevice
{
    /**
    * @brief Allocate storage
    *
    * A device matrix has to be constructed from a host matrix. It simply
        copies all internal data of the host matrix to the device
        @param src  source on the host
    */
    CooSparseBlockMatDevice( const CooSparseBlockMat& src)
    {
        data = src.data;
        rows_idx = src.rows_idx, cols_idx = src.cols_idx, data_idx = src.data_idx;
        num_rows = src.num_rows, num_cols = src.num_cols, num_entries = src.num_entries;
        n = src.n, left = src.left, right = src.right;
    }
    
    /**
    * @brief Apply the matrix to a vector
    *
    * @param alpha multiplies input
    * @param x input
    * @param beta premultiplies output
    * @param y output may not equal input
    */
    void symv(double alpha, const thrust::device_vector<double>& x, double beta, thrust::device_vector<double>& y) const;
    private:
    typedef thrust::device_vector<double> DVec;
    typedef thrust::device_vector<int> IVec;
    void launch_multiply_kernel(double alpha, const DVec& x, double beta, DVec& y) const;
    
    DVec data;
    IVec cols_idx, rows_idx, data_idx; 
    int num_rows, num_cols, num_entries;
    int n, left, right;
};

///@cond
inline void EllSparseBlockMatDevice::symv( const DVec& x, DVec& y) const
{
    launch_multiply_kernel( x,y);
}
inline void CooSparseBlockMatDevice::symv( double alpha, const DVec& x, double beta, DVec& y) const
{
    launch_multiply_kernel(alpha, x, beta, y);
}


template <>
struct MatrixTraits<EllSparseBlockMatDevice>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const EllSparseBlockMatDevice>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<CooSparseBlockMatDevice>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const CooSparseBlockMatDevice>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
void EllSparseBlockMatDevice::launch_multiply_kernel( const DVec& x, DVec& y) const
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
void CooSparseBlockMatDevice::launch_multiply_kernel( double alpha, const DVec& x, double beta, DVec& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);
    assert( beta == 1);

#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=0; i<num_entries; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + rows_idx[i])*n+k)*right+j;
        double temp=0;
        for( int q=0; q<n; q++) //multiplication-loop
            temp+= data[ (data_idx[i]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i])*n+q)*right+j];
        y[I] += alpha*temp;
    }
}
#else

// multiply kernel
 __global__ void ell_multiply_kernel(
         const double* data, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int n, const int size,
         const int left, const int right, 
         const double* x, double *y
         )
{
    //int size = left*num_rows*n*right;
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<size; row += grid_size)
    {
        int rr = row/right, rrn = rr/n;
        int s=rrn/num_rows, 
            i = (rrn)%num_rows, 
            k = (rr)%n, 
            j=row%right;
        int B, J;
        double temp=0;
        //y[row] = 0;
        //if( i==0||i==num_rows-1)
            for( int d=0; d<blocks_per_line; d++)
            {
                B = (data_idx[i*blocks_per_line+d]*n+k)*n;
                J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
                for( int q=0; q<n; q++) //multiplication-loop
                    temp +=data[ B+q]* x[(J+q)*right+j];
                    //y[row] +=data[ B+q]* x[(J+q)*right+j];
                y[row]=temp;
            }
            //wird nicht schneller!
        //else
        //    for( int d=0; d<blocks_per_line; d++)
        //    {
        //        B = (data_idx[blocks_per_line+d]*n+k)*n;
        //        J = (s*num_cols+cols_idx[blocks_per_line+d])*n;
        //        for( int q=0; q<n; q++) //multiplication-loop
        //            //y[row] +=data[ B+q]* x[(J+q)*right+j];
        //            temp +=data[ B+q]* x[(J+q)*right+j];
        //        y[row]=temp;
        //    }
    }

}
// multiply kernel
 __global__ void coo_multiply_kernel(
         const double* data, const int* rows_idx, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols, const int num_entries,
         const int n, 
         const int left, const int right, 
         double alpha, const double* x, double *y
         )
{
    int size = left*num_entries*n*right;
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int idx = thread_id; idx<size; idx += grid_size)
    {
        int s=idx/(n*num_entries*right), 
            i = (idx/(right*n))%num_entries, 
            k = (idx/right)%n, 
            j=idx%right;
        int I = ((s*num_rows+rows_idx[i])*n+k)*right+j;
        double temp = 0;
        int B = data_idx[i];
        int J = cols_idx[i];
        for( int q=0; q<n; q++) //multiplication-loop
            temp += data[ (B*n + k)*n+q]* x[((s*num_cols + J)*n+q)*right+j];
        y[I] += alpha*temp;
    }

}

void EllSparseBlockMatDevice::launch_multiply_kernel( const DVec& x, DVec& y) const
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
        data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, size, left, right, x_ptr,y_ptr);
}
void CooSparseBlockMatDevice::launch_multiply_kernel( double alpha, const DVec& x, double beta, DVec& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);
    assert( beta == 1);
    //set up kernel parameters
    const size_t BLOCK_SIZE = 256; 
    const size_t size = left*right*num_entries*n;
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);

    const double* data_ptr = thrust::raw_pointer_cast( &data[0]);
    const int* rows_ptr = thrust::raw_pointer_cast( &rows_idx[0]);
    const int* cols_ptr = thrust::raw_pointer_cast( &cols_idx[0]);
    const int* block_ptr = thrust::raw_pointer_cast( &data_idx[0]);
    const double* x_ptr = thrust::raw_pointer_cast( &x[0]);
    double* y_ptr = thrust::raw_pointer_cast( &y[0]);
    coo_multiply_kernel <<<NUM_BLOCKS, BLOCK_SIZE>>> ( 
        data_ptr, rows_ptr, cols_ptr, block_ptr, num_rows, num_cols, num_entries, n, left, right, alpha, x_ptr,y_ptr);
}
#endif
///@endcond


} //namespace dg
