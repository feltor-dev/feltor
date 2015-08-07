#pragma once

#include <thrust/device_vector.h>
#include <cusp/system/cuda/utils.h>
#include "sparseblockmat.h"

namespace dg
{
//mixed derivatives for jump terms missing
struct SparseBlockMatGPU
{
    SparseBlockMatGPU( const SparseBlockMat& src)
    {
        unsigned size=src.diag.size();
        diag.resize(size), row.resize(size), col.resize(size), block.resize(size);
        for( unsigned i=0; i<src.diag.size(); i++)
        {
            diag[i]=src.diag[i], row[i]=src.row[i], col[i]=src.col[i], block[i]=src.block[i];
        }
        n=src.n;
        num_rows=src.num_rows, num_cols=src.num_cols;
        num_blocks=src.num_blocks;
        left = src.left, right=src.right;
        norm=src.norm;
    }

    typedef thrust::device_vector<double> DVec;
    typedef thrust::device_vector<int> IVec;
    void symv(const DVec& x, DVec& y) const
    {
        //dg::Timer t;
        //t.tic();
        dg::blas1::detail::doScal(y,0,ThrustVectorTag());
        //t.toc();
        //std::cout << "Scal took "<<t.diff()<<"s\n";
        for( unsigned m=0; m<diag.size(); m++)
        {
            //t.tic();
            multiply_diagonal_launcher( x,y,m);
            //t.toc();
            //std::cout << "Diagonal "<<m<<" took "<<t.diff()<<"s\n";
        }
        if( !norm.empty())
            dg::blas1::detail::doPointwiseDot( norm, y, y, ThrustVectorTag());
    }
    void multiply_diagonal_launcher( const DVec& x, DVec& y, unsigned m) const;
    
    std::vector<DVec> diag;
    std::vector<IVec> row, col, block; //column is the vector idx, and block is an index into the diag array determine how to traverse the matrix
    std::vector<unsigned> n; //Block size (nxn blocks)
    std::vector<unsigned> num_rows, num_cols, num_blocks;//number of entries
    std::vector<unsigned> left, right; //size of the one 
    DVec norm; //the normalization vector
};

template <>
struct MatrixTraits<SparseBlockMatGPU>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const SparseBlockMatGPU>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@cond

//diagonal multiply kernel
 __global__ void multiply_diagonal(
         const unsigned n, 
         const unsigned num_blocks, const unsigned num_rows, const unsigned num_cols,
         const unsigned left, const unsigned right, 
         const double* diag, const double* x, double *y,
         const int* row, const int* col, const int* block
         )
{
    int size = left*num_blocks*n*right;
    const int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    //const int thread_idI = blockIdx.y;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int idx = thread_idx; idx<size; idx += grid_size)
    {
        int s=idx/(n*num_blocks*right),  i = (idx/(right*n))%num_blocks, k = (idx/right)%n, j=idx%right;
        for( unsigned int q=0; q<n; q++) //multiplication-loop
            y[((s*num_rows + row[i])*n+k)*right+j] += 
                diag[(block[i]*n+k)*n+q]*x[((s*num_cols + col[i])*n+q)*right+j];
        //for( unsigned q=0; q<n; q++) //multiplication-loop
            //y[((s*num_rows + i)*n+k)*right+j] += 
                //diag[(k)*n+q]*x[((s*num_cols + i)*n+q)*right+j];
    }

}

void SparseBlockMatGPU::multiply_diagonal_launcher( const DVec& x, DVec& y, unsigned m) const
{
    assert( x.size() == y.size());
    //set up kernel parameters
    const size_t BLOCK_SIZE = 256; //a multiple of n = 2,3,4,5 
    const size_t size = left[m]*right[m]*num_blocks[m]*n[m];
    const size_t NUM_BLOCKS = std::min<size_t>(size/BLOCK_SIZE+1, 65000);

    const double* diag_ptr = thrust::raw_pointer_cast( &diag[m][0]);
    const double* x_ptr = thrust::raw_pointer_cast( &x[0]);
    double* y_ptr = thrust::raw_pointer_cast( &y[0]);
    const int* row_ptr = thrust::raw_pointer_cast( &row[m][0]);
    const int* col_ptr = thrust::raw_pointer_cast( &col[m][0]);
    const int* block_ptr = thrust::raw_pointer_cast( &block[m][0]);
    multiply_diagonal <<<NUM_BLOCKS, BLOCK_SIZE>>> ( 
        n[m], num_blocks[m], num_rows[m], num_cols[m], left[m], right[m], 
        diag_ptr, x_ptr, y_ptr, row_ptr, col_ptr, block_ptr);
}

struct TestClass
{

    TestClass( const SparseBlockMat& src)
    {
    //assume src is symmetric and periodic
        unsigned size=src.diag.size();
        diag.resize(size);
        for( unsigned i=0; i<src.diag.size(); i++)
        {
            diag[i]=src.diag[i];
        }
        n=src.n[0];
        num=src.num_rows[0];
        left = src.left[0], right=src.right[0];
        norm=src.norm;
    }

    typedef thrust::device_vector<double> DVec;
    typedef thrust::device_vector<int> IVec;
    void symv(const DVec& x, DVec& y) const
    {
        //dg::Timer t;
        //t.tic();
        dg::blas1::detail::doScal(y,0,ThrustVectorTag());
        //t.toc();
        //std::cout << "Scal took "<<t.diff()<<"s\n";
        for( unsigned m=0; m<diag.size(); m++)
        {
            //t.tic();
            multiply_diagonal_launcher( x,y,m);
            //t.toc();
            //std::cout << "Diagonal "<<m<<" took "<<t.diff()<<"s\n";
            }
        if( !norm.empty())
            dg::blas1::detail::doPointwiseDot( norm, y, y, ThrustVectorTag());
    }
    void multiply_diagonal_launcher( const DVec& x, DVec& y, unsigned m) const;
    
    std::vector<DVec> diag;//three entries
    unsigned n; //Block size (nxn blocks)
    unsigned num;//number of rows
    unsigned left, right; //size of the one 
    DVec norm; //the normalization vector
};
//diagonal multiply kernel
 __global__ void multiply_periodic_diagonal(
         const unsigned n, const unsigned m,
         const unsigned num,
         const unsigned left, const unsigned right, 
         const double* diag, const double* x, double *y
         )
{
    int size = left*num*n*right;
    const int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    //const int thread_idI = blockIdx.y;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int idx = thread_idx; idx<size; idx += grid_size)
    {
        int s=idx/(n*num*right),  i = (idx/(right*n))%num, k = (idx/right)%n, j=idx%right;
        int col;
        switch( m)
        {
            case(0): col = i; break;
            case(1): col = (i+1)%num; break;
            case(2): col = (i-1+num)%num; break;
        }
        for( unsigned int q=0; q<n; q++) //multiplication-loop
            y[idx] += 
                diag[(m*n+k)*n+q]*x[((s*num + col)*n+q)*right+j];
    }

}
void TestClass::multiply_diagonal_launcher( const DVec& x, DVec& y, unsigned m) const
{
    assert( x.size() == y.size());
    //set up kernel parameters
    const size_t BLOCK_SIZE = 256; //a multiple of n = 2,3,4,5 
    const size_t size = left*right*num*n;
    const size_t NUM_BLOCKS = std::min<size_t>(size/BLOCK_SIZE+1, 65000);

    const double* diag_ptr = thrust::raw_pointer_cast( &diag[m][0]);
    const double* x_ptr = thrust::raw_pointer_cast( &x[0]);
    double* y_ptr = thrust::raw_pointer_cast( &y[0]);
    multiply_periodic_diagonal <<<NUM_BLOCKS, BLOCK_SIZE>>> ( 
        n, m, num, left, right, 
        diag_ptr, x_ptr, y_ptr);
}
template <>
struct MatrixTraits<TestClass>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const TestClass>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

///@endcond

} //namespace dg
