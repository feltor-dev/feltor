#pragma once

#include <thrust/host_vector.h>
#include "thrust_vector.cuh"
#include "matrix_traits.h"

namespace dg
{
//mixed derivatives for jump terms missing
struct SparseBlockMat
{
    SparseBlockMat(){}
    SparseBlockMat( int num_block_rows, int num_block_cols, int num_blocks_per_line, int num_different_blocks, int n):
        data(num_different_blocks*n*n), cols_idx( num_block_rows*num_blocks_per_line), data_idx(cols_idx.size()),
        num_rows(num_block_rows), num_cols(num_block_cols), blocks_per_line(num_blocks_per_line),
        n(n),left(1), right(1){}
    //distribute the inner block to howmany processes
    /**
    * @brief Reduce a global matrix into equal chunks among mpi processes
    *
    * @param coord The mpi proces coordinate of the proper dimension
    * @param howmany[3] # of processes 0 is left, 1 is the middle, 2 is right
    */
    void distribute_rows( int coord, int howmany[3])
    {
        assert( num_rows == num_cols);
        int chunk_size = num_rows/howmany[1];
        SparseBlockMat temp(chunk_size, chunk_size+2, blocks_per_line, data.size()/(n*n), n);
        temp.left = left/howmany[0];
        temp.right = right/howmany[2];
        //first copy data elements even though not all might be needed it doesn't slow down things either
        for( unsigned  i=0; i<data.size(); i++)
            temp.data[i] = data[i];
        //now grab the right chunk of cols and data indices
        for( unsigned i=0; i<temp.cols_idx.size(); i++)
        {
            temp.cols_idx[i] = cols_idx[ coord*temp.num_rows+i];
            temp.data_idx[i] = data_idx[ coord*temp.num_rows+i];
            //data indices are correct but cols are still the global indices
            temp.cols_idx[i] = temp.cols_idx[i] - coord*chunk_size + n;
        }

    }
    
    typedef thrust::host_vector<double> HVec;
    typedef thrust::host_vector<int> IVec;
    void symv(const HVec& x, HVec& y) const;
    
    HVec data;
    IVec cols_idx, data_idx; 
    int num_rows, num_cols, blocks_per_line;
    int n, left, right;
};

void SparseBlockMat::symv(const HVec& x, HVec& y) const
{
    for( int s=0; s<left; s++)
    for( int i=0; i<num_rows; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        y[((s*num_rows + i)*n+k)*right+j] =0;
        for( int d=0; d<blocks_per_line; d++)
        for( int q=0; q<n; q++) //multiplication-loop
            y[((s*num_rows + i)*n+k)*right+j] += 
                data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
    }
}

template <>
struct MatrixTraits<SparseBlockMat>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const SparseBlockMat>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

} //namespace dg
