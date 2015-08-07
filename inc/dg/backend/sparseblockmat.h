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
        data(num_different_blocks*n*n), cols_idx( num_rows*num_blocks_per_line), data_idx(cols_idx.size()),
        num_rows(num_block_rows), num_cols(num_block_cols), blocks_per_line(num_blocks_per_line),
        n(n),left(1), right(1){}
    
    typedef thrust::host_vector<double> HVec;
    typedef thrust::host_vector<int> IVec;
    void symv(const HVec& x, HVec& y) const;
    
    HVec data;
    IVec cols_idx, data_idx; 
    int num_rows, num_cols, blocks_per_line;
    int n, left, right;
    HVec norm; //the normalization vector
};

void SparseBlockMat::symv(const HVec& x, HVec& y) const
{
    for( unsigned s=0; s<left; s++)
    for( unsigned i=0; i<num_rows; i++)
    for( unsigned k=0; k<n; k++)
    for( unsigned j=0; j<right; j++)
    {
        y[((s*num_rows + i)*n+k)*right+j] =0;
        for( unsigned d=0; d<blocks_per_line; d++)
        for( unsigned q=0; q<n; q++) //multiplication-loop
            y[((s*num_rows + i)*n+k)*right+j] += 
                data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                x[((s*num_cols + col[i*blocks_per_line+d])*n+q)*right+j];
    }
        
    if( !norm.empty())
        dg::blas1::detail::doPointwiseDot( norm, y, y, ThrustVectorTag());
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
