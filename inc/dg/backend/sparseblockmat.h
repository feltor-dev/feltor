#pragma once

#include <thrust/host_vector.h>
#include "matrix_traits.h"

namespace dg
{

struct EllSparseBlockMat
{
    typedef double value_type;
    EllSparseBlockMat(){}
    EllSparseBlockMat( int num_block_rows, int num_block_cols, int num_blocks_per_line, int num_different_blocks, int n):
        data(num_different_blocks*n*n), cols_idx( num_block_rows*num_blocks_per_line), data_idx(cols_idx.size()),
        num_rows(num_block_rows), num_cols(num_block_cols), blocks_per_line(num_blocks_per_line),
        n(n),left(1), right(1){}
    
    typedef thrust::host_vector<double> HVec;
    typedef thrust::host_vector<int> IVec;
    void symv(const HVec& x, HVec& y) const;
    
    HVec data;
    IVec cols_idx, data_idx; 
    int num_rows, num_cols, blocks_per_line;
    int n, left, right;
};


//only one block per line assumed
struct CooSparseBlockMat
{
    typedef double value_type;
    CooSparseBlockMat(){}
    CooSparseBlockMat( int num_block_rows, int num_block_cols, int n, int left, int right):
        num_rows(num_block_rows), num_cols(num_block_cols), num_entries(0),
        n(n),left(left), right(right){}

    void add_value( int row, int col, const thrust::host_vector<double>& element)
    {
        assert( (int)element.size() == n*n);
        int index = data.size()/n/n;
        data.insert( data.end(), element.begin(), element.end());
        rows_idx.push_back(row);
        cols_idx.push_back(col);
        data_idx.push_back( index );

        num_entries++;
    }
    
    typedef thrust::host_vector<double> HVec;
    typedef thrust::host_vector<int> IVec;
    void symv(double alpha, const HVec& x, double beta, HVec& y) const;
    
    HVec data;
    IVec cols_idx, rows_idx, data_idx; 
    int num_rows, num_cols, num_entries;
    int n, left, right;
};
///@cond

void EllSparseBlockMat::symv(const HVec& x, HVec& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);

    int offset[blocks_per_line];
    for( int d=0; d<blocks_per_line; d++)
        offset[d] = cols_idx[blocks_per_line+d]-1;
if(right==1) //alle dx Ableitungen
{
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
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + i)*n+k)*right+j;
        y[I] =0;
        for( int d=0; d<blocks_per_line; d++)
        for( int q=0; q<n; q++) //multiplication-loop
            y[I] += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
    }
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
        y[((s*num_rows + i)*n+k)*right+j] =0;

    for( int d=0; d<blocks_per_line; d++)
    {
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
    }
    }
    }
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + i)*n+k)*right+j;
        y[I] =0;
        for( int d=0; d<blocks_per_line; d++)
        for( int q=0; q<n; q++) //multiplication-loop
            y[I] += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
    }
    //simplest implementation
    //for( int s=0; s<left; s++)
    //for( int i=0; i<num_rows; i++)
    //for( int k=0; k<n; k++)
    //for( int j=0; j<right; j++)
    //{
    //    int I = ((s*num_rows + i)*n+k)*right+j;
    //    y[I] =0;
    //    for( int d=0; d<blocks_per_line; d++)
    //    for( int q=0; q<n; q++) //multiplication-loop
    //        y[I] += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
    //            x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
    //}
}

void CooSparseBlockMat::symv( double alpha, const HVec& x, double beta, HVec& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);
    assert( beta == 1);

    //simplest implementation
    for( int s=0; s<left; s++)
    for( int i=0; i<num_entries; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + rows_idx[i])*n+k)*right+j;
        for( int q=0; q<n; q++) //multiplication-loop
            y[I] += alpha*data[ (data_idx[i]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i])*n+q)*right+j];
    }
}

template <>
struct MatrixTraits<EllSparseBlockMat>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const EllSparseBlockMat>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<CooSparseBlockMat>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const CooSparseBlockMat>
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

} //namespace dg
