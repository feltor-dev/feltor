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
    SparseBlockMat( unsigned d, unsigned num_rows, unsigned num_cols):diag(d), row(d), col(d), block(d),
        n(d),num_rows(d, num_rows), num_cols(d, num_cols), num_blocks(d), left(d,1), right(d,1){}
    
    typedef thrust::host_vector<double> HVec;
    typedef thrust::host_vector<short> IVec;
    void symv(const HVec& x, HVec& y) const;
    
    //every element of the std::vector represents one diagonal (mixed derivatives possible)
    std::vector<HVec> diag;
    std::vector<IVec> row, col, block; //column is the vector block idx, and block is an index into the diag array determine how to traverse the matrix
    std::vector<unsigned> n; //Block size (nxn blocks)
    std::vector<unsigned> num_rows, num_cols, num_blocks;//number of entries
    std::vector<unsigned> left, right; //size of the one 
    HVec norm; //the normalization vector
};

void SparseBlockMat::symv(const HVec& x, HVec& y) const
{
    dg::blas1::detail::doScal(y,0, ThrustVectorTag());
    for( unsigned m=0; m<diag.size(); m++)//execute serially
    for( unsigned s=0; s<left[m]; s++)
    for( unsigned i=0; i<num_blocks[m]; i++)
    for( unsigned k=0; k<n[m]; k++)
    for( unsigned j=0; j<right[m]; j++)
    for( unsigned q=0; q<n[m]; q++) //multiplication-loop
        y[((s*num_rows[m] + row[m][i])*n[m]+k)*right[m]+j] += 
            diag[m][(block[m][i]*n[m]+k)*n[m]+q]*
            x[((s*num_cols[m] + col[m][i])*n[m]+q)*right[m]+j];
        
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
