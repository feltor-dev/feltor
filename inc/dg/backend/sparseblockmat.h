#pragma once

#include <thrust/host_vector.h>
#include "matrix_traits.h"

namespace dg
{

//mixed derivatives for jump terms missing
/**
* @brief Ell Sparse Block Matrix format
*
* @ingroup sparsematrix
* The basis of this format is the ell sparse matrix format, i.e. a format
where the numer of entries per line is fixed. 
* The clue is that instead of a values array we use an index array with 
indices into a data array that contains the actual blocks. This safes storage if the number
of nonrecurrent blocks is small. 
The indices and blocks are those of a one-dimensional problem. When we want
to apply the matrix to a multidimensional vector we can multiply it by 
Kronecker deltas of the form
\f[  1\otimes M \otimes 1\f]
where \f$ 1\f$ are diagonal matrices of variable size and \f$ M\f$ is our
one-dimensional matrix. 
*/
template<class value_type>
struct EllSparseBlockMat
{
    //typedef value_type value_type;//!< value type
    /**
    * @brief default constructor does nothing
    */
    EllSparseBlockMat(){}
    /**
    * @brief Allocate storage
    *
    * @param num_block_rows number of rows. Each row contains blocks.
    * @param num_block_cols number of columns.
    * @param num_blocks_per_line number of blocks in each line
    * @param num_different_blocks number of nonrecurrent blocks
    * @param n each block is of size nxn
    */
    EllSparseBlockMat( int num_block_rows, int num_block_cols, int num_blocks_per_line, int num_different_blocks, int n):
        data(num_different_blocks*n*n), cols_idx( num_block_rows*num_blocks_per_line), data_idx(cols_idx.size()),
        num_rows(num_block_rows), num_cols(num_block_cols), blocks_per_line(num_blocks_per_line),
        n(n),left(1), right(1){}

    template< class OtherValueType>
    EllSparseBlockMat( const EllSparseBlockMat<OtherValueType>& src)
    {
        data = src.data;
        cols_idx = src.cols_idx, data_idx = src.data_idx;
        num_rows = src.num_rows, num_cols = src.num_cols, blocks_per_line = src.blocks_per_line;
        n = src.n, left = src.left, right = src.right;
    }
    
    typedef thrust::host_vector<int> IVec;//!< typedef for easy programming
    /**
    * @brief Apply the matrix to a vector
    *
    * @param x input
    * @param y output may not equal input
    */
    void symv(const thrust::host_vector<value_type>& x, thrust::host_vector<value_type>& y) const;
    
    thrust::host_vector<value_type> data;//!< The data array is of size n*n*num_different_blocks and contains the blocks
    IVec cols_idx; //!< is of size num_block_rows*num_blocks_per_line and contains the column indices 
    IVec data_idx; //!< has the same size as cols_idx and contains indices into the data array
    int num_rows; //!< number of rows, each row contains blocks
    int num_cols; //!< number of columns
    int blocks_per_line; //!< number of blocks in each line
    int n;  //!< each block has size n*n
    int left; //!< size of the left Kronecker delta
    int right; //!< size of the right Kronecker delta (is e.g 1 for a x - derivative)

    /**
    * @brief Display internal data to a stream
    *
    * @param os the output stream
    */
    void display( std::ostream& os = std::cout) const;
};


/**
* @brief Coo Sparse Block Matrix format
*
* @ingroup sparsematrix
* The basis of this format is the well-known coordinate sparse matrix format.
* The clue is that instead of a values array we use an index array with 
indices into a data array that contains the actual blocks. This safes storage if the number
of nonrecurrent blocks is small. 
The indices and blocks are those of a one-dimensional problem. When we want
to apply the matrix to a multidimensional vector we can multiply it by 
Kronecker deltas of the form
\f[  1\otimes M \otimes 1\f]
where \f$ 1\f$ are diagonal matrices of variable size and \f$ M\f$ is our
one-dimensional matrix. 
@note This matrix type is used for the computation of boundary points in 
mpi - distributed matrices 
*/
template<class value_type>
struct CooSparseBlockMat
{
    /**
    * @brief default constructor does nothing
    */
    CooSparseBlockMat(){}
    /**
    * @brief Allocate storage
    *
    * @param num_block_rows number of rows. Each row contains blocks.
    * @param num_block_cols number of columns.
    * @param n each block is of size nxn
    * @param left size of the left Kronecker delta
    * @param right size of the right Kronecker delta
    */
    CooSparseBlockMat( int num_block_rows, int num_block_cols, int n, int left, int right):
        num_rows(num_block_rows), num_cols(num_block_cols), num_entries(0),
        n(n),left(left), right(right){}

    /**
    * @brief Convenience function to assemble the matrix
    *
    * appends the given matrix entry to the existing matrix
    * @param row row index
    * @param col column index
    * @param element new block
    */
    void add_value( int row, int col, const thrust::host_vector<value_type>& element)
    {
        assert( (int)element.size() == n*n);
        int index = data.size()/n/n;
        data.insert( data.end(), element.begin(), element.end());
        rows_idx.push_back(row);
        cols_idx.push_back(col);
        data_idx.push_back( index );

        num_entries++;
    }
    
    typedef thrust::host_vector<int> IVec;//!< typedef for easy programming
    /**
    * @brief Apply the matrix to a vector
    *
    * @param alpha multiplies input
    * @param x input
    * @param beta premultiplies output
    * @param y output may not equal input
    */
    void symv(value_type alpha, const thrust::host_vector<value_type>& x, value_type beta, thrust::host_vector<value_type>& y) const;
    /**
    * @brief Display internal data to a stream
    *
    * @param os the output stream
    */
    void display(std::ostream& os = std::cout) const;
    
    thrust::host_vector<value_type> data;//!< The data array is of size n*n*num_different_blocks and contains the blocks
    IVec cols_idx; //!< is of size num_block_rows and contains the column indices 
    IVec rows_idx; //!< is of size num_block_rows and contains the row 
    IVec data_idx; //!< has the same size as cols_idx and contains indices into the data array
    int num_rows; //!< number of rows, each row contains blocks
    int num_cols; //!< number of columns
    int num_entries; //!< number of entries in the matrix
    int n;  //!< each block has size n*n
    int left; //!< size of the left Kronecker delta
    int right; //!< size of the right Kronecker delta (is e.g 1 for a x - derivative)
};
///@cond

template<class value_type>
void EllSparseBlockMat<value_type>::symv(const thrust::host_vector<value_type>& x, thrust::host_vector<value_type>& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);

    /* //THIS IMPLEMENTATION IS NOT VALID SINCE NOT ALL MATRICES ARE TRIVIAL (E.G. X-POINT
    int offset[blocks_per_line];
    for( int d=0; d<blocks_per_line; d++)
        offset[d] = cols_idx[blocks_per_line+d]-1;
if(right==1) //alle dx Ableitungen
{
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<n; k++)
    {
        value_type temp=0;
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
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp+=data[(d*n + k)*n+q]*x[((s*num_cols + i+offset[d])*n+q)];
        y[(s*num_rows+i)*n+k]=temp;
    }
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<n; k++)
    {
        value_type temp=0;
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
            for( int q=0; q<n; q++) //multiplication-loop
                y[I] += data[ (d*n+k)*n+q]*x[((s*num_cols + J)*n+q)*right+j];
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
*/
    //simplest implementation
    for( int s=0; s<left; s++)
    for( int i=0; i<num_rows; i++)
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
}

template<class T>
void EllSparseBlockMat<T>::display( std::ostream& os) const
{
    os << "Data array has   "<<data.size()/n/n<<" blocks of size "<<n<<"x"<<n<<"\n";
    os << "num_rows         "<<num_rows<<"\n";
    os << "num_cols         "<<num_cols<<"\n";
    os << "blocks_per_line  "<<blocks_per_line<<"\n";
    os << "n                "<<n<<"\n";
    os << "left             "<<left<<"\n";
    os << "right            "<<right<<"\n";
    os << "Columns: \n";
    for( int i=0; i<num_rows; i++)
    {
        for( int d=0; d<blocks_per_line; d++)
            os << cols_idx[i*blocks_per_line + d] <<" ";
        os << "\n";
    }
    os << "\n Data: \n";
    for( int i=0; i<num_rows; i++)
    {
        for( int d=0; d<blocks_per_line; d++)
            os << data_idx[i*blocks_per_line + d] <<" ";
        os << "\n";
    }
    os << std::endl;
    
}

template<class value_type>
void CooSparseBlockMat<value_type>::display( std::ostream& os) const
{
    os << "Data array has   "<<data.size()/n/n<<" blocks of size "<<n<<"x"<<n<<"\n";
    os << "num_rows         "<<num_rows<<"\n";
    os << "num_cols         "<<num_cols<<"\n";
    os << "num_entries      "<<num_entries<<"\n";
    os << "n                "<<n<<"\n";
    os << "left             "<<left<<"\n";
    os << "right            "<<right<<"\n";
    os << " Columns: \n";
    for( int i=0; i<num_entries; i++)
        os << cols_idx[i] <<" ";
    os << "\n Rows: \n";
    for( int i=0; i<num_entries; i++)
        os << rows_idx[i] <<" ";
    os << "\n Data: \n";
    for( int i=0; i<num_entries; i++)
        os << data_idx[i] <<" ";
    os << std::endl;
    
}
template<class value_type>
void CooSparseBlockMat<value_type>::symv( value_type alpha, const thrust::host_vector<value_type>& x, value_type beta, thrust::host_vector<value_type>& y) const
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

template <class T>
struct MatrixTraits<EllSparseBlockMat<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class T>
struct MatrixTraits<const EllSparseBlockMat<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class T>
struct MatrixTraits<CooSparseBlockMat<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class T>
struct MatrixTraits<const CooSparseBlockMat<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

} //namespace dg
