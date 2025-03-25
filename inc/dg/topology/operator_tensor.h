#pragma once

#include <algorithm>

#include "operator.h"

namespace dg
{

///@addtogroup lowlevel
///@{


/**
* @brief Form the tensor product between two operators
*
* Computes C_{ijkl} = op1_{ij}op2_{kl}
* @tparam T The value type
* @param op1 The left hand side
* @param op2 The right hand side
*
* @return The  tensor product
*/
template< class T>
SquareMatrix<T> tensorproduct( const SquareMatrix<T>& op1, const SquareMatrix<T>& op2)
{
    assert( op1.size() == op2.size());
    unsigned n = op1.size();
    SquareMatrix<T> prod( n*n);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    prod(i*n+k, j*n+l) = op1(i,j)*op2(k,l);
    return prod;
}


/**
* @brief Tensor product between Identity matrix and an operator
*
\f[ M = \begin{pmatrix}
op &   &   &   &   & \\
  & op &   &   &   & \\
  &   & op &   &   & \\
  &   &   & op &   & \\
  &   &   &...&   &
  \end{pmatrix}
  \f]
* Can be used to create tensors that operate on each dg-vector entry
* @tparam T value type
* @param N Size of the identity (=number of times op is repeated in the matrix)
* @param op The SquareMatrix
* @return A newly allocated sparse matrix (of size  <tt> N*op.size()</tt> )
* @sa fast_transform
*/
template< class T>
dg::SparseMatrix<int,T, thrust::host_vector> tensorproduct( unsigned N, const SquareMatrix<T>& op)
{
    assert( N>0);
    unsigned n = op.size();
    // allocate output matrix
    thrust::host_vector<int> A_row_offsets(n*N+1), A_column_indices( N*n*n);
    thrust::host_vector<T> A_values( N*n*n);
    A_row_offsets[0] = 0;
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<n; i++)
        {
            A_row_offsets[k*n+i+1]      = A_row_offsets[k*n+i] + n;
            for( unsigned j=0; j<n; j++)
                //if( op(i,j) != 0)
                {
                    A_column_indices[(k*n+i)*n+j]   = k*n+j;
                    A_values[(k*n+i)*n+j]           = op(i,j);
                }
        }
    return {n*N, n*N, A_row_offsets, A_column_indices, A_values};
}


/**
 * @brief Multiply 1d matrices by diagonal block matrices from left and right
 *
 * computes (1xleft)m(1xright)
 * @tparam T value type
 * @param left The left hand side
 * @param m The matrix
 * @param right The right hand side
 *
 * @return A newly allocated sparse matrix
 */
template< class T>
dg::SparseMatrix<int, T, thrust::host_vector> sandwich( const SquareMatrix<T>& left,  const dg::SparseMatrix<int, T, thrust::host_vector>& m, const SquareMatrix<T>& right)
{
    assert( left.size() == right.size());
    unsigned n = left.size();
    unsigned N = m.num_rows()/n;
    return tensorproduct( N, left)*(m*tensorproduct( N, right));
}



///@}


}//namespace dg


