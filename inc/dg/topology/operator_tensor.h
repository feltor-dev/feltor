#pragma once

#include <algorithm>

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
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
* @return A newly allocated cusp matrix (of size  <tt> N*op.size()</tt> )
* @sa fast_transform
*/
template< class T>
cusp::coo_matrix<int,T, cusp::host_memory> tensorproduct( unsigned N, const SquareMatrix<T>& op)
{
    assert( N>0);
    unsigned n = op.size();
    //compute number of nonzeroes in op
    unsigned number = n*n;
    // allocate output matrix
    cusp::coo_matrix<int, T, cusp::host_memory> A(n*N, n*N, N*number);
    number = 0;
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
                //if( op(i,j) != 0)
                {
                    A.row_indices[number]      = k*n+i;
                    A.column_indices[number]   = k*n+j;
                    A.values[number]           = op(i,j);
                    number++;
                }
    return A;
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
 * @return A newly allocated cusp matrix
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> sandwich( const SquareMatrix<T>& left,  const cusp::coo_matrix<int, T, cusp::host_memory>& m, const SquareMatrix<T>& right)
{
    assert( left.size() == right.size());
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    unsigned n = left.size();
    unsigned N = m.num_rows/n;
    Matrix r = tensorproduct( N, right);
    Matrix l = tensorproduct( N, left);
    Matrix mr(m ), lmr(m);

    cusp::multiply( m, r, mr);
    cusp::multiply( l, mr, lmr);
    return lmr;
}



///@}


}//namespace dg


