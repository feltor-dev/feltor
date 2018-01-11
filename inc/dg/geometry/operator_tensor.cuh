#pragma once 

#include <algorithm>

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include "operator.h"
#include "grid.h"

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
Operator<T> tensorproduct( const Operator<T>& op1, const Operator<T>& op2)
{
#ifdef DG_DEBUG
    assert( op1.size() == op2.size());
#endif //DG_DEBUG
    unsigned n = op1.size();
    Operator<T> prod( n*n);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    prod(i*n+k, j*n+l) = op1(i,j)*op2(k,l);
    return prod;
}


/**
* @brief Tensor product between Delta and an operator
*
* Can be used to create tensors that operate on each dg-vector entry
* The DG Tensor Product 1 x op
* @tparam T value type
* @param N Size of the delta operator
* @param op The Operator
*
* @return A newly allocated cusp matrix
*/
template< class T>
cusp::coo_matrix<int,T, cusp::host_memory> tensorproduct( unsigned N, const Operator<T>& op)
{
    assert( N>0);
    unsigned n = op.size();
    //compute number of nonzeroes in op
    unsigned number =0;
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            //if( op(i,j) != 0)
                number++;
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
 * @brief Multiply 1d matrices by diagonal block batrices from left and right
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
cusp::coo_matrix<int, T, cusp::host_memory> sandwich( const Operator<T>& left,  const cusp::coo_matrix<int, T, cusp::host_memory>& m, const Operator<T>& right)
{
    assert( left.size() == right.size());
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    unsigned n = left.size();
    unsigned N = m.num_rows/n;
    Matrix r = tensorproduc( N, right);
    Matrix l = tensorproduc( N, left);
    Matrix mr(m ), lmr(m);

    cusp::multiply( m, r, mr);
    cusp::multiply( l, mr, lmr);
    return lmr;
}



///@}

    
}//namespace dg


