#ifndef _DG_OPERATOR_MATRIX_
#define _DG_OPERATOR_MATRIX_
#include <algorithm>

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include "operator_dynamic.h"
#include "grid.cuh"

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
Operator<T> tensor( const Operator<T>& op1, const Operator<T>& op2)
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


//namespace create
//{

//creates 1 x op where 1 is the NxN identity matrix
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
cusp::coo_matrix<int,T, cusp::host_memory> tensor( unsigned N, const Operator<T>& op)
{
    assert( N>0);
    unsigned n = op.size();
    //compute number of nonzeroes in op
    unsigned number =0;
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( op(i,j) != 0)
                number++;
    // allocate output matrix
    cusp::coo_matrix<int, T, cusp::host_memory> A(n*N, n*N, N*number);
    number = 0;
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
                if( op(i,j) != 0)
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
    Matrix r = tensor( N, right);
    Matrix l = tensor( N, left);
    Matrix mr(m ), lmr(m);

    cusp::multiply( m, r, mr);
    cusp::multiply( l, mr, lmr);
    return lmr;
}
//sandwich l space matrix to make x space matrix
/*
 * @brief Transforms a 1d matrix in l-space to x-space
 *
 * computes (1xbackward)m(1xforward)
 * @tparam T value type
 * @param m The matrix
 *
 * @return A newly allocated cusp matrix containing the x-space version of m
 */
/*
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> sandwich( unsigned n, const cusp::coo_matrix<int, T, cusp::host_memory>& m)
{
    Operator<T> forward1d = create::forward(n);
    Operator<T> backward1d = create::backward(n);
    return sandwich( backward1d, m, forward1d);
}
*/

//use symmetry of matrix
template<class Matrix>
void transverse( const Matrix& in, Matrix& out)
{
    //USE MATRIX SYMMETRY AND DO A THRUST::SORT_BY_KEY ON VALUES
    //WRITE PRECONDITIONS AND MAKE SURE LAPLACE FUNCTIONS SET ALL VALUES
    //EVTL NUR IN XSPACE DA MUSS MAN SICH NICHT UM VORZEICHENWECHSEL KÜMMERN
    typedef typename Matrix::value_type value_type;
    typedef int index_type;
    //cusp::print( in);
    out = in;
    /*
    thrust::sort( out.row_indices.begin(), out.row_indices.end(), thrust::greater<index_type>());
    out.sort_by_row();
    out.row_indices.swap( out.column_indices); //transpose
    //Punktspiegelung
    out.sort_by_row();
    thrust::sort( out.row_indices.begin(), out.row_indices.end(), thrust::greater<index_type>());
    out.sort_by_row_and_column();
    */
    thrust::host_vector<int> keys( in.num_entries);
    thrust::sequence( keys.begin(), keys.end());
    thrust::sort_by_key( keys.begin(), keys.end(), out.values.begin(), thrust::greater<value_type>());

}


///@}

    
}//namespace dg

#endif //_DG_OPERATOR_MATRIX_

