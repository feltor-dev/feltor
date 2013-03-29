#ifndef _DG_OPERATOR_MATRIX_
#define _DG_OPERATOR_MATRIX_

#include <cusp/coo_matrix.h>
#include "operator.cuh"

namespace dg
{


template< class T, size_t n>
Operator<T, n*n> tensorProduct( const Operator< T, n>& op1, const Operator<T, n>& op2)
{
    Operator<T, n*n> prod;
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    prod(i*n+k, j*n+l) = op1(i,j)*op2(k,l);
    return prod;
}


namespace create
{

//creates 1 x op where 1 is the NxN identity matrix
template< class T, size_t n>
cusp::coo_matrix<int,T, cusp::host_memory> operatorMatrix( unsigned N, const Operator<T,n>& op)
{
    //compute number of nonzeroes in op
    unsigned number =0;
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( op(i,j) != 0)
                number++;
    // allocate output matrix
    cusp::coo_matrix<int, T, cusp::host_memory> A(n*N, n*N, N*number);
    unsigned num_triplets = N*number;
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



}//namespace create
    
}//namespace dg

#endif //_DG_OPERATOR_MATRIX_

