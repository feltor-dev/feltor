#ifndef _DG_CREATION_CUH
#define _DG_CREATION_CUH

#include "operator.cuh"

namespace dg
{
namespace create
{
namespace detail{

//pay attention to duplicate values
template< class T, size_t n>
void add_index( cusp::coo_matrix<int, T, cusp::host_memory>& hm, 
                int& number, 
                unsigned i, unsigned j, unsigned k, unsigned l, 
                T value )
{
    hm.row_indices[number] = n*i+k;
    hm.column_indices[number] = n*j+l;
    hm.values[number] = value;
    number++;
}

//take care that the matrix is properly sorted after use (sort_by_row_and_column)
//take care to not add duplicate values
template< class T, size_t n>
void add_operator( cusp::coo_matrix<int, T, cusp::host_memory>& hm, 
                int& number, 
                unsigned i, unsigned j, 
                Operator<T,n>& op )
{
    for( unsigned k=0; k<n; k++)
        for( unsigned l=0; l<n; l++)
            add_index( hm, number, i,j, k,l, op(k,l));
}

} //namespace detail
} //namespace create
} //namespace dg 


#endif // _DG_CREATION_CUH
