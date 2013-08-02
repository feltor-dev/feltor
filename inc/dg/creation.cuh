#ifndef _DG_CREATION_CUH
#define _DG_CREATION_CUH

#include "operator_dynamic.h"

///@cond
namespace dg
{
namespace create
{
namespace detail{

//pay attention to duplicate values
template< class T>
void add_index( unsigned n,
                cusp::coo_matrix<int, T, cusp::host_memory>& hm, 
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
template< class T>
void add_operator( cusp::coo_matrix<int, T, cusp::host_memory>& hm, 
                int& number, 
                unsigned i, unsigned j, 
                Operator<T>& op )
{
    for( unsigned k=0; k<op.size(); k++)
        for( unsigned l=0; l<op.size(); l++)
            add_index( op.size(), hm, number, i,j, k,l, op(k,l));
}

} //namespace detail
} //namespace create
} //namespace dg 

///@endcond

#endif // _DG_CREATION_CUH
