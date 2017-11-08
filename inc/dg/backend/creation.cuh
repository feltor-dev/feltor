#ifndef _DG_CREATION_CUH
#define _DG_CREATION_CUH

#include <cusp/coo_matrix.h>
#include "operator.h"

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

//add line row_index to the matrix hm beginning from col_begin 
template< class T>
void add_line( cusp::coo_matrix<int, T, cusp::host_memory>& hm, 
                int& number, 
                unsigned row_index, 
                unsigned col_begin, 
                std::vector<T>& vec )
{
    for( unsigned k=0; k<vec.size(); k++)
    {
        hm.row_indices[number] = row_index;
        hm.column_indices[number] = col_begin+k;
        hm.values[number] = vec[k];
        number++;
    }
}
//add line row_index to the matrix hm beginning from col_begin 
template< class T>
void add_line( cusp::coo_matrix<int, T, cusp::host_memory>& hm, 
                int& number, 
                unsigned row_index, 
                unsigned col_begin, 
                unsigned n, 
                unsigned Nx,
                std::vector<T>& vec )
{
    for( unsigned k=0; k<n; k++)
        for( unsigned l=0; l<n; l++)
        {
            hm.row_indices[number] = row_index;
            hm.column_indices[number] = col_begin+k*n*Nx + l;
            hm.values[number] = vec[k*n+l];
            number++;
        }
}

} //namespace detail
} //namespace create
} //namespace dg 
///@endcond

#endif // _DG_CREATION_CUH
