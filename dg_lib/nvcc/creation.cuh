#ifndef _DG_CREATION_CUH
#define _DG_CREATION_CUH

namespace dg
{
namespace create
{
namespace detail{

template< size_t n>
void add_index( cusp::coo_matrix<int, double, cusp::host_memory>& hm, 
                int& number, 
                unsigned i, unsigned j, unsigned k, unsigned l, 
                double value )
{
    hm.row_indices[number] = n*i+k;
    hm.column_indices[number] = n*j+l;
    hm.values[number] = value;
    number++;
}
} //namespace detail
} //namespace create
} //namespace dg 


#endif // _DG_CREATION_CUH
