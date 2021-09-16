#pragma once

#include "grid.h"
#include "fem_weights.h"

namespace dg{
namespace create{

template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> fem_dx(
    const RealGrid1d<real_type>& g, bc bcx)
{
    unsigned num_entries = 2*g.size();
    cusp::coo_matrix<int, real_type, cusp::host_memory> A(g.size(), g.size(), num_entries);
    A.row_indices[0] = 0;
    if( bcx == dg::PER)
    {
        A.column_indices[0] = g.size()-1;
        A.values[0] = -0.5;
    }
    else if( bcx == dg::DIR || bcx == dg::DIR_NEU)
    {
        A.column_indices[0] = 0;
        A.values[0] = +0.5;
    }
    else if( bcx == dg::NEU || bcx == dg::NEU_DIR)
    {
        A.column_indices[0] = 0;
        A.values[0] = -0.5;
    }
    A.row_indices[1] = 0;
    A.column_indices[1] = 1;
    A.values[1] = 0.5;

    for( unsigned i=1; i<g.size()-1; i++)
    {
        A.row_indices[2*i] = i;
        A.column_indices[2*i] = i-1;
        A.values[2*i] = -0.5;
        A.row_indices[2*i+1] = i;
        A.column_indices[2*i+1] = i+1;
        A.values[2*i+1] = 0.5;
    }
    unsigned I = g.size()-1;
    A.row_indices[2*I] = I;
    A.column_indices[2*I] = I-1;
    A.values[2*I] = -0.5;

    A.row_indices[2*I+1] = I;
    if( bcx == dg::PER)
    {
        A.column_indices[2*I+1] = 0;
        A.values[2*I+1] = 0.5;
    }
    else if( bcx == dg::DIR || bcx == dg::NEU_DIR)
    {
        A.column_indices[2*I+1] = I;
        A.values[2*I+1] = -0.5;
    }
    else if( bcx == dg::NEU || bcx == dg::DIR_NEU)
    {
        A.column_indices[2*I+1] = I;
        A.values[2*I+1] = 0.5;
    }
    return A;
}
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> fem_dx(
    const RealGrid1d<real_type>& g)
{
    return fem_dx( g, g.bcx());
}

}//namespace create
}//namespace dg
