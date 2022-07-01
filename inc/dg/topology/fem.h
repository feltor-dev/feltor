#pragma once

#include "grid.h"
#include "xspacelib.h"
#include "fem_weights.h"

namespace dg{
namespace create{

///@addtogroup fem
///@{
//
/*!@class hide_fem_mass_doc
* @brief \f$ S_{ij} = \int v_i(x) v_j(x) \f$ finite element mass matrix
*
* Or in other words the matrix of projection integrals \f$ \int v_i(x) v_j(x)\f$
* @tparam real_type The value type
* @param g The grid
* @return Host Matrix
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/

///@copydoc hide_fem_mass_doc
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> fem_mass(
    const RealGrid1d<real_type>& g)
{
    unsigned num_entries = 3*g.size();
    cusp::coo_matrix<int, real_type, cusp::host_memory> A(g.size(), g.size(), num_entries);
    std::vector<real_type> xx = g.dlt().abscissas();
    std::vector<real_type> xa( g.n()+2);
    for( unsigned i=0; i<g.n(); i++)
        xa[i+1]=xx[i];
    xa[0] = xx[g.n()-1]-2;
    xa[g.n()+1] = xx[0]+2;
    const real_type* x = &xa[1];
    for( unsigned i=0; i<g.N(); i++)
        for( int k=0; k<(int)g.n(); k++)
        {
            if( i==0 && k == 0)
            {
                A.row_indices[0] = 0;
                A.column_indices[0] = 0;
                A.values[0] = g.h()/12*(4*x[0]+6+2*x[1]);
                //A.values[0] = g.h()/6*(x[1]-x[-1]);
                A.row_indices[1] = 0;
                A.column_indices[1] = 1;
                A.values[1] = g.h()/12*(x[1]-x[0]);
                A.row_indices[2] = 0;
                //A.column_indices[2] = g.size()-1;
                A.column_indices[2] = 2; // dummy entry to simplify assembly
                A.values[2] = 0.;
                //A.values[2] = g.h()/12.*(x[0]-x[-1]);
                continue;
            }
            int I = 3*(i*g.n()+k);
            if( (i==g.N()-1) && (k == (int)g.n()-1))
            {
                A.row_indices[I] = g.size()-1;
                A.column_indices[I] = g.size()-2;
                A.values[I] = g.h()/12.*(x[k]-x[k-1]);
                A.row_indices[I+1] = g.size()-1;
                A.column_indices[I+1] = g.size()-1;
                A.values[I+1] = g.h()/12*(-4*x[k]+6-2*x[k-1]);
                //A.values[I+1] = g.h()/6*(x[k+1]-x[k-1]);
                A.row_indices[I+2] = g.size()-1;
                //A.column_indices[I+2] = 0;
                A.column_indices[I+2] = g.size()-3; // dummy entry to simplify assembly
                A.values[I+2] = 0.;
                //A.values[I+2] = g.h()/12.*(x[k+1]-x[k]);
                continue;
            }
            A.row_indices[I] = i*g.n()+k;
            A.column_indices[I] = i*g.n()+k-1;
            A.values[I] = g.h()/12.*(x[k]-x[k-1]);
            A.row_indices[I+1] = i*g.n()+k;
            A.column_indices[I+1] = i*g.n()+k;
            A.values[I+1] = g.h()/6.*(x[k+1]-x[k-1]);
            A.row_indices[I+2] = i*g.n()+k;
            A.column_indices[I+2] = i*g.n()+k+1;
            A.values[I+2] = g.h()/12.*(x[k+1]-x[k]);
        }
    return A;
}

///@copydoc hide_fem_mass_doc
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> fem_mass(
    const aRealTopology2d<real_type>& g)
{
    cusp::csr_matrix<int,real_type,cusp::host_memory> mx = fem_mass(g.gx());
    cusp::csr_matrix<int,real_type,cusp::host_memory> my = fem_mass(g.gy());
    return dg::tensorproduct( my, mx);
}

///@}
}//namespace create
}//namespace dg
