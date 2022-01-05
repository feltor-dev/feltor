#pragma once
#include "weights.h"

/*! @file

  * @brief Creation functions for finite element utilities
  */
namespace dg {
namespace create{
///@cond
namespace detail
{
template<class real_type>
std::vector<real_type> fem_weights( const DLT<real_type>& dlt)
{
    std::vector<real_type> x = dlt.abscissas();
    std::vector<real_type> w = x;
    unsigned n = x.size();
    if( n== 1)
        w[0] = 2;
    else
    {
        w[0] = 0.5*(x[1] - (x[n-1]-2));
        w[n-1] = 0.5*((x[0]+2) - x[n-2]);
        for( unsigned i=1; i<n-1; i++)
            w[i] = 0.5*(x[i+1]-x[i-1]);
    }
    return w;
}
}//namespace detail
///@endcond

///@addtogroup fem
///@{

/*!@class hide_fem_weights_doc
* @brief finite element weight coefficients
*
 * These will emulate the trapezoidal rule for integration
* @param g The grid
* @return Host Vector
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/
/*!@class hide_fem_inv_weights_doc
* @brief inverse finite element weight coefficients
* @param g The grid
* @return Host Vector
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/

///@copydoc hide_fem_weights_doc
template<class real_type>
thrust::host_vector<real_type> fem_weights( const RealGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> v( g.size());
    std::vector<real_type> w = detail::fem_weights(g.dlt());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n() + j] = g.h()/2.*w[j];
    return v;
}
///@copydoc hide_fem_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> fem_inv_weights( const RealGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> v = fem_weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@copydoc hide_fem_weights_doc
template<class real_type>
thrust::host_vector<real_type> fem_weights( const aRealTopology2d<real_type>& g)
{
    thrust::host_vector<real_type> v( g.size());
    std::vector<real_type> w = detail::fem_weights(g.dlt());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()/4.*
                w[detail::get_i(g.n(),g.Nx(), i)]*
                w[detail::get_j(g.n(),g.Nx(), i)];
    return v;
}
///@copydoc hide_fem_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> fem_inv_weights( const aRealTopology2d<real_type>& g)
{
    thrust::host_vector<real_type> v = fem_weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@copydoc hide_fem_weights_doc
template<class real_type>
thrust::host_vector<real_type> fem_weights( const aRealTopology3d<real_type>& g)
{
    thrust::host_vector<real_type> v( g.size());
    std::vector<real_type> w = detail::fem_weights(g.dlt());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()*g.hz()/4.*
               w[detail::get_i(g.n(), g.Nx(), i)]*
               w[detail::get_j(g.n(), g.Nx(), i)];
    return v;
}

///@copydoc hide_fem_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> fem_inv_weights( const aRealTopology3d<real_type>& g)
{
    thrust::host_vector<real_type> v = fem_weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
///@}
}//namespace create
}//namespace dg
