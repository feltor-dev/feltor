#pragma once

#include <thrust/host_vector.h>
#include "grid.h"
#include "../enums.h"

/*! @file

  * @brief Creation functions for integration weights and their inverse
  */

namespace dg{
namespace create{

///@addtogroup highlevel
///@{

/*!@class hide_weights_doc
* @brief Nodal weight coefficients
* @param g The grid
* @return Host Vector
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/
/*!@class hide_inv_weights_doc
* @brief inverse nodal weight coefficients
* @param g The grid
* @return Host Vector
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/
/*!@class hide_weights_coo_doc
* @brief nodal weight coefficients
* @param g The grid
* @param coo The coordinate for which to generate the weights (in 2d only \c dg::x and \c dg::y are allowed)
* @return Host Vector with full grid size
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/

///@copydoc hide_weights_doc
///@copydoc hide_code_evaluate1d
template<class real_type>
thrust::host_vector<real_type> weights( const RealGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n() + j] = g.h()/2.*g.dlt().weights()[j];
    return v;
}
///@copydoc hide_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> inv_weights( const RealGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@copydoc hide_weights_doc
///@copydoc hide_code_evaluate2d
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopology2d<real_type>& g)
{
    thrust::host_vector<real_type> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()/4.*
                g.dlty().weights()[(i/(g.nx()*g.Nx()))%g.ny()]*
                g.dltx().weights()[i%g.nx()];
    return v;
}
///@copydoc hide_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopology2d<real_type>& g)
{
    thrust::host_vector<real_type> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@copydoc hide_weights_coo_doc
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopology2d<real_type>& g, enum coo2d coo)
{
    thrust::host_vector<real_type> w( g.size());
    if( coo == coo2d::x) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hx()/2.* g.dltx().weights()[i%g.nx()];
    }
    else if( coo == coo2d::y) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hy()/2.* g.dlty().weights()[(i/(g.nx()*g.Nx()))%g.ny()];
    }
    return w;
}


///@copydoc hide_weights_doc
///@copydoc hide_code_evaluate3d
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopology3d<real_type>& g)
{
    // this implementation is binary compatible with nz = 1 old implementation
    thrust::host_vector<real_type> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()*g.hz()/4.*
               (g.dltz().weights()[(i/(g.nx()*g.ny()*g.Nx()*g.Ny()))%g.nz()]/2.)*
               g.dlty().weights()[(i/(g.nx()*g.Nx()))%g.ny()]*
               g.dltx().weights()[i%g.nx()];
    return v;
}

///@copydoc hide_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopology3d<real_type>& g)
{
    thrust::host_vector<real_type> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@copydoc hide_weights_coo_doc
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopology3d<real_type>& g, enum coo3d coo)
{
    thrust::host_vector<real_type> w( g.size());
    if( coo == coo3d::x) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hx()/2.* g.dltx().weights()[i%g.nx()];
    }
    else if( coo == coo3d::y) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hy()/2.* g.dlty().weights()[(i/(g.nx()*g.Nx()))%g.ny()];
    }
    else if( coo == coo3d::z) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hz()/2.* g.dltz().weights()[(i/(g.nx()*g.Nx()*g.ny()*g.Ny()))%g.nz()];
    }
    else if( coo == coo3d::xy) {
        for( unsigned i=0; i<g.size(); i++)
        {
            w[i] = g.hx()/2.* g.dltx().weights()[i%g.nx()];
            w[i]*= g.hy()/2.* g.dlty().weights()[(i/(g.nx()*g.Nx()))%g.ny()];
        }
    }
    else if( coo == coo3d::yz) {
        for( unsigned i=0; i<g.size(); i++)
        {
            w[i] = g.hy()/2.* g.dlty().weights()[(i/(g.nx()*g.Nx()))%g.ny()];
            w[i]*= g.hz()/2.* g.dltz().weights()[(i/(g.nx()*g.Nx()*g.ny()*g.Ny()))%g.nz()];
        }
    }
    else if( coo == coo3d::xz) {
        for( unsigned i=0; i<g.size(); i++)
        {
            w[i] = g.hx()/2.* g.dltx().weights()[i%g.nx()];
            w[i]*= g.hz()/2.* g.dltz().weights()[(i/(g.nx()*g.Nx()*g.ny()*g.Ny()))%g.nz()];
        }
    }
    return w;
}

///@}
}//namespace create
}//namespace dg
