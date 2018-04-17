#pragma once

#include <thrust/host_vector.h>
#include "grid.h"
#include "../enums.h"

/*! @file

  * @brief contains creation functions for integration weights and their inverse
  */

namespace dg{
namespace create{

///@addtogroup highlevel
///@{

/*!@class hide_weights_doc
<<<<<<< HEAD:inc/dg/geometry/weights.cuh
* @brief create host vector containing X-space weight coefficients
=======
* @brief create host_vector containing X-space weight coefficients
>>>>>>> master:inc/dg/backend/weights.cuh
* @param g The grid
* @return Host Vector
* @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
*/
/*!@class hide_inv_weights_doc
<<<<<<< HEAD:inc/dg/geometry/weights.cuh
* @brief create host vector containing inverse X-space weight coefficients
=======
* @brief create host_vector containing inverse X-space weight coefficients
>>>>>>> master:inc/dg/backend/weights.cuh
* @param g The grid
* @return Host Vector
* @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
*/
/*!@class hide_weights_coo_doc
* @brief create host vector containing X-space weight coefficients
* @param g The grid
* @param coo The coordinate for which to generate the weights (in 2d only \c dg::x and \c dg::y are allowed)
* @return Host Vector with full grid size
* @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
*/

///@copydoc hide_weights_doc
///@copydoc hide_code_evaluate1d
thrust::host_vector<double> weights( const Grid1d& g)
{
    thrust::host_vector<double> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n() + j] = g.h()/2.*g.dlt().weights()[j];
    return v;
}
///@copydoc hide_inv_weights_doc
thrust::host_vector<double> inv_weights( const Grid1d& g)
{
    thrust::host_vector<double> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@cond
namespace detail{

int get_i( unsigned n, int idx) { return idx%(n*n)/n;}
int get_j( unsigned n, int idx) { return idx%(n*n)%n;}
int get_i( unsigned n, unsigned Nx, int idx) { return (idx/(n*Nx))%n;}
int get_j( unsigned n, unsigned Nx, int idx) { return idx%n;}
}//namespace detail
///@endcond

///@copydoc hide_weights_doc
///@copydoc hide_code_evaluate2d
thrust::host_vector<double> weights( const aTopology2d& g)
{
    thrust::host_vector<double> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()/4.*
                g.dlt().weights()[detail::get_i(g.n(),g.Nx(), i)]*
                g.dlt().weights()[detail::get_j(g.n(),g.Nx(), i)];
    return v;
}
///@copydoc hide_inv_weights_doc
thrust::host_vector<double> inv_weights( const aTopology2d& g)
{
    thrust::host_vector<double> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@copydoc hide_weights_coo_doc
thrust::host_vector<double> weights( const aTopology2d& g, enum coo2d coo)
{
    thrust::host_vector<double> w( g.size());
    if( coo == coo2d::x) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hx()/2.* g.dlt().weights()[i%g.n()];
    }
    else if( coo == coo2d::y) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hy()/2.* g.dlt().weights()[(i/(g.n()*g.Nx()))%g.n()];
    }
    return w;
}


///@copydoc hide_weights_doc
///@copydoc hide_code_evaluate3d
thrust::host_vector<double> weights( const aTopology3d& g)
{
    thrust::host_vector<double> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()*g.hz()/4.*
               g.dlt().weights()[detail::get_i(g.n(), g.Nx(), i)]*
               g.dlt().weights()[detail::get_j(g.n(), g.Nx(), i)];
    return v;
}

///@copydoc hide_inv_weights_doc
thrust::host_vector<double> inv_weights( const aTopology3d& g)
{
    thrust::host_vector<double> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

///@copydoc hide_weights_coo_doc
thrust::host_vector<double> weights( const aTopology3d& g, enum coo3d coo)
{
    thrust::host_vector<double> w( g.size());
    if( coo == coo3d::x) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hx()/2.* g.dlt().weights()[i%g.n()];
    }
    else if( coo == coo3d::y) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hy()/2.* g.dlt().weights()[(i/(g.n()*g.Nx()))%g.n()];
    }
    else if( coo == coo3d::z) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hz();
    }
    else if( coo == coo3d::xy) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hx()*g.hy()/4.* g.dlt().weights()[i%g.n()]*g.dlt().weights()[(i/(g.n()*g.Nx()))%g.n()];
    }
    else if( coo == coo3d::yz) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hy()*g.hz()/2.* g.dlt().weights()[(i/(g.n()*g.Nx()))%g.n()];
    }
    else if( coo == coo3d::xz) {
        for( unsigned i=0; i<g.size(); i++)
            w[i] = g.hx()*g.hz()/2.* g.dlt().weights()[i%g.n()];
    }
    return w;
}

///@}
}//namespace create
}//namespace dg
