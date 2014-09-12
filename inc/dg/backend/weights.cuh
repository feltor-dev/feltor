#pragma once

#include <thrust/host_vector.h>
#include "grid.h"

/*! @file

  * contains creation functions for integration weights 
  * and their inverse
  */

namespace dg{
namespace create{
    
///@cond
/**
* @brief create host_vector containing 1d X-space abscissas 
*
* same as evaluation of f(x) = x on the grid
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> abscissas( const Grid1d<T>& g)
{
    thrust::host_vector<T> v(g.size()); 
    T xp=1.;
    for( unsigned i=0; i<g.N(); i++)
    {
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n()+j] =  g.h()/2.*(xp + g.dlt().abscissas()[j])+g.x0();
        xp+=2.;
    }
    return v;
}
///@endcond


///@addtogroup highlevel
///@{


///@cond
/**
* @brief create host_vector containing 1d X-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> weights( const Grid1d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n() + j] = g.h()/2.*g.dlt().weights()[j];
    return v;
}
/**
* @brief create host_vector containing 1d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> inv_weights( const Grid1d<T>& g)
{
    thrust::host_vector<T> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

namespace detail{

int get_i( unsigned n, int idx) { return idx%(n*n)/n;}
int get_j( unsigned n, int idx) { return idx%(n*n)%n;}
int get_i( unsigned n, unsigned Nx, int idx) { return (idx/(n*Nx))%n;}
int get_j( unsigned n, unsigned Nx, int idx) { return idx%n;}
}//namespace detail
///@endcond

/**
* @brief create host_vector containing 2d X-space integration weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> weights( const Grid2d<T>& g)
{
    //choose layout
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        //v[i] = g.hx()*g.hy()/4.*g.dlt().weights()[detail::get_i(g.n(), i)]*g.dlt().weights()[detail::get_j(g.n(), i)];
        v[i] = g.hx()*g.hy()/4.*
                g.dlt().weights()[detail::get_i(g.n(),g.Nx(), i)]*
                g.dlt().weights()[detail::get_j(g.n(),g.Nx(), i)];
    return v;
}
/**
* @brief create host_vector containing 2d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> inv_weights( const Grid2d<T>& g)
{
    thrust::host_vector<T> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
/**
* @brief create host_vector containing 3d X-space weight coefficients for integration
*
* If cylindrical coordinates are used the coefficients are multiplied by R
* to give the correct volume form.
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> weights( const Grid3d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        //v[i] = g.hz()*g.hx()*g.hy()/4.*g.dlt().weights()[detail::get_i(g.n(), i)]*g.dlt().weights()[detail::get_j(g.n(), i)];
        v[i] = g.hz()*g.hx()*g.hy()/4.*
               g.dlt().weights()[detail::get_i(g.n(), g.Nx(), i)]*
               g.dlt().weights()[detail::get_j(g.n(), g.Nx(), i)];
    if( g.system() == cylindrical)
    {
        Grid1d<T> gR( g.x0(), g.x1(), g.n(), g.Nx());
        thrust::host_vector<T> absc( abscissas( gR)); 
        for( unsigned i=0; i<g.size(); i++)
            v[i] *= absc[i%(g.n()*g.Nx())];
    }
    return v;
}
/**
* @brief create host_vector containing 3d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> inv_weights( const Grid3d<T>& g)
{
    thrust::host_vector<T> v = weights( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}


///@}
}//namespace create
}//namespace dg
