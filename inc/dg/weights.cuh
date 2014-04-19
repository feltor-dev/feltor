#pragma once

#include "thrust/host_vector.h"
#include "grid.cuh"

namespace dg{
namespace create{

///@addtogroup highlevel
///@{

///@cond
/**
* @brief create host_vector containing 1d L-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> s1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n()+j] = g.hx() / (double)(2*j+1);
    return v;
}
/**
* @brief create host_vector containing 1d L-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> t1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v = s1d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
///@endcond
/**
* @brief create host_vector containing 2d L-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> s2d( const Grid2d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    unsigned n= g.n();
    for( unsigned i=0; i<g.Ny(); i++)
        for( unsigned j=0; j<g.Nx(); j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v[i*n*n*g.Nx()+j*n*n+k*n+l] = g.hx()*g.hy() / (double)((2*k+1)*(2*l+1));
    return v;
}
/**
* @brief create host_vector containing 2d L-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> t2d( const Grid2d<T>& g)
{
    thrust::host_vector<T> v = s2d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;

}

/**
* @brief create host_vector containing 3d L-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> s3d( const Grid3d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    unsigned n= g.n();
    for( unsigned s=0; s<g.Nz(); s++)
    for( unsigned i=0; i<g.Ny(); i++)
        for( unsigned j=0; j<g.Nx(); j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v[s*n*n*g.Nx()*g.Ny()+i*n*n*g.Nx()+j*n*n+k*n+l] = g.hx()*g.hy()*g.hz() / (double)((2*k+1)*(2*l+1));
    return v;
}

/**
* @brief create host_vector containing 3d L-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> t3d( const Grid3d<T>& g)
{
    thrust::host_vector<T> v = s3d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;

}

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
thrust::host_vector<T> w1d( const Grid1d<T>& g)
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
thrust::host_vector<T> v1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v = w1d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

namespace detail{

int get_i( unsigned n, int idx) { return idx%(n*n)/n;}
int get_j( unsigned n, int idx) { return idx%(n*n)%n;}
}//namespace detail
///@endcond

/**
* @brief create host_vector containing 2d X-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> w2d( const Grid2d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()/4.*g.dlt().weights()[detail::get_i(g.n(), i)]*g.dlt().weights()[detail::get_j(g.n(), i)];
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
thrust::host_vector<T> v2d( const Grid2d<T>& g)
{
    thrust::host_vector<T> v = w2d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
/**
* @brief create host_vector containing 3d X-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> w3d( const Grid3d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hz()*g.hx()*g.hy()/4.*g.dlt().weights()[detail::get_i(g.n(), i)]*g.dlt().weights()[detail::get_j(g.n(), i)];
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
thrust::host_vector<T> v3d( const Grid3d<T>& g)
{
    thrust::host_vector<T> v = w3d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
///@cond
/**
* @brief create host_vector containing 1d X-space abscissas 
*
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

///@}
}//namespace create
}//namespace dg
