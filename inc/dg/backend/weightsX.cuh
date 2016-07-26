#pragma once

#include "weights.cuh"

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

/**
* @brief create host_vector containing 1d X-space weight coefficients
*
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> weights( const dg::GridX1d& g) { return weights( g.grid()); }
/**
* @brief create host_vector containing 1d X-space inverse weight coefficients
*
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> inv_weights( const GridX1d& g) { return inv_weights( g.grid()); }

/**
* @brief create host_vector containing 2d X-space integration weight coefficients
*
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> weights( const GridX2d& g) { return weights( g.grid()); }
/**
* @brief create host_vector containing 2d X-space inverse weight coefficients
*
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> inv_weights( const GridX2d& g) { return inv_weights( g.grid()); }

/**
* @brief create host_vector containing 3d X-space weight coefficients for integration
*
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> weights( const GridX3d& g) { return weights(g.grid()); }

/**
* @brief create host_vector containing 3d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> inv_weights( const GridX3d& g) { return inv_weights(g.grid()); }

///@}
}//namespace create
}//namespace dg
