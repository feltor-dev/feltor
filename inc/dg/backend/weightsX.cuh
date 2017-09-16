#pragma once

#include "weights.cuh"
#include "gridX.h"

/*! @file

  * @brief contains creation functions for integration weights 
  * and their inverse on X-point topology
  */

namespace dg{
namespace create{
///@cond
thrust::host_vector<double> abscissas( const GridX1d& g)
{
    return abscissas(g.grid());
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
thrust::host_vector<double> weights( const aTopologyX2d& g) { return weights( g.grid()); }
/**
* @brief create host_vector containing 2d X-space inverse weight coefficients
*
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> inv_weights( const aTopologyX2d& g) { return inv_weights( g.grid()); }

/**
* @brief create host_vector containing 3d X-space weight coefficients for integration
*
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> weights( const aTopologyX3d& g) { return weights(g.grid()); }

/**
* @brief create host_vector containing 3d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
thrust::host_vector<double> inv_weights( const aTopologyX3d& g) { return inv_weights(g.grid()); }

///@}
}//namespace create
}//namespace dg
