#pragma once

#include "weights.cuh"
#include "gridX.h"

/*! @file

  * @brief contains creation functions for integration weights
  * and their inverse on X-point topology
  */

namespace dg{
namespace create{

///@addtogroup highlevel
///@{

/**
* @brief create host_vector containing 1d X-space weight coefficients
*
* @param g The grid
*
* @return Host Vector
*/
template<class real_type>
thrust::host_vector<real_type> weights( const dg::RealGridX1d<real_type>& g) { return weights( g.grid()); }
/**
* @brief create host_vector containing 1d X-space inverse weight coefficients
*
* @param g The grid
*
* @return Host Vector
*/
template<class real_type>
thrust::host_vector<real_type> inv_weights( const RealGridX1d<real_type>& g) { return inv_weights( g.grid()); }

/**
* @brief create host_vector containing 2d X-space integration weight coefficients
*
* @param g The grid
*
* @return Host Vector
*/
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopologyX2d<real_type>& g) { return weights( g.grid()); }
/**
* @brief create host_vector containing 2d X-space inverse weight coefficients
*
* @param g The grid
*
* @return Host Vector
*/
template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopologyX2d<real_type>& g) { return inv_weights( g.grid()); }

/**
* @brief create host_vector containing 3d X-space weight coefficients for integration
*
* @param g The grid
*
* @return Host Vector
*/
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopologyX3d<real_type>& g) { return weights(g.grid()); }

/**
* @brief create host_vector containing 3d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid
*
* @return Host Vector
*/
template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopologyX3d<real_type>& g) { return inv_weights(g.grid()); }

///@}
}//namespace create
}//namespace dg
