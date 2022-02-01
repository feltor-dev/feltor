#pragma once

#include "weights.h"
#include "gridX.h"

/*! @file

  * @brief Creation functions for integration weights
  * and their inverse on X-point topology
  */

namespace dg{
namespace create{

///@addtogroup highlevel
///@{

///@copydoc hide_weights_doc
template<class real_type>
thrust::host_vector<real_type> weights( const dg::RealGridX1d<real_type>& g) { return weights( g.grid()); }
///@copydoc hide_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> inv_weights( const RealGridX1d<real_type>& g) { return inv_weights( g.grid()); }

///@copydoc hide_weights_doc
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopologyX2d<real_type>& g) { return weights( g.grid()); }
///@copydoc hide_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopologyX2d<real_type>& g) { return inv_weights( g.grid()); }

///@copydoc hide_weights_doc
template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopologyX3d<real_type>& g) { return weights(g.grid()); }

///@copydoc hide_inv_weights_doc
template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopologyX3d<real_type>& g) { return inv_weights(g.grid()); }

///@}
}//namespace create
}//namespace dg
