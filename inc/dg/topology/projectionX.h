#pragma once
#include "projection.h"
#include "gridX.h"

/*!@file

  @brief Creation of projection matrices for X point topology
 */
namespace dg{
namespace create{
///@addtogroup interpolation
///@{

// TODO document
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const RealGridX1d<real_type>& g_new, const RealGridX1d<real_type>& g_old,std::string method = "dg") {
    return projection(g_new.grid(), g_old.grid(),method);
}

// TODO document
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopologyX2d<real_type>& g_new, const aRealTopologyX2d<real_type>& g_old,std::string method = "dg") {
    return projection(g_new.grid(), g_old.grid(),method);
}

// TODO document
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopologyX3d<real_type>& g_new, const aRealTopologyX3d<real_type>& g_old,std::string method = "dg") {
    return projection(g_new.grid(), g_old.grid(),method);
}

///@}

}//namespace create
}//namespace dg
