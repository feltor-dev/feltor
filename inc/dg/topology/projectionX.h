#pragma once
#include "projection.h"
#include "gridX.h"

/*!@file

  @brief Creation of projection matrices for X point topology
 */
namespace dg{
///@addtogroup interpolation
///@{
namespace create{

///@copydoc interpolationT(const RealGrid1d&,const RealGrid1d&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolationT( const RealGridX1d<real_type>& g_new, const RealGridX1d<real_type>& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}
///@copydoc interpolationT(const RealGrid1d&,const RealGrid1d&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolationT( const aRealTopologyX2d<real_type>& g_new, const aRealTopologyX2d<real_type>& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}
///@copydoc interpolationT(const RealGrid1d&,const RealGrid1d&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolationT( const aRealTopologyX3d<real_type>& g_new, const aRealTopologyX3d<real_type>& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}

///@copydoc projection(const RealGrid1d&,const RealGrid1d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const RealGridX1d<real_type>& g_new, const RealGridX1d<real_type>& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@copydoc projection(const RealGrid1d&,const RealGrid1d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopologyX2d<real_type>& g_new, const aRealTopologyX2d<real_type>& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@copydoc projection(const RealGrid1d&,const RealGrid1d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopologyX3d<real_type>& g_new, const aRealTopologyX3d<real_type>& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@}

}//namespace create
}//namespace dg
