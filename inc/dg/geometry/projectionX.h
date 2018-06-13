#pragma once
#include "projection.cuh"
#include "gridX.h"

/*!@file

  @brief contains creation of projection matrices for X point topology
 */
namespace dg{
///@addtogroup interpolation
///@{
namespace create{

///@copydoc interpolationT(const Grid1d&,const Grid1d&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolationT( const BasicGridX1d<real_type>& g_new, const BasicGridX1d<real_type>& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolationT( const aBasicTopologyX2d<real_type>& g_new, const aBasicTopologyX2d<real_type>& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolationT( const aBasicTopologyX3d<real_type>& g_new, const aBasicTopologyX3d<real_type>& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const BasicGridX1d<real_type>& g_new, const BasicGridX1d<real_type>& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aBasicTopologyX2d<real_type>& g_new, const aBasicTopologyX2d<real_type>& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aBasicTopologyX3d<real_type>& g_new, const aBasicTopologyX3d<real_type>& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@}

}//namespace create
}//namespace dg
