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
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const GridX1d& g_new, const GridX1d& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const aTopologyX2d& g_new, const aTopologyX2d& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const aTopologyX3d& g_new, const aTopologyX3d& g_old) {
    return interpolationT( g_new.grid(), g_old.grid());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
cusp::coo_matrix< int, double, cusp::host_memory> projection( const GridX1d& g_new, const GridX1d& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
cusp::coo_matrix< int, double, cusp::host_memory> projection( const aTopologyX2d& g_new, const aTopologyX2d& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
cusp::coo_matrix< int, double, cusp::host_memory> projection( const aTopologyX3d& g_new, const aTopologyX3d& g_old) {
    return projection(g_new.grid(), g_old.grid());
}

///@}

}//namespace create
}//namespace dg
