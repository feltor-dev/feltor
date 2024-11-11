#pragma once

#include <cassert>

#include "gridX.h"
#include "dx.h"
#include "weightsX.h"
#include "dg/backend/sparseblockmat.h"

/*! @file
  @brief Simple 1d derivatives on X-point topology
  */
namespace dg
{
namespace create
{
///@addtogroup lowlevel
///@{
/**
* @brief Create and assemble a host Matrix for the derivative in 1d
*
* @ingroup creation
* @param g 1D grid with X-point topology
* @param bcx boundary condition
* @param dir The direction of the first derivative
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> dx( const RealGridX1d<real_type>& g, bc bcx, direction dir = centered)
{
    auto grid = g.grid();
    if( g.outer_N() == 0) return dx_normed( grid.n(), grid.N(), grid.h(), dg::PER, dir);
    EllSparseBlockMat<real_type> DX = dx_normed( grid.n(), grid.N(), grid.h(), bcx, dir);
    for( int i=0; i<DX.blocks_per_line; i++)
    {
        if( DX.cols_idx[DX.blocks_per_line*(g.outer_N()-1)+i] == (int)g.outer_N())
            DX.cols_idx[DX.blocks_per_line*(g.outer_N()-1)+i] += g.inner_N();
        if( DX.cols_idx[DX.blocks_per_line*(g.outer_N())+i] == (int)g.outer_N()-1)
            DX.cols_idx[DX.blocks_per_line*(g.outer_N())+i] += g.inner_N();
        if( DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N()-1)+i] == (int)(g.N()-g.outer_N()))
            DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N()-1)+i] -= g.inner_N();
        if( DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N())+i] == (int)(g.N()-g.outer_N()-1))
            DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N())+i] -= g.inner_N();
    }
    return DX;
}

/**
* @brief Create and assemble a host Matrix for the derivative in 1d
*
* Take the boundary condition from the grid
* @ingroup creation
* @param g 1D grid with X-point topology
* @param dir The direction of the first derivative
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> dx( const RealGridX1d<real_type>& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}
/**
* @brief Create and assemble a host Matrix for the jump in 1d
*
* @ingroup creation
* @param g 1D grid with X-point topology
* @param bcx boundary condition
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> jump( const RealGridX1d<real_type>& g, bc bcx)
{
    if( g.outer_N() == 0) return jump( g.n(), g.N(), g.h(), dg::PER);
    EllSparseBlockMat<real_type> J = jump( g.n(),g.N(),g.h(), bcx);
    for( int i=0; i<J.blocks_per_line; i++)
    {
        if( J.cols_idx[J.blocks_per_line*(g.outer_N()-1)+i] == (int)g.outer_N())
            J.cols_idx[J.blocks_per_line*(g.outer_N()-1)+i] += g.inner_N();
        if( J.cols_idx[J.blocks_per_line*(g.outer_N())+i] == (int)g.outer_N()-1)
            J.cols_idx[J.blocks_per_line*(g.outer_N())+i] += g.inner_N();
        if( J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N()-1)+i] == (int)(g.N()-g.outer_N()))
            J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N()-1)+i] -= g.inner_N();
        if( J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N())+i] == (int)(g.N()-g.outer_N()-1))
            J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N())+i] -= g.inner_N();
    }
    return J;
}
/**
* @brief Create and assemble a host Matrix for the jump in 1d
*
* Take the boundary condition from the grid
* @ingroup creation
* @param g 1D grid with X-point topology
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> jump( const RealGridX1d<real_type>& g)
{
    return jump( g, g.bcx());
}

///@}
} //namespace create
} //namespace dg

