#pragma once

#include "grid.h"
#include "dx.h"

/*! @file
  @brief Convenience functions to create 2D derivatives
  */

namespace dg{

    // Check derivativesT.h for more overloads

/**
 * @brief Contains functions used for matrix creation
 */
namespace create{
///@cond
namespace detail
{
template<class real_type, size_t Nd>
void update_left_right( unsigned coord, EllSparseBlockMat<real_type>& mat, const aRealTopology<real_type,Nd>& g)
{
    // Also used by fast_interpolation ...
    unsigned right_size = 1, left_size = 1;
    for( unsigned u=0; u<coord; u++)
        right_size*= g.shape(u);
    for( unsigned u=coord+1; u<Nd; u++)
        left_size *= g.shape(u);
    mat.set_right_size( right_size);
    mat.set_left_size( left_size);
}
}
///@endcond

///@addtogroup creation
///@{
/**
 * @brief Create a derivative along given coordinate
 *
 * @param coord the coordinate along which to derive
 * @param g The grid on which to create derivative
 * @param bc The boundary condition
 * @param dir centered, forward or backward
 *
 * @return A host matrix
 */
template<class real_type, size_t Nd>
EllSparseBlockMat<real_type> derivative( unsigned coord,
    const aRealTopology<real_type, Nd>& g, dg::bc bc, direction dir = centered)
{
    if( coord >= Nd)
        throw Error( Message(_ping_)<<"coord>=Nd not allowed! You typed: "<<coord<<" while Nd is "<<Nd);
    EllSparseBlockMat<real_type> dd = dx_normed(
            g.n(coord), g.N(coord), g.h(coord), bc, dir);
    detail::update_left_right( coord, dd, g);
    return dd;
}

/**
 * @brief Create a jump matrix along given coordinate
 *
 * @param coord the coordinate along which to jump
 * @param g The grid on which to create jump
 * @param bc The boundary condition
 *
 * @return A host matrix
 */
template<class real_type, size_t Nd>
EllSparseBlockMat<real_type> jump( unsigned coord,
    const aRealTopology<real_type, Nd>& g, dg::bc bc)
{
    if( coord >= Nd)
        throw Error( Message(_ping_)<<"coord>=Nd not allowed! You typed: "<<coord<<" while Nd is "<<Nd);
    EllSparseBlockMat<real_type> dd = jump(
            g.n(coord), g.N(coord), g.h(coord), bc);
    detail::update_left_right( coord, dd, g);
    return dd;
}


///@}

} //namespace create

} //namespace dg

