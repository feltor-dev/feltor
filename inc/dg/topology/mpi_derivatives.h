#pragma once

#include "dg/backend/sparseblockmat.h"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/typedefs.h"
#include "functions.h"
#include "derivatives.h"
#include "mpi_grid.h"

namespace dg{

namespace create{

///@cond
namespace detail{


// Also used by fast_interpolation
template<class real_type, size_t Nd>
RealGrid<real_type,Nd> local_global_grid( unsigned coord,
    const aRealMPITopology<real_type, Nd>& g)
{
    // global grid in coord, local grids else
    std::array<RealGrid<real_type,1>,Nd> axes;
    for( unsigned u=0; u<Nd; u++)
        axes[u] = g.local().axis(u);
    axes[coord] = g.global().axis(coord);
    return RealGrid<real_type,Nd>{axes};
}

} //namespace detail

///@endcond
///@addtogroup creation
///@{
//


/**
* @brief Create a derivative along the given coordinate for mpi
*
* @param coord the coordinate along which to derive
* @param g A mpi grid
* @param bc boundary condition
* @param dir centered, forward or backward
*
* @return  A mpi matrix
*/
template<class real_type, size_t Nd>
dg::MHMatrix_t<real_type> derivative( unsigned coord,
    const aRealMPITopology<real_type, Nd>& g, dg::bc bc, direction dir = centered)
{
    return dg::make_mpi_sparseblockmat( dg::create::derivative( coord,
            detail::local_global_grid(coord, g), bc, dir),
            g.axis(coord), g.axis(coord));
}

/**
 * @brief Create a jump matrix along given coordinate for mpi
 *
 * @param coord the coordinate along which to jump
 * @param g The mpi grid on which to create jump
 * @param bc The boundary condition
 *
 * @return A mpi matrix
 */
template<class real_type, size_t Nd>
dg::MHMatrix_t<real_type> jump( unsigned coord,
    const aRealMPITopology<real_type, Nd>& g, dg::bc bc)
{
    return dg::make_mpi_sparseblockmat( dg::create::jump( coord,
            detail::local_global_grid(coord, g), bc),
            g.axis(coord), g.axis(coord));
}

///@}


} //namespace create
} //namespace dg
