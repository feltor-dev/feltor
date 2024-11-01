#pragma once
// must be included after both derivatives.h, derivativesX.h and mpi_derivatives.h are included
// else there is a problem with ADL
// https://stackoverflow.com/questions/40875453/declare-function-after-template-defined

#include "derivatives.h"
#include "derivativesX.h"
#ifdef MPI_VERSION
#include "mpi_derivatives.h"
#endif

/*! @file
  @brief Convenience functions to create derivatives
  */
namespace dg{
namespace create{
///@addtogroup creation
///@{
/**
* @brief Create and assemble a host Matrix for the derivative in x
*
* A shortcut for
* @code
  dx( g, g.bcx(), dir);
* @endcode
* Take the boundary condition from the grid
* @param g Any grid
* @param dir The direction of the first derivative
*
* @return Host Matrix
*/
template<class Topology>
auto dx( const Topology& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}
/**
* @brief Create and assemble a host Matrix for the jump in x
*
* A shortcut for
* @code
  jumpX( g, g.bcx());
* @endcode
* Take the boundary condition from the grid
*
* @return Host Matrix
 * @copydoc hide_code_blas2_symv
*/
template<class Topology>
auto jumpX( const Topology& g)
{
    return jumpX( g, g.bcx());
}
/**
* @brief Create and assemble a host Matrix for the derivative in y
*
* A shortcut for
* @code
  dy( g, g.bcy(), dir);
* @endcode
* Take the boundary condition from the grid
* @param g Any grid
* @param dir The direction of the first derivative
*
* @return Host Matrix
*/
template<class Topology>
auto dy( const Topology& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}
/**
* @brief Create and assemble a host Matrix for the jump in y
*
* A shortcut for
* @code
  jumpY( g, g.bcy());
* @endcode
* Take the boundary condition from the grid
* @param g Any grid
*
* @return Host Matrix
*/
template<class Topology>
auto jumpY( const Topology& g)
{
    return jumpY( g, g.bcy());
}
/**
* @brief Create and assemble a host Matrix for the derivative in z
*
* A shortcut for
* @code
  dz( g, g.bcz(), dir);
* @endcode
* Take the boundary condition from the grid
* @param g Any grid
* @param dir The direction of the first derivative
*
* @return Host Matrix
*/
template<class Topology>
auto dz( const Topology& g, direction dir = centered)
{
    return dz( g, g.bcz(), dir);
}
/**
* @brief Create and assemble a host Matrix for the jump in z
*
* A shortcut for
* @code
  jumpZ( g, g.bcz());
* @endcode
* Take the boundary condition from the grid
* @param g Any grid
*
* @return Host Matrix
*/
template<class Topology>
auto jumpZ( const Topology& g)
{
    return jumpZ( g, g.bcz());
}
///@}
} //namespace create
}//namespace dg


