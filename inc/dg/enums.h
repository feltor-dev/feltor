#pragma once

namespace dg
{
/**
 * @brief Switch between boundary conditions
 * 
 * @ingroup creation
 */
enum bc{ 
    PER = 0, //!< periodic boundaries
    DIR = 1, //!< homogeneous dirichlet boundaries
    DIR_NEU = 2, //!< Dirichlet on left, Neumann on right boundary
    NEU_DIR = 3, //!< Neumann on left, Dirichlet on right boundary
    NEU = 4 //!< Neumann on both boundaries
};
/**
 * @brief Switch between normalisations
 *
 * @ingroup creation
 */
enum norm{
    normed,   //!< indicates that output is properly normalized
    not_normed //!< indicates that normalisation weights (either T or V) are missing from output
};
/**
 * @brief Direction of a discrete derivative
 *
 * @ingroup creation
 */
enum direction{
    forward, //!< forward derivative
    backward, //!< backward derivative
    centered //!< centered derivative
};
/**
 * @brief Coordinate system
 *
 * @ingroup creation
 */
enum system{
    cartesian, //!< cartesian coordinate system
    cylindrical //!< cylindrical coordinate system
};
///@addtogroup creation
///@{
/**
 * @brief Switch between x-space and l-space
 * DEPRECATED
 */
enum space {
    XSPACE, //!< indicates, that the given matrix operates on x-space values
    LSPACE  //!< indicates, that the given matrix operates on l-space values
};
///@}
}//namespace dg
