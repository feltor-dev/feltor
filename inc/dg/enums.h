#pragma once
#include <string>
#include "backend/exceptions.h"

/*! @file
  @brief enums
  */

namespace dg
{
///@addtogroup creation
///@{

///@brief Switch between boundary conditions
enum bc{
    PER = 0, //!< periodic boundaries
    DIR = 1, //!< homogeneous dirichlet boundaries
    DIR_NEU = 2, //!< Dirichlet on left, Neumann on right boundary
    NEU_DIR = 3, //!< Neumann on left, Dirichlet on right boundary
    NEU = 4 //!< Neumann on both boundaries
};


/**
 * @brief write a string describing boundary condition to an output stream
 *
 * converts
 * - dg::PER to PERIODIC
 * - dg::DIR to DIRICHLET
 * - dg::NEU to NEUMANN
 * - dg::NEU_DIR to NEU_DIR
 * - dg::DIR_NEU to DIR_NEU
 * - or "Not specified!!"
 * @param bcx the boundary condition
 * @return a string
 */
inline std::string bc2str( bc bcx)
{
    std::string s;
    switch(bcx)
    {
        case(dg::PER): s = "PERIODIC"; break;
        case(dg::DIR): s = "DIRICHLET"; break;
        case(dg::NEU): s = "NEUMANN"; break;
        case(dg::DIR_NEU): s = "DIR_NEU"; break;
        case(dg::NEU_DIR): s = "NEU_DIR"; break;
        default: s = "Not specified!!";
    }
    return s;
}

/**
 * @brief convert a string to a bc
 *
 * converts
 * - PER, per, periodic, PERIODIC and Periodic to dg::PER
 * - DIR, dir, dirichlet, DIRICHLET and Dirichlet to dg::DIR
 * - NEU, neu, neumann, NEUMANN and Neumann to dg::NEU
 * - NEU_DIR and neu_dir to dg::NEU_DIR
 * - DIR_NEU and dir_neu to dg::DIR_NEU
 *
 * @param s the input string
 * @return a valid boundary condition
 * \throw std::runtime_error if string doesn't match any of the above
 */
inline bc str2bc( std::string s)
{
    if( s=="PER"||s=="per"||s=="periodic"||s=="Periodic" || s == "PERIODIC")
        return PER;
    if( s=="DIR"||s=="dir"||s=="dirichlet"||s=="Dirichlet" || s == "DIRICHLET")
        return DIR;
    if( s=="NEU"||s=="neu"||s=="neumann"||s=="Neumann" || s=="NEUMANN")
        return NEU;
    if( s=="NEU_DIR"||s=="neu_dir" )
        return NEU_DIR;
    if( s=="DIR_NEU"||s=="dir_neu" )
        return DIR_NEU;
    throw std::runtime_error( "Boundary condition '"+s+"' not recognized!");
}

/**
 * @brief invert boundary condition
 *
 * @param bound boundary condition to invert
 * @return NEU for DIR, DIR for NEU, NEU_DIR for DIR_NEU, DIR_NEU for NEU_DIR and PER for PER
 */
inline bc inverse( bc bound)
{
    if( bound == DIR) return NEU;
    if( bound == NEU) return DIR;
    if( bound == DIR_NEU) return NEU_DIR;
    if( bound == NEU_DIR) return DIR_NEU;
    return PER;
}

///@brief Direction of a discrete derivative
enum direction{
    forward, //!< forward derivative (cell to the right and current cell)
    backward, //!< backward derivative (cell to the left and current cell)
    centered //!< centered derivative (cell to the left and right and current cell)
};


/**
 * @brief convert a string to a direction
 *
 * converts
 * - "forward" to forward
 * - "backward" to backward
 * - "centered" to centered
 *
 * @param s the input string
 * @return a valid direction
 * \throw std::runtime_error if string doesn't match any of the above
 */
inline direction str2direction( std::string s)
{
    if( "forward" == s)
        return forward;
    if( "backward" == s)
        return backward;
    if( "centered" == s)
        return centered;
    throw std::runtime_error( "Direction '"+s+"' not recognized!");
}
/**
 * @brief convert a direciton to string
 *
 * converts
 * - forward to "forward"
 * - backward to "backward"
 * - centered to "centered"
 *
 * @param dir the input direction
 * @return a string
 */
inline std::string direction2str( enum direction dir)
{
    std::string s;
    switch(dir)
    {
        case(dg::forward): s = "forward"; break;
        case(dg::backward): s = "backward"; break;
        case(dg::centered): s = "centered"; break;
        default: s = "Not specified!!";
    }
    return s;
}

/**
 * @brief invert direction
 *
 * @param dir direction to invert
 * @return backward for forward, forward for backward, centered for centered
 */
inline direction inverse( direction dir)
{
    if( dir == forward) return backward;
    if( dir == backward) return forward;
    return centered;
}

///@brief Space of DG coefficients
enum space{
    lspace, //!< DG Polynomial space "modal values"
    xspace //!< Configuration space "nodal values"
};

// With arbitrary dimensionality enabled maybe we can rethink if
// enum coo { x = 0, y = 1, ...} is a good idea? And how and where to use it?
///@brief 2d coordinates
enum class coo2d : char
{
    x = 'x', //!< x direction
    y = 'y', //!< y direction
};
///@brief 3d coordinates
enum class coo3d : char
{
    x = 'x', //!< x direction
    y = 'y', //!< y direction
    z = 'z', //!< z direction
};

///@}
}//namespace dg
