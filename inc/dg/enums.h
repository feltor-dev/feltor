#pragma once
#include <string>
#include "backend/exceptions.h"

/*! @file 
  @brief enums
  */

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
 * @ingroup creation
 */
std::string bc2str( bc bcx)
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
 * @ingroup creation
 */
bc str2bc( std::string s)
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
    throw std::runtime_error( "No matching boundary condition!");
}

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
}//namespace dg
