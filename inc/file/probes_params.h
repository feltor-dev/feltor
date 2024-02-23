#pragma once
#include "../dg/backend/typedefs.h"

namespace dg
{
namespace file
{

/**
 * @brief Parameter struct for probe values
 * @ingroup Cpp
 */
struct ProbesParams
{
    std::vector< dg::HVec> coords; //!< coordinates (only master thread holds them in MPI, other processes remain empty)
    std::vector<std::string> coords_names; //!< name of coordinates (same size as \c coords)
    std::string format; //!< format string for coords
    bool probes = false; //!< indicates if coords are empty or "probes" field did not exist (all MPI processes must agree)

    /**
     * @brief Check that coords all have same size and return that size
     *
     * @note throws if sizes are not equal
     * @return size of coords
     */
    unsigned get_coords_sizes( ) const
    {
        unsigned m_num_pins = coords[0].size();
        for( unsigned i=1; i<coords.size(); i++)
        {
            unsigned num_pins = coords[i].size();
            if( m_num_pins != num_pins)
                throw std::runtime_error( "Size of "+coords_names[i] +" probes array ("
                        +std::to_string(num_pins)+") does not match that of "+coords_names[0]+" ("
                        +std::to_string(m_num_pins)+")!");
        }
        return m_num_pins;
    }
};
}//namespace file
}//namespace dg
