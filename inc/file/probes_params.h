#pragma once
#include <vector>

namespace dg
{
namespace file
{

/**
 * @brief Parameter struct for probe values
 * @ingroup probes
 * @sa \c dg::file::parse_probes and \c dg::file::Probes
 */
struct ProbesParams
{
    /**
     * @brief Coordinates list
     *
     * - coords[0] (if present) is the list of x-coordinates
     * - coords[1] (if present) is the list of y-coordinates
     * - coords[2] (if present) is the list of z-coordinates
     * .
     *
     * All lists must have same size. \c coords will be passed to \c
     * dg::create::interpolation function in the \c dg::file::Probes class.
     * @note In MPI all coordinates from all threads will be interpolated but
     * only the coordinates that the master thread is holding will be written
     * to file. Also note that \c parse_probes only reads coords on the master
     * thread the other ranks remain empty
     */
    std::vector< std::vector<double>> coords;
    std::vector< std::string> coords_names; //!< Name of coordinates (must have same size as \c coords)
    /**
     * @brief Optional format string for \c coords.
     *
     * The \c Probes class will write this as a group attribute
     */
    std::string format;
    /**
     * @brief Indicate existence of probes
     *
     * If \c false then no probes exist. The \c dg::file::Probes class
     * will write no probes in that case and all its member functions return immediately.
     *
     * The \c dg::file::parse_probes will set this flag if "probes" field does not exist in input file.
     *
     * @note All MPI processes must agree on the value of \c probes
     */
    bool probes = false;

    /**
     * @brief Check that coords all have same size and return that size
     *
     * @note throws if sizes are not equal
     * @return size of coords
     */
    unsigned get_coords_sizes( ) const
    {
        unsigned m_num_pins = (unsigned)coords[0].size();
        for( unsigned i=1; i<coords.size(); i++)
        {
            unsigned num_pins = (unsigned)coords[i].size();
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
