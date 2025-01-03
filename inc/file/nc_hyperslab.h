#pragma once
#include <vector>
#include <netcdf.h>
namespace dg{
namespace file{
/*!@brief A NetCDF Hyperslab
 *
 * https://docs.unidata.ucar.edu/netcdf-c/4.9.2/programming_notes.html#specify_hyperslab
 */
struct NcHyperslab
{
    NcHyperslab(std::vector<size_t> start, std::vector<size_t> count)
        : m_start(start), m_count(count)
    {
        assert( start.size() == count.size());
    }
    template<class Topology>
    NcHyperslab( const Topology& grid, bool reverse = true)
    {
        auto ss = grid.start();
        auto cc = grid.count();
        m_start = std::vector<size_t>( ss.begin(), ss.end());
        m_count = std::vector<size_t>( cc.begin(), cc.end());
        if( reverse)
        {
            std::reverse( m_start.begin(), m_start.end());
            std::reverse( m_count.begin(), m_count.end());
        }
    }

    template<class Topology>
    NcHyperslab( unsigned slice, const Topology& grid, bool reverse = true)
          : NcHyperslab( grid, reverse)
    {
        m_start.insert( m_start.begin(), slice);
        m_count.insert( m_count.begin(), 1);
    }
    const unsigned ndims() const { return m_start.size();}

    const std::vector<size_t>& start() const { return m_start;}
    const std::vector<size_t>& count() const { return m_count;}
    const size_t* startp() const { return &m_start[0];}
    const size_t* countp() const { return &m_count[0];}
    private:
    std::vector<size_t> m_start, m_count;
};

}//namespace file
}//namespace dg
