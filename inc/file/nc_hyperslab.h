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
    NcHyperslab( size_t slice)
        : m_start( 1, slice), m_count( 1, 1)
    {}

    NcHyperslab(std::vector<size_t> count)
        : m_start( count.size(), 0), m_count( count)
    {}
    NcHyperslab(std::vector<size_t> start, std::vector<size_t> count)
        : m_start(start), m_count(count)
    {
        assert( start.size() == count.size());
    }

    template<class Topology>
    NcHyperslab( const Topology& grid, bool reverse)
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
    NcHyperslab( size_t slice, const Topology& grid, bool reverse)
          : NcHyperslab( grid, reverse)
    {
        m_start.insert( m_start.begin(), slice);
        m_count.insert( m_count.begin(), 1);
    }
    unsigned ndim() const { return m_start.size();}

    const std::vector<size_t>& start() const { return m_start;}
    const std::vector<size_t>& count() const { return m_count;}
    const size_t* startp() const { return &m_start[0];}
    const size_t* countp() const { return &m_count[0];}
    private:
    std::vector<size_t> m_start, m_count;
};

#ifdef MPI_VERSION
struct MPINcHyperslab
{

    MPINcHyperslab(std::vector<size_t> local_start,
        std::vector<size_t> local_count, MPI_Comm comm)
    : m_slab( local_start, local_count), m_comm(comm)
    {
    }

    template<class Topology>
    MPINcHyperslab( const Topology& grid, bool reverse)
    : m_slab( grid, reverse), m_comm(grid.communicator())
    {
    }

    template<class Topology>
    MPINcHyperslab( size_t slice, const Topology& grid, bool reverse)
    : m_slab( slice, grid, reverse), m_comm(grid.communicator())
    {
    }
    unsigned ndim() const { return m_slab.ndim();}

    const std::vector<size_t>& start() const { return m_slab.start();}
    const std::vector<size_t>& count() const { return m_slab.count();}
    MPI_Comm communicator() const { return m_comm;}
    const size_t* startp() const { return m_slab.startp();}
    const size_t* countp() const { return m_slab.countp();}
    private:
    NcHyperslab m_slab;
    MPI_Comm m_comm;
};
#endif

}//namespace file
}//namespace dg
