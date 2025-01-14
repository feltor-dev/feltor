#pragma once
#include <vector>
#include <netcdf.h>

namespace dg{
namespace file{
/*! @class hide_dimension_order
 *
 * @attention When writing variables, NetCDF-C always assumes that the last
 * dimension of the NetCDF variable varies fastest in the given array.  This is
 * in contrast to the default behaviour of our \c dg::evaluate function, which
 * produces vectors where the first dimension of the given grid varies
 * fastest. Thus, when defining variable dimensions the dimension name of the
 * first grid dimension needs to come last.
 * @note The unlimited dimension, if present, must be the first dimension.
 */

/*!@brief A NetCDF Hyperslab for \c SerialNcFile
 *
 * This is how to
 * @sa <a href="https://docs.unidata.ucar.edu/netcdf-c/4.9.2/programming_notes.html">specify a hyperslab</a>
 *
 * @copydoc hide_dimension_order
 * @ingroup Cpp
 */
struct NcHyperslab
{
    /*! @brief <tt> {slice}, {count}</tt>
     *
     * @param start the starting position of a 1d variable
     * @param count the count of a 1d variable
     */
    NcHyperslab( size_t start, size_t count = 1)
        : m_start( 1, start), m_count( 1, count)
    {
    }

    /*! @brief <tt> start, count</tt>
     *
     * @param start specific start vector
     * @param count specific count vector
     */
    NcHyperslab(std::vector<size_t> start, std::vector<size_t> count)
        : m_start(start), m_count(count)
    {
        assert( start.size() == count.size());
    }

    /*! @brief <tt> grid.start(), grid.count()</tt>
     *
     * @tparam Topology Topolgy::start() and *::count() need to return an
     * iterable that can be used to construct <tt> std::vector<size_t></tt>
     * @param grid explicitly set start and count
     */
    template<class Topology>
    NcHyperslab( const Topology& grid)
    {
        auto ss = grid.start();
        auto cc = grid.count();
        m_start = std::vector<size_t>( ss.begin(), ss.end());
        m_count = std::vector<size_t>( cc.begin(), cc.end());
    }

    /// Same as <tt> NcHyperslab{ start0, 1, grid}</tt>
    template<class Topology>
    NcHyperslab( size_t start0, const Topology& grid)
          : NcHyperslab( start0, 1, grid)
    {
    }
    /*! @brief <tt> {start0, grid.start()}, {count0, grid.count()}</tt>
     *
     * @tparam Topology Topolgy::start() and *::count() need to return an
     * iterable that can be used to construct \c std::vector<size_t>
     * @param start0 The start coordinate of the unlimited dimension
     * is prepended to the grid.start()
     * @param count0 The count coordinate of the unlimited dimension
     * is prepended to the grid.count()
     * @param grid explicitly set start() and count()
     */
    template<class Topology>
    NcHyperslab( size_t start0, size_t count0, const Topology& grid)
          : NcHyperslab( grid)
    {
        m_start.insert( m_start.begin(), start0);
        m_count.insert( m_count.begin(), count0);
    }

    /// @return Size of start and count vectors
    unsigned ndim() const { return m_start.size();}

    /// @return start vector
    const std::vector<size_t>& start() const { return m_start;}
    /// @return count vector
    const std::vector<size_t>& count() const { return m_count;}
    /// @return start vector
    std::vector<size_t>& start() { return m_start;}
    /// @return count vector
    std::vector<size_t>& count() { return m_count;}
    /// @return pointer to first element of start
    const size_t* startp() const { return &m_start[0];}
    /// @return pointer to first element of count
    const size_t* countp() const { return &m_count[0];}
    private:
    std::vector<size_t> m_start, m_count;
};

#ifdef MPI_VERSION
/*!@brief A NetCDF Hyperslab for \c MPINcFile
 *
 * In MPI the data of arrays is usually distributed among processes and
 * each process needs to know where their chunk of data needs to be written
 * in the global array. It is also possible that fewer ranks than are present
 * in \c file.communicator() actually hold relevant data.
 *
 * @copydetails NcHyperslab
 * @ingroup Cpp
 */
struct MPINcHyperslab
{

    /*! @brief <tt> start, count</tt>
     *
     * @param local_start specific local start vector
     * @param local_count specific local count vector
     * @param comm communicator of ranks that hold relevant data
     */
    MPINcHyperslab(std::vector<size_t> local_start,
        std::vector<size_t> local_count, MPI_Comm comm)
    : m_slab( local_start, local_count), m_comm(comm)
    {
    }

    /*! @brief <tt> grid.start(), grid.count(), grid.communicator()</tt>
     *
     * @tparam MPITopology MPITopolgy::start() and *::count() need to return an
     * iterable that can be used to construct <tt> std::vector<size_t></tt>
     * MPITopology.communicator() needs to return the communicator of ranks
     * that hold data
     * @param grid explicitly set start and count and comm
     */
    template<class MPITopology>
    MPINcHyperslab( const MPITopology& grid)
    : m_slab( grid), m_comm(grid.communicator())
    {
    }

    /// Same as <tt> MPINcHyperslab{ start0, 1, grid}</tt>
    template<class MPITopology>
    MPINcHyperslab( size_t start0, const MPITopology& grid)
    : MPINcHyperslab( start0, 1, grid)
    {
    }
    /*! @brief <tt> {start0, grid.start()}, {count0, grid.count()},
     * grid.communicator()</tt>
     *
     * @tparam MPITopology MPITopolgy::start() and *::count() need to return an
     * iterable that can be used to construct <tt> std::vector<size_t></tt>
     * MPITopology.communicator() needs to return the communicator of ranks
     * that hold data
     * @param start0 The start coordinate of the unlimited dimension
     * is prepended to the grid.start()
     * @param count0 The count coordinate of the unlimited dimension
     * is prepended to the grid.count()
     * @param grid explicitly set start and count and comm
     */
    template<class MPITopology>
    MPINcHyperslab( size_t start0, size_t count0, const MPITopology& grid)
    : m_slab( start0, count0, grid), m_comm( grid.communicator())
    {
    }

    ///@copdoc NcHyperslab::ndim()
    unsigned ndim() const { return m_slab.ndim();}

    ///@copdoc NcHyperslab::start()
    const std::vector<size_t>& start() const { return m_slab.start();}
    ///@copdoc NcHyperslab::count()
    const std::vector<size_t>& count() const { return m_slab.count();}
    ///@copdoc NcHyperslab::start()
    std::vector<size_t>& start() { return m_slab.start();}
    ///@copdoc NcHyperslab::count()
    std::vector<size_t>& count() { return m_slab.count();}
    /// @return MPI Communicator specifying participating ranks
    MPI_Comm communicator() const { return m_comm;}
    ///@copdoc NcHyperslab::startp()
    const size_t* startp() const { return m_slab.startp();}
    ///@copdoc NcHyperslab::countp()
    const size_t* countp() const { return m_slab.countp();}
    private:
    NcHyperslab m_slab;
    MPI_Comm m_comm;
};
#endif

}//namespace file
}//namespace dg
