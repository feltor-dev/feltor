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
 * @ingroup netcdf
 */
struct NcHyperslab
{
    /*! @brief <tt>{start}, {count}</tt>
     *
     * One dimensional slab
     * @param start the starting position of a 1d variable
     * @param count the count of a 1d variable
     */
    NcHyperslab( size_t start, size_t count = 1)
        : m_start( 1, start), m_count( 1, count)
    {
    }

    /*! @brief <tt>start, count</tt>
     *
     * \c start.size() dimensional slab
     * @param start specific start vector
     * @param count specific count vector (must have same size as \c start)
     */
    NcHyperslab(std::vector<size_t> start, std::vector<size_t> count)
        : m_start(start), m_count(count)
    {
        assert( start.size() == count.size());
    }

    /*! <tt>{0}, {file.get_dims_shape( file.get_var_dims(name))}</tt>
     *
     * Infer hyperslab from the dimensions of the variable
     * @param file Reference to the file object \c get_dims_shape and \c
     * get_var_dims are called
     * @param name Name of the variable to inquire
     */
    template<class File>
    NcHyperslab( const File& file, std::string name)
    {
        auto dims = file.get_var_dims(name);
        m_count = std::vector<size_t>( get_dims_shape( dims));
        m_start = std::vector<size_t>( dims.size(), 0);
    }

    /*! @brief <tt>{0 , data.size()}</tt>
     *
     * A one-dimensional slab
     * @tparam ContainerType ContainerType::size() must be callable
     * @param data explicitly set one dimensional count
     * @attention This only works for one-dimensional data
     */
    template<class ContainerType, std::enable_if_t< dg::is_vector_v<
        ContainerType, dg::SharedVectorTag>, bool> = true>
    NcHyperslab( const ContainerType& data)
    : NcHyperslab( 0, data.size())
    {
    }
    /*! @brief <tt>grid.start(), grid.count()</tt>
     *
     * @tparam Topology Topolgy::start() and *::count() need to return an
     * iterable that can be used to construct <tt> std::vector<size_t></tt>
     * @param grid explicitly set start and count
     * @copydoc hide_dimension_order
     */
    template<class Topology, std::enable_if_t<
        !dg::is_vector_v<Topology>, bool> = true>
    NcHyperslab( const Topology& grid)
    {
        auto ss = grid.start();
        auto cc = grid.count();
        m_start = std::vector<size_t>( ss.begin(), ss.end());
        m_count = std::vector<size_t>( cc.begin(), cc.end());
    }

    /// Same as <tt>NcHyperslab{ start0, 1, param}</tt>
    template<class T>
    NcHyperslab( size_t start0, const T& param)
          : NcHyperslab( start0, 1, param)
    {
    }

    /*! @brief <tt>{start0, NcHyperslab( param).start()}, {count0, NcHyperslab(param).count()}</tt>
     *
     * @tparam T <tt>NcHyperslab::NcHyperslab<T>(param)</tt> must be callable
     * @param start0 The start coordinate of the unlimited dimension
     * is prepended to \c NcHyperslab(param)
     * @param count0 The count coordinate of the unlimited dimension
     * is prepended to \c NcHyperslab(param)
     * @param param forwarded to \c NcHyperslab(param)
     */
    template<class T>
    NcHyperslab( size_t start0, size_t count0, const T& param)
          : NcHyperslab( param)
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
 * @ingroup netcdf
 */
struct MPINcHyperslab
{

    /*! @brief <tt>{local_start}, {local_count}</tt>
     *
     * One dimensional slab
     * @param local_start the starting position of a 1d variable
     * @param local_count the count of a 1d variable
     * @param comm communicator of ranks that hold relevant data
     */
    MPINcHyperslab( size_t local_start, size_t local_count, MPI_Comm comm)
    : m_slab( local_start, local_count), m_comm(comm)
    {
    }


    /*! @brief <tt>local_start, local_count, comm</tt>
     *
     * \c local_start.size() dimensional slab
     * @param local_start specific local start vector
     * @param local_count specific local count vector (must have same size as \c local_start)
     * @param comm communicator of ranks that hold relevant data
     */
    MPINcHyperslab(std::vector<size_t> local_start,
        std::vector<size_t> local_count, MPI_Comm comm)
    : m_slab( local_start, local_count), m_comm(comm)
    {
    }

    /*! @brief <tt>{local_start(data) , local_size(data), data.communicator()}</tt>
     *
     * Infer the local start and count by the size of the data vector.  The
     * local size is communicated to all processes in \c data.communicator()
     * and using the rank one can infer the starting position of the local data
     * chunk.  This assumes that the data is ordered by rank.
     * @attention This only works for one-dimensional data
     *
     * @tparam ContainerType \c ContainerType::size() and
     * \c ContainerType::communicator() must be callable
     * @param data explicitly set one dimensional start and count
     */
    template<class ContainerType, std::enable_if_t< dg::is_vector_v<
        ContainerType, dg::MPIVectorTag>, bool> = true>
    MPINcHyperslab( const ContainerType& data)
    : m_slab ( 0) // "default" construct
    {
        int count = data.size();
        MPI_Comm comm = data.communicator();
        int rank, size;
        MPI_Comm_rank( comm, &rank);
        MPI_Comm_size( comm, &size);
        std::vector<int> counts ( size);
        MPI_Allgather( &count, 1, MPI_INT, &counts[0], 1, MPI_INT, comm);
        size_t start = 0;
        for( int r=0; r<rank; r++)
            start += counts[r];

        *this = MPINcHyperslab{ start, (size_t)count, comm};
    }

    /*! @brief <tt>grid.start(), grid.count(), grid.communicator()</tt>
     *
     * @tparam MPITopology MPITopolgy::start() and *::count() need to return an
     * iterable that can be used to construct <tt>std::vector<size_t></tt>
     * MPITopology.communicator() needs to return the communicator of ranks
     * that hold data
     * @param grid explicitly set start and count and comm
     * @copydoc hide_dimension_order
     */
    template<class MPITopology, std::enable_if_t<!dg::is_vector_v<MPITopology>,
    bool> = true>
    MPINcHyperslab( const MPITopology& grid)
    : m_slab( grid), m_comm(grid.communicator())
    {
    }

    /// Same as <tt>MPINcHyperslab{ start0, 1, grid}</tt>
    template<class T>
    MPINcHyperslab( size_t start0, const T& param)
    : MPINcHyperslab( start0, 1, param)
    {
    }
    /*! @brief <tt>{start0, MPINcHyperslab( param).start()}, {count0, MPINcHyperslab(param).count(), MPINcHyperslab.communicator()}</tt>
     *
     * @tparam MPITopology MPITopolgy::start() and *::count() need to return an
     * iterable that can be used to construct <tt>std::vector<size_t></tt>
     * MPITopology.communicator() needs to return the communicator of ranks
     * that hold data
     * @param start0 The start coordinate of the unlimited dimension
     * is prepended to the grid.start()
     * @param count0 The count coordinate of the unlimited dimension
     * is prepended to the grid.count()
     * @param param explicitly set start and count and comm
     */
    template<class T>
    MPINcHyperslab( size_t start0, size_t count0, const T& param)
    : MPINcHyperslab( param)
    {
        m_slab.start().insert( m_slab.start().begin(), start0);
        m_slab.count().insert( m_slab.count().begin(), count0);
    }

    ///@copydoc NcHyperslab::ndim()
    unsigned ndim() const { return m_slab.ndim();}

    ///@copydoc NcHyperslab::start()
    const std::vector<size_t>& start() const { return m_slab.start();}
    ///@copydoc NcHyperslab::count()
    const std::vector<size_t>& count() const { return m_slab.count();}
    ///@copydoc NcHyperslab::start()
    std::vector<size_t>& start() { return m_slab.start();}
    ///@copydoc NcHyperslab::count()
    std::vector<size_t>& count() { return m_slab.count();}
    /// @return MPI Communicator specifying ranks that participate in reading/writing data
    MPI_Comm communicator() const { return m_comm;}
    ///@copydoc NcHyperslab::startp()
    const size_t* startp() const { return m_slab.startp();}
    ///@copydoc NcHyperslab::countp()
    const size_t* countp() const { return m_slab.countp();}
    private:
    NcHyperslab m_slab;
    MPI_Comm m_comm;
};

///@cond
namespace detail
{
/**
 * @brief Convert a global rank to a rank within a given communicator
 *
 * Essentially a utility wrapper around \c MPI_Group_translate_ranks
 * This function can be used to determine if the world_rank 0 (the "master" process)
 * belongs to the communicator of the calling process or not
 * @code
 * int local_master_rank = dg::mpi_comm_global2local_rank( comm, 0);
 * if ( local_master_rank == MPI_UNDEFINED)
 * // master process is not part of group
 * else
 * // do something
 * @endcode
 * @param comm The communicator / process group. Must be sub-group of or same as \c global_comm
 * @param global_rank a rank within \c global_comm
 * @param global_comm the communicator, which \c global_rank refers to
 * @return rank of \c global_rank, \c global_comm in \c comm,
 * \c MPI_UNDEFINED if \c global_rank is not part of \c comm
 */
inline int mpi_comm_global2local_rank( MPI_Comm comm, int global_rank = 0, MPI_Comm global_comm = MPI_COMM_WORLD )
{
    MPI_Group local_group, global_group;
    MPI_Comm_group(comm, &local_group);//local call
    MPI_Comm_group(MPI_COMM_WORLD, &global_group);//local call
    int local_root_rank;
    MPI_Group_translate_ranks(global_group, 1, &global_rank, local_group, &local_root_rank);
    return local_root_rank;
}
} // namespace detail
///@endcond
#endif

}//namespace file
}//namespace dg
