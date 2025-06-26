#pragma once

#include <exception>
#include <netcdf.h>
#include "dg/topology/grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "dg/topology/mpi_grid.h"
#include "dg/backend/mpi_init.h"
#endif //MPI_VERSION
#include "nc_error.h"
#include "nc_hyperslab.h"

/*!@file
 *
 * Define variable writers
 */

namespace dg
{
namespace file
{
/**
 * @addtogroup legacy
 * @{
 * @class hide_tparam_host_vector
 * @tparam host_vector For shared Topology: Type with \c data() member that
 * returns pointer to first element in CPU (host) adress space, meaning it
 * cannot be a GPU vector. For MPI Topology: must be \c MPI_Vector
 * \c host_vector::value_type must match data type of variable in file.
 *
 * @class hide_comment_slice
 * @note In NetCDF all variables that share an unlimited dimension are
 * considered to have the same size in that dimension. In fact, the size of the
 * unlimited dimension is the maximum of the sizes of all the variables sharing
 * that unlimited dimension. All variables are artificially filled up with
 * filler Values to match that maximum size. It is entirely possible to skip
 * writing data for variables for some times. It is also possible to write data
 * to unlimited variables at \c slice>=size (in which case all variables
 * sharing the unlimited dimension will increase in size) but it is not
 * possible to read data at \c slice>=size. It is the user's responsibility
 * to manage the slice value across variables.
 *
 * @class hide_master_comment
 * @note The "master" thread is assumed to be the process with \c rank==0
 * in \c MPI_COMM_WORLD.  The \c MPI_COMM_WORLD rank of a process is usually the
 * same in a Cartesian communicator of the same size but is not guaranteed. So
 * always check \c MPI_COMM_WORLD ranks for file write operations.
 *
 * @class hide_parallel_param
 * @param parallel This parameter is ignored in the serial version.
 * In the MPI version this parameter indicates whether each process
 * reads/writes to the file independently in parallel (\c true)
 * or each process funnels its data through the master rank (\c false),
 * which involves communication but may be faster than the former method.
 *
 * @note
 * In an MPI environment
 *  - all processes must call this function,
 *  - processes that do not belong to the same communicator as the master process
 * return immediately
 *  .
 * @note In an MPI program it may happen that the data to read/write is partitioned among
 * a process group smaller than \c MPI_COMM_WORLD, e.g. when reading/writing a 2d slice of a 3d vector.
 * In this example case \c grid.communicator() is only 2d not 3d. Remember that **only
 * the group containing the master process reads/writes its data to the file**, while all other processes
 * immediately return.
 * There are two ways to relyable read/write the data in such a case:
 *  - Manually assemble the data on the master process and construct an MPI
 *  grid with a Cartesian communicator containing only one process (using e.g. \c MPI_Comm_split on \c MPI_COMM_WORLD followed by \c MPI_Cart_create)
 *  - Manually assemble the data on the MPI group that contains the master process (cf \c MPI_Cart_sub)
 *  .
 * @class hide_parallel_write
 * @attention With the serial NetCDF library only a single "master" process can **write** in a
 * NetCDF file (creation, defining dimension ids, variables ids, writing etc).
 * Thus, in an MPI program
 *  - \c parallel should be \c false
 *  - the program links to **serial NetCDF and hdf5**
 *  - only the master thread needs to know the \c ncid, variable or dimension names, the slice to write etc.
 *  .
 * There is a parallel NetCDF library where all processes can have write
 * access in parallel. In this case
 *  - \c parallel should be \c true
 *  - the program links to **parallel NetCDF and hdf5**
 *  - the file must be opened with the \c NC_MPIIO flag from the \c NetCDF_par.h header and the
 * variable be marked with \c NC_COLLECTIVE access
 *  - all threads need to know the \c ncid, variable and dimension names, the slice to write etc.
 *  .
 * Note that serious performance penalties have been
 * observed on some platforms for parallel writing NetCDF.
 *
 * @class hide_parallel_read
 * @note In contrast to writing, reading a NetCDF-4 file can always be done in parallel
 *  See https://docs.h5py.org/en/stable/mpi.html So all processes in MPI
 *  **can** open a file, get variable ids and subsequently read it, etc. even if only
 *  serial NetCDF is used. The default for \c parallel is always \c true in
 *  which case all processes **must** have previously opened the file and
 *  inquire e.g. the varid
 */

///@cond
namespace detail
{

template<class T>
inline int put_var_T( int /*ncid*/, int /*varid*/, const T* /*data*/)
{
    assert( false && "Type not supported!\n" );
    return NC_EBADTYPE;
}
template<>
inline int put_var_T<float>( int ncid, int varid, const float* data){
    return nc_put_var_float( ncid, varid, data);
}
template<>
inline int put_var_T<double>( int ncid, int varid, const double* data){
    return nc_put_var_double( ncid, varid, data);
}
template<>
inline int put_var_T<int>( int ncid, int varid, const int* data){
    return nc_put_var_int( ncid, varid, data);
}
template<>
inline int put_var_T<unsigned>( int ncid, int varid, const unsigned* data){
    return nc_put_var_uint( ncid, varid, data);
}
template<class T>
inline int put_vara_T( int /*ncid*/, int /*varid*/, const size_t* /*startp*/, const size_t* /*countp*/, const T* /*data*/)
{
    assert( false && "Type not supported!\n" );
    return NC_EBADTYPE;
}
template<>
inline int put_vara_T<float>( int ncid, int varid, const size_t* startp, const size_t* countp, const float* data){
    return nc_put_vara_float( ncid, varid, startp, countp, data);
}
template<>
inline int put_vara_T<double>( int ncid, int varid, const size_t* startp, const size_t* countp, const double* data){
    return nc_put_vara_double( ncid, varid, startp, countp, data);
}
template<>
inline int put_vara_T<int>( int ncid, int varid, const size_t* startp, const size_t* countp, const int* data){
    return nc_put_vara_int( ncid, varid, startp, countp, data);
}
template<>
inline int put_vara_T<unsigned>( int ncid, int varid, const size_t* startp, const size_t* countp, const unsigned* data){
    return nc_put_vara_uint( ncid, varid, startp, countp, data);
}
#ifdef MPI_VERSION
template<class host_vector>
void put_vara_detail(int ncid, int varid,
        const MPINcHyperslab& slab,
        const host_vector& data,
        thrust::host_vector<dg::get_value_type<host_vector>>& receive,
        MPI_Comm global_comm = MPI_COMM_WORLD
        )
{
    MPI_Comm comm = slab.communicator();
    // we need to identify the global root rank within the groups and mark the
    // entire group
    int local_root_rank = dg::file::detail::mpi_comm_global2local_rank(comm, 0,
        global_comm);
    if (local_root_rank == MPI_UNDEFINED)
        return;
    unsigned ndim = slab.ndim(); // same on all processes
    int rank, size;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);

    // Send start and count vectors to root
    std::vector<size_t> r_start( rank == local_root_rank ? size * ndim : 0);
    std::vector<size_t> r_count( rank == local_root_rank ? size * ndim : 0);
    MPI_Gather( slab.startp(), ndim, dg::getMPIDataType<size_t>(),
                &r_start[0], ndim, dg::getMPIDataType<size_t>(),
                local_root_rank, comm);
    MPI_Gather( slab.countp(), ndim, dg::getMPIDataType<size_t>(),
                &r_count[0], ndim, dg::getMPIDataType<size_t>(),
                local_root_rank, comm);

    MPI_Datatype mpitype = dg::getMPIDataType<get_value_type<host_vector>>();
    int err = NC_NOERR; // store error flag
    if( rank == local_root_rank )
    {
        // Sanity check
        int ndims;
        int e = nc_inq_varndims( ncid, varid, &ndims);
        if( not e)
            err = e;
        if( (unsigned)ndims != slab.ndim())
            err = 1001; // Our own error code

        std::vector<size_t> sizes( size, 1);
        for( int r = 0 ; r < size; r++)
            for( unsigned u=0; u<ndim; u++)
                sizes[r]*= r_count[r*ndim + u];

        // host_vector could be a View
        unsigned max_size = *std::max_element( sizes.begin(), sizes.end());
        receive.resize( max_size);
        for( int r=0; r<size; r++)
        {
            if(r!=rank)
            {
                MPI_Status status;
                MPI_Recv( receive.data(), (int)sizes[r], mpitype,
                      r, r, comm, &status);
                int e = detail::put_vara_T( ncid, varid, &r_start[r*ndim],
                        &r_count[r*ndim], receive.data()); // write received
                if( not e) err = e;

            }
            else // write own data
            {
                int e = detail::put_vara_T( ncid, varid, slab.startp(),
                        slab.countp(), thrust::raw_pointer_cast(data.data()));
                if( not e) err = e;
            }
        }
    }
    else
    {
        size_t num = 1;
        for( unsigned u=0; u<ndim; u++)
            num*= slab.count()[u];
        MPI_Send( thrust::raw_pointer_cast(data.data()), num, mpitype,
            local_root_rank, rank, comm);
    }
    MPI_Bcast( &err, 1, dg::getMPIDataType<int>(), local_root_rank, comm);
    if( err)
        throw NC_Error( err);
    return;
}

// all comms must be same size
template<class host_vector, class MPITopology>
void put_vara_detail(int ncid, int varid, unsigned slice,
        const MPITopology& grid, const MPI_Vector<host_vector>& data,
        bool vara, bool parallel = false)
{
    MPINcHyperslab slab( grid);
    if( vara)
        slab = MPINcHyperslab( slice, grid);
    if( parallel)
    {
        file::NC_Error_Handle err;
        err = detail::put_vara_T( ncid, varid,
                slab.startp(), slab.countp(), data.data().data());
    }
    else
    {
        thrust::host_vector<dg::get_value_type<host_vector>> receive;
        put_vara_detail( ncid, varid, slab, data.data(), receive);
    }
}
#endif // MPI_VERSION
} // namespace detail
///@endcond
//

/**
*
* @brief DEPRECATED Write an array to NetCDF file
*
* Convenience wrapper around \c nc_put_var
*
* The purpose of this function is mainly to simplify output in an MPI environment and to provide
* the same interface also in a shared memory system for uniform programming.
* This version is for a time-independent variable,
* i.e. writes a single variable in one go and is actually equivalent
* to \c nc_put_var. The dimensionality is given by the grid.
* @note This function throws a \c dg::file::NC_Error if an error occurs
* @tparam Topology One of the dG defined grids (e.g. \c dg::RealGrid2d)
* Determines if shared memory or MPI version is called
* @copydoc hide_tparam_host_vector
* @param ncid NetCDF file or group ID
* @param varid Variable ID
*
* [unnamed grid] Make serial and mpi interface equal
* @param data data to be written to the NetCDF file
*
* [unnamed bool] This parameter is there to make serial and parallel interface equal.
* @copydoc hide_master_comment
* @copydoc hide_parallel_write
*/
template<class host_vector, class Topology>
void put_var( int ncid, int varid, const Topology& /*grid*/,
    const host_vector& data, bool /*parallel*/ = false)
{
    file::NC_Error_Handle err;
    err = detail::put_var_T( ncid, varid, data.data());
}

/**
* @brief DEPRECATED Write an array to NetCDF file
*
* Convenience wrapper around \c nc_put_vara
*
* The purpose of this function is mainly to simplify output in an MPI environment and to provide
* the same interface also in a shared memory system for uniform programming.
* This version is for a time-dependent variable,
* i.e. writes a single time-slice into the file.
* The dimensionality is given by the grid.
* @note This function throws a \c dg::file::NC_Error if an error occurs
* @tparam Topology One of the dG defined grids (e.g. \c dg::RealGrid2d)
* Determines if shared memory or MPI version is called
* @copydoc hide_tparam_host_vector
* @param ncid NetCDF file or group ID
* @param varid Variable ID
* @param slice The number of the time-slice to write (first element of the \c startp array in \c nc_put_vara)
* @copydoc hide_comment_slice
* @param grid The grid from which to construct \c start and \c count variables to forward to \c nc_put_vara
* @param data data to be written to the NetCDF file
*
* [unnamed bool] This parameter is there to make serial and parallel interface equal.
* @copydoc hide_master_comment
* @copydoc hide_parallel_write
*/
template<class host_vector, class Topology>
void put_vara( int ncid, int varid, unsigned slice, const Topology& grid,
    const host_vector& data, bool /*parallel*/ = false)
{
    file::NC_Error_Handle err;
    NcHyperslab slab( slice, grid);
    err = detail::put_vara_T( ncid, varid, slab.startp(), slab.countp(),
            data.data());
}
// scalar data

/**
 * @brief DEPRECATED Write a scalar to the NetCDF file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to write
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID (Note that in NetCDF variables without dimensions are scalars)
 *
 * [unnamed RealGrid0d] a Tag to signify scalar ouput (and help the compiler choose this function over the array output function). Can be of type <tt> dg::RealMPIGrid<real_type> </tt>
 * @param data The (single) datum to write.
 *
 * [unnamed bool] This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes can call this function but only the master thread writes.
 * @copydoc hide_master_comment
 */
template<class T, class real_type>
void put_var( int ncid, int varid, const RealGrid0d<real_type>& /*grid*/,
    T data, bool /*parallel*/ = false)
{
    file::NC_Error_Handle err;
    err = detail::put_var_T( ncid, varid, &data);
}
/**
 * @brief DEPRECATED Write a scalar to the NetCDF file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to write
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID (Note that in NetCDF variables without dimensions are scalars)
 * @param slice The number of the time-slice to write (first element of the \c startp array in \c nc_put_vara)
 * @copydoc hide_comment_slice
 *
 * [unnamed RealGrid0d] a Tag to signify scalar ouput (and help the compiler choose this function over the array output function). Can be of type <tt> dg::RealMPIGrid<real_type> </tt>
 * @param data The (single) datum to write.
 *
 * [unnamed bool] This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes can call this function but only the master thread writes.
 * @copydoc hide_master_comment
 */
template<class T, class real_type>
void put_vara( int ncid, int varid, unsigned slice, const RealGrid0d<real_type>& /*grid*/,
    T data, bool /*parallel*/ = false)
{
    file::NC_Error_Handle err;
    size_t count = 1;
    size_t start = slice; // convert to size_t
    err = detail::put_vara_T( ncid, varid, &start, &count, &data);
}

///@}

///@cond
#ifdef MPI_VERSION

// array data
template<class host_vector, class MPITopology>
void put_var(int ncid, int varid, const MPITopology& grid,
    const dg::MPI_Vector<host_vector>& data, bool parallel = false)
{
    detail::put_vara_detail( ncid, varid, 0, grid, data, false, parallel);
}

template<class host_vector, class MPITopology>
void put_vara(int ncid, int varid, unsigned slice,
    const MPITopology& grid, const dg::MPI_Vector<host_vector>& data,
    bool parallel = false)
{
    detail::put_vara_detail( ncid, varid, slice, grid, data, true, parallel);
}
// scalar data

template<class T, class real_type>
void put_var( int ncid, int varid, const RealMPIGrid0d<real_type>&,
    T data, bool parallel = false)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0)
        put_var( ncid, varid, dg::RealGrid0d<real_type>(), data, parallel);
}
template<class T, class real_type>
void put_vara( int ncid, int varid, unsigned slice, const RealMPIGrid0d<real_type>& /*grid*/,
    T data, bool parallel = false)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0)
        put_vara( ncid, varid, slice, dg::RealGrid0d<real_type>(), data, parallel);
}

#endif //MPI_VERSION

/// DEPRECATED
/// alias for \c put_var
template<class host_vector, class Topology>
void put_var_double(int ncid, int varid, const Topology& grid,
    const host_vector& data, bool parallel = false)
{
    put_var( ncid, varid, grid, data, parallel);
}

/// DEPRECATED
/// alias for \c put_vara
template<class host_vector, class Topology>
void put_vara_double(int ncid, int varid, unsigned slice, const Topology& grid,
    const host_vector& data, bool parallel = false)
{
    put_vara( ncid, varid, slice, grid, data, parallel);
}
///@endcond

}//namespace file
}//namespace dg
