#pragma once

#include <exception>
#include <netcdf.h>
#include "dg/topology/grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "dg/topology/mpi_grid.h"
#endif //MPI_VERSION
#include "nc_error.h"
#include "nc_hyperslab.h"

/*!@file
 *
 * Define variable readers
 */

namespace dg
{
namespace file
{

///@cond
namespace detail
{

template<class T>
inline int get_var_T( int /*ncid*/, int /*varID*/, T* /*data*/)
{
    assert( false && "Type not supported!\n" );
    return NC_EBADTYPE;
}
template<>
inline int get_var_T<float>( int ncid, int varID, float* data){
    return nc_get_var_float( ncid, varID, data);
}
template<>
inline int get_var_T<double>( int ncid, int varID, double* data){
    return nc_get_var_double( ncid, varID, data);
}
template<>
inline int get_var_T<int>( int ncid, int varID, int* data){
    return nc_get_var_int( ncid, varID, data);
}
template<>
inline int get_var_T<unsigned>( int ncid, int varID, unsigned* data){
    return nc_get_var_uint( ncid, varID, data);
}
template<class T>
inline int get_vara_T( int /*ncid*/, int /*varID*/, const size_t* /*startp*/, const size_t* /*countp*/, T* /*data*/)
{
    assert( false && "Type not supported!\n" );
    return NC_EBADTYPE;
}
template<>
inline int get_vara_T<float>( int ncid, int varID, const size_t* startp, const size_t* countp, float* data){
    return nc_get_vara_float( ncid, varID, startp, countp, data);
}
template<>
inline int get_vara_T<double>( int ncid, int varID, const size_t* startp, const size_t* countp, double* data){
    return nc_get_vara_double( ncid, varID, startp, countp, data);
}
template<>
inline int get_vara_T<int>( int ncid, int varID, const size_t* startp, const size_t* countp, int* data){
    return nc_get_vara_int( ncid, varID, startp, countp, data);
}
template<>
inline int get_vara_T<unsigned>( int ncid, int varID, const size_t* startp, const size_t* countp, unsigned* data){
    return nc_get_vara_uint( ncid, varID, startp, countp, data);
}
#ifdef MPI_VERSION
template<class host_vector>
void get_vara_detail(int ncid, int varid,
        const MPINcHyperslab& slab,
        host_vector& data,
        thrust::host_vector<dg::get_value_type<host_vector>>& to_send,
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
    int err = NC_NOERR;
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
        to_send.resize( max_size);
        for( int r=0; r<size; r++)
        {
            if(r!=rank)
            {
                int e = detail::get_vara_T( ncid, varid, &r_start[r*ndim],
                        &r_count[r*ndim], to_send.data()); // read data
                MPI_Send( to_send.data(), (int)sizes[r], mpitype, r, r, comm);
                if( not e) err = e;
            }
            else // read own data
            {
                int e = detail::get_vara_T( ncid, varid, slab.startp(),
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
        MPI_Status status;
        MPI_Recv( thrust::raw_pointer_cast(data.data()), num, mpitype,
                  local_root_rank, rank, comm, &status);
    }
    MPI_Bcast( &err, 1, dg::getMPIDataType<int>(), local_root_rank, comm);
    if( err)
        throw NC_Error( err);
    return;
}

// all comms must be same size
template<class host_vector, class MPITopology>
void get_vara_detail(int ncid, int varid, unsigned slice,
        const MPITopology& grid, MPI_Vector<host_vector>& data,
        bool vara, bool parallel = false)
{
    MPINcHyperslab slab( grid);
    if( vara)
        slab = MPINcHyperslab( slice, grid);
    if( parallel)
    {
        file::NC_Error_Handle err;
        err = detail::get_vara_T( ncid, varid,
                slab.startp(), slab.countp(), data.data().data());
    }
    else
    {
        thrust::host_vector<dg::get_value_type<host_vector>> to_send;
        get_vara_detail( ncid, varid, slab, data.data(), to_send);
    }
}
#endif // MPI_VERSION
} // namespace detail
///@endcond

/**
 * @addtogroup legacy
 * @{
 */
/**
* @brief DEPRECATED Convenience wrapper around \c nc_get_var
*
* The purpose of this function is mainly to simplify input in an MPI
* environment and to provide the same interface also in a shared memory system
* for uniform programming.  This version is for a time-independent variable,
* i.e. reads a single variable in one go and is actually equivalent to \c
* nc_get_var. The dimensionality is given by the grid.
* @note This function throws a \c dg::file::NC_Error if an error occurs
* @tparam Topology One of the dG defined grids (e.g. \c dg::RealGrid2d)
* Determines if shared memory or MPI version is called
* @copydoc hide_tparam_host_vector
* @param ncid NetCDF file or group ID
* @param varid Variable ID
* [unnamed Topology] The grid from which to construct \c start and \c count variables to forward to \c nc_get_vara
* @param data contains the read data on return (must be of size \c grid.size() )
* [unnamed bool] This parameter is there to make serial and parallel interface equal.
* @copydoc hide_master_comment
* @copydoc hide_parallel_read
*/
template<class host_vector, class Topology>
void get_var( int ncid, int varid, const Topology& /*grid*/,
    host_vector& data, bool /*parallel*/ = true)
{
    file::NC_Error_Handle err;
    err = detail::get_var_T( ncid, varid, data.data());
}

/**
* @brief DEPRECATED Convenience wrapper around \c nc_get_vara()
*
* The purpose of this function is mainly to simplify input in an MPI environment and to provide
* the same interface also in a shared memory system for uniform programming.
* This version is for a time-dependent variable,
* i.e. reads a single time-slice from the file.
* The dimensionality is given by the grid.
* @note This function throws a \c dg::file::NC_Error if an error occurs
* @tparam Topology One of the dG defined grids (e.g. \c dg::RealGrid2d)
* Determines if shared memory or MPI version is called
* @copydoc hide_tparam_host_vector
* @param ncid NetCDF file or group ID
* @param varid Variable ID
* @param slice The number of the time-slice to read (first element of the \c startp array in \c nc_get_vara)
* @param grid The grid from which to construct \c start and \c count variables to forward to \c nc_get_vara
* @param data contains the read data on return (must be of size \c grid.size() )
*
* [unnamed bool] This parameter is there to make serial and parallel interface equal.
* @copydoc hide_master_comment
* @copydoc hide_parallel_read
*/
template<class host_vector, class Topology>
void get_vara( int ncid, int varid, unsigned slice, const Topology& grid,
    host_vector& data, bool /*parallel*/ = true)
{
    file::NC_Error_Handle err;
    NcHyperslab slab( slice, grid);
    err = detail::get_vara_T( ncid, varid, slab.startp(), slab.countp(),
            data.data());
}
// scalar data

/**
 * @brief DEPRECATED Read a scalar from the netcdf file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to read
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID
 *
 * [unnamed RealGrid0d] a Tag to signify scalar ouput (and help the compiler choose this function over the array input function). Can be of type <tt> dg::RealMPIGrid0d<real_type> </tt>
 * @param data The (single) datum read from file.
 *
 * [unnamed bool] This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes call this function and all processes read.
 * @copydoc hide_parallel_read
 */
template<class T, class real_type>
void get_var( int ncid, int varid, const RealGrid0d<real_type>& /*grid*/,
    T& data, bool /*parallel*/ = true)
{
    file::NC_Error_Handle err;
    err = detail::get_var_T( ncid, varid, &data);
}
/**
 * @brief DEPRECATED Read a scalar to the netcdf file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to read
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID
 * @param slice The number of the time-slice to read (first element of the \c startp array in \c nc_get_vara)
 *
 * [unnamed RealGrid0d] a Tag to signify scalar ouput (and help the compiler choose this function over the array output function). Can be of type <tt> dg::RealMPIGrid<real_type> </tt>
 * @param data The (single) datum to read.
 *
 * [unnamed bool] This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes call this function and all processes read.
 * @copydoc hide_master_comment
 * @copydoc hide_parallel_read
 */
template<class T, class real_type>
void get_vara( int ncid, int varid, unsigned slice, const RealGrid0d<real_type>& /*grid*/,
    T& data, bool /*parallel*/ = true)
{
    file::NC_Error_Handle err;
    size_t count = 1;
    size_t start = slice; // convert to size_t
    err = detail::get_vara_T( ncid, varid, &start, &count, &data);
}

///@}

///@cond
#ifdef MPI_VERSION

template<class host_vector, class MPITopology>
void get_var(int ncid, int varid, const MPITopology& grid,
    dg::MPI_Vector<host_vector>& data, bool parallel = true)
{
    detail::get_vara_detail( ncid, varid, 0, grid, data, false, parallel);
}

template<class host_vector, class MPITopology>
void get_vara(int ncid, int varid, unsigned slice,
    const MPITopology& grid, dg::MPI_Vector<host_vector>& data,
    bool parallel = true)
{
    detail::get_vara_detail( ncid, varid, slice, grid, data, true, parallel);
}

// scalar data

template<class T, class real_type>
void get_var( int ncid, int varid, const RealMPIGrid0d<real_type>& grid,
    T& data, bool parallel = true)
{
    get_var( ncid, varid, dg::RealGrid0d<real_type>(), data, parallel);
}

template<class T, class real_type>
void get_vara( int ncid, int varid, unsigned slice, const RealMPIGrid0d<real_type>& /*grid*/,
    T& data, bool parallel = true)
{
    get_vara( ncid, varid, slice, dg::RealGrid0d<real_type>(), data, parallel);
}
#endif //MPI_VERSION
///@endcond

}//namespace file
}//namespace dg
