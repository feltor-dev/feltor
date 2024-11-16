#pragma once

#include <exception>
#include <netcdf.h>
#include "dg/topology/grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "dg/topology/mpi_grid.h"
#endif //MPI_VERSION
#include "nc_error.h"

/*!@file
 *
 * Define variable readers
 */

// Variable ids are persistent once created
// Attribute ids can change if one deletes attributes
namespace dg
{
namespace file
{
/**
 * @addtogroup Input
 * @{
 */

///@cond
namespace detail
{

template<class T>
inline int get_var_T( int ncid, int varID, T* data)
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
inline int get_vara_T( int ncid, int varID, const size_t* startp, const size_t* countp, T* data)
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

template<class host_vector, class MPITopology>
void get_vara_detail(int ncid, int varid, unsigned slice, const MPITopology& grid, MPI_Vector<host_vector>& data, bool vara, bool parallel = true)
{
    MPI_Comm comm = grid.communicator();
    int local_root_rank = dg::mpi_comm_global2local_rank(comm);
    if (local_root_rank == MPI_UNDEFINED)
        return;
    unsigned grid_ndims = grid.ndim();
    auto cc = grid.local().get_shape();
    std::vector<size_t> count( cc.begin(), cc.end());
    std::reverse( count.begin(), count.end());
    if( vara)
        count.insert( count.begin(), 1);
    int rank, size;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    std::vector<size_t> start(vara ? grid_ndims+1 : grid_ndims,0);
    if( vara)
        start[0] = slice;
    file::NC_Error_Handle err;
    if( parallel)
    {
        int coords[grid_ndims];
        MPI_Cart_coords( comm, rank, grid_ndims, coords);
        for( unsigned i=0; i<grid_ndims; i++)
            start[vara ? i+1 : i] = count[ vara ? i+1 : i]*coords[grid_ndims-1-i];
        err = detail::get_vara_T( ncid, varid, &start[0], &count[0],
            data.data().data());
    }
    else
    {
        MPI_Status status;
        size_t local_size = data.data().size();
        std::vector<int> coords( grid_ndims*size);
        for( int rrank=0; rrank<size; rrank++)
            MPI_Cart_coords( comm, rrank, grid_ndims, &coords[grid_ndims*rrank]);
        if( rank == local_root_rank )
        {
            host_vector to_send( data.data());
            for( int rrank=0; rrank<size; rrank++)
            {
                for ( unsigned i=0; i<grid_ndims; i++)
                    start[vara ? i+1 : i] = count[ vara ? i+1 : i]*coords[grid_ndims*rrank + grid_ndims-1-i];
                if(rrank!=rank)
                {
                    err = detail::get_vara_T( ncid, varid, &start[0], &count[0],
                        to_send.data()); // read data to send
                    MPI_Send( to_send.data(), local_size, dg::getMPIDataType<get_value_type<host_vector>>(),
                          rrank, rrank, comm);
                }
                else // read own data
                    err = detail::get_vara_T( ncid, varid, &start[0], &count[0], data.data().data());
            }
        }
        else
            MPI_Recv( data.data().data(), local_size, dg::getMPIDataType<get_value_type<host_vector>>(),
                      local_root_rank, rank, comm, &status);
        MPI_Barrier( comm);
    }
}
#endif // MPI_VERSION
} // namespace detail
///@endcond

/**
* @brief Convenience wrapper around \c nc_get_var
*
* The purpose of this function is mainly to simplify input in an MPI environment and to provide
* the same interface also in a shared memory system for uniform programming.
* This version is for a time-independent variable,
* i.e. reads a single variable in one go and is actually equivalent
* to \c nc_get_var. The dimensionality is given by the grid.
* @note This function throws a \c dg::file::NC_Error if an error occurs
* @tparam Topology One of the dG defined grids (e.g. \c dg::RealGrid2d)
* Determines if shared memory or MPI version is called
* @copydoc hide_tparam_host_vector
* @param ncid NetCDF file or group ID
* @param varid Variable ID
* @param grid The grid from which to construct \c start and \c count variables to forward to \c nc_get_vara
* @param data contains the read data on return (must be of size \c grid.size() )
* @copydoc hide_parallel_param
* @copydoc hide_master_comment
* @copydoc hide_parallel_read
*/
template<class host_vector, class Topology>
void get_var( int ncid, int varid, const Topology& grid,
    host_vector& data, bool parallel = true)
{
    file::NC_Error_Handle err;
    err = detail::get_var_T( ncid, varid, data.data());
}

/**
* @brief Convenience wrapper around \c nc_get_vara()
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
* @copydoc hide_parallel_param
* @copydoc hide_master_comment
* @copydoc hide_parallel_read
*/
template<class host_vector, class Topology>
void get_vara( int ncid, int varid, unsigned slice, const Topology& grid,
    host_vector& data, bool parallel = true)
{
    file::NC_Error_Handle err;
    auto cc = grid.get_shape();
    std::vector<size_t> count( cc.begin(), cc.end());
    std::reverse( count.begin(), count.end());
    count.insert( count.begin(), 1);
    std::vector<size_t> start( count.size(), 0);
    start[0] = slice;
    err = detail::get_vara_T( ncid, varid, &start[0], &count[0], data.data());
}
// scalar data

/**
 * @brief Read a scalar from the netcdf file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to read
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID
 * @param grid a Tag to signify scalar ouput (and help the compiler choose this function over the array input function). Can be of type <tt> dg::RealMPIGrid0d<real_type> </tt>
 * @param data The (single) datum read from file.
 * @param parallel This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes call this function and all processes read.
 * @copydoc hide_parallel_read
 */
template<class T, class real_type>
void get_var( int ncid, int varid, const RealGrid0d<real_type>& grid,
    T& data, bool parallel = true)
{
    file::NC_Error_Handle err;
    err = detail::get_var_T( ncid, varid, &data);
}
/**
 * @brief Read a scalar to the netcdf file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to read
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID
 * @param slice The number of the time-slice to read (first element of the \c startp array in \c nc_put_vara)
 * @param grid a Tag to signify scalar ouput (and help the compiler choose this function over the array input function). Can be of type <tt> dg::RealMPIGrid0d<real_type> </tt>
 * @param data The (single) datum to read.
 * @param parallel This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes call this function and all processes read.
 * @copydoc hide_master_comment
 * @copydoc hide_parallel_read
 */
template<class T, class real_type>
void get_vara( int ncid, int varid, unsigned slice, const RealGrid0d<real_type>& grid,
    T& data, bool parallel = true)
{
    file::NC_Error_Handle err;
    size_t count = 1;
    size_t start = slice; // convert to size_t
    err = detail::get_vara_T( ncid, varid, &start, &count, &data);
}


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
void get_vara( int ncid, int varid, unsigned slice, const RealMPIGrid0d<real_type>& grid,
    T& data, bool parallel = true)
{
    get_vara( ncid, varid, slice, dg::RealGrid0d<real_type>(), data, parallel);
}
#endif //MPI_VERSION
///@endcond

///@}
}//namespace file
}//namespace dg
