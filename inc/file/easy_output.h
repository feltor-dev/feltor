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
 * Define variable writers
 */

// Variable ids are persistent once created
// Attribute ids can change if one deletes attributes
namespace dg
{
namespace file
{
/**
 * @addtogroup Output
 * @{
 * @class hide_master_comment
 * @note In serial netcdf only a single "master" process can operate on the
 * netcdf file (creation, open, defining dimension ids, variables ids, reading,
 * writing etc).  The "master" thread is assumed to be the process with \c rank==0
 * in \c MPI_COMM_WORLD.  The \c MPI_COMM_WORLD rank of a process is usually the
 * same in a Cartesian communicator of the same size but is not guaranteed. So
 * always check \c MPI_COMM_WORLD ranks for file operations.
 *
 * @class hide_parallel_param
 * @param parallel This parameter is ignored in the serial version.
 * In the MPI version this parameter indicates whether each process
 * reads/writes to the file independently in parallel (\c true)
 * or each process funnels its data through the master rank (\c false),
 * which involves communication but may be faster than the former method.
 * @attention In the MPI version (i) all processes must call this function,
 * (ii) processes that do not belong to the same communicator as the master process
 * return immediately and
 * (iii) if \c parallel==true a **parallel netcdf and hdf5** must be linked, the
 * file opened with the \c NC_MPIIO flag from the \c netcdf_par.h header and the
 * variable be marked with \c NC_COLLECTIVE access while if \c parallel==false
 * we need **serial netcdf and hdf5** and only the master thread needs to open
 * and access the file.  Note that serious performance penalties have been
 * observed on some platforms for parallel netcdf.
 * @note In an MPI program it may happen that the data to read/write is partitioned among
 * a process group smaller than \c MPI_COMM_WORLD, e.g. when reading/writing a 2d slice of a 3d vector.
 * In this example case \c grid.communicator() is only 2d not 3d. Remember that **only
 * the group containing the master process reads/writes its data to the file**, while all other processes
 * immediately return.
 * There are two ways to relyable read/write the data in such a case:
 * (i) Manually assemble the data on the master process OR
 * (ii) Manually assemble the data on the MPI group that contains the master process
 * @sa \c dg::mpi_comm_global2local_rank
 */

///@cond
namespace detail
{

template<class T>
inline int put_var_T( int ncid, int varID, const T* data)
{
    assert( false && "Type not supported!\n" );
    return NC_EBADTYPE;
}
template<>
inline int put_var_T<float>( int ncid, int varID, const float* data){
    return nc_put_var_float( ncid, varID, data);
}
template<>
inline int put_var_T<double>( int ncid, int varID, const double* data){
    return nc_put_var_double( ncid, varID, data);
}
template<>
inline int put_var_T<int>( int ncid, int varID, const int* data){
    return nc_put_var_int( ncid, varID, data);
}
template<>
inline int put_var_T<unsigned>( int ncid, int varID, const unsigned* data){
    return nc_put_var_uint( ncid, varID, data);
}
template<class T>
inline int put_vara_T( int ncid, int varID, const size_t* startp, const size_t* countp, const T* data)
{
    assert( false && "Type not supported!\n" );
    return NC_EBADTYPE;
}
template<>
inline int put_vara_T<float>( int ncid, int varID, const size_t* startp, const size_t* countp, const float* data){
    return nc_put_vara_float( ncid, varID, startp, countp, data);
}
template<>
inline int put_vara_T<double>( int ncid, int varID, const size_t* startp, const size_t* countp, const double* data){
    return nc_put_vara_double( ncid, varID, startp, countp, data);
}
template<>
inline int put_vara_T<int>( int ncid, int varID, const size_t* startp, const size_t* countp, const int* data){
    return nc_put_vara_int( ncid, varID, startp, countp, data);
}
template<>
inline int put_vara_T<unsigned>( int ncid, int varID, const size_t* startp, const size_t* countp, const unsigned* data){
    return nc_put_vara_uint( ncid, varID, startp, countp, data);
}
#ifdef MPI_VERSION
// we need to identify the global root rank within the groups and mark the entire group
// all comms must be same size

template<class host_vector, class MPITopology>
void put_vara_detail(int ncid, int varid, unsigned slice, const MPITopology& grid, const MPI_Vector<host_vector>& data, bool vara, bool parallel = false)
{
    MPI_Comm comm = grid.communicator();
    int local_root_rank = dg::mpi_comm_global2local_rank(comm);
    if (local_root_rank == MPI_UNDEFINED)
        return;
    unsigned grid_ndims = grid.ndim();
    auto cc = shape( grid.local());
    std::reverse( cc.begin(), cc.end());
    if( vara)
        cc.insert( cc.begin(), 1);
    std::vector<size_t> count( cc.begin(), cc.end());
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
        err = detail::put_vara_T( ncid, varid, &start[0], &count[0],
            data.data().data());
    }
    else
    {
        MPI_Status status;
        size_t local_size = data.data().size();
        std::vector<int> coords( grid_ndims*size, 0);
        for( int rrank=0; rrank<size; rrank++)
            MPI_Cart_coords( comm, rrank, grid_ndims, &coords[grid_ndims*rrank]);
        if( rank == local_root_rank )
        {
            host_vector receive( data.data());
            for( int rrank=0; rrank<size; rrank++)
            {
                for ( unsigned i=0; i<grid_ndims; i++)
                    start[vara ? i+1 : i] = count[ vara ? i+1 : i]*coords[grid_ndims*rrank + grid_ndims-1-i];
                if(rrank!=rank)
                {
                    MPI_Recv( receive.data(), local_size, dg::getMPIDataType<get_value_type<host_vector>>(),
                          rrank, rrank, comm, &status);
                    err = detail::put_vara_T( ncid, varid, &start[0], &count[0],
                        receive.data()); // write received data
                }
                else // write own data
                {
                    err = detail::put_vara_T( ncid, varid, &start[0], &count[0], data.data().data());
                }
            }
        }
        else
        {
            MPI_Send( data.data().data(), local_size, dg::getMPIDataType<get_value_type<host_vector>>(),
                      local_root_rank, rank, comm);
        }
    }
    MPI_Barrier( comm);
    return;
}
#endif // MPI_VERSION
} // namespace detail
///@endcond

/**
*
* @brief Write an array to netcdf file
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
* @tparam host_vector For shared Topology: Type with \c data() member that
* returns pointer to first element in CPU (host) adress space, meaning it
* cannot be a GPU vector. For MPI Topology: must be \c MPI_Vector
* @param ncid NetCDF file or group ID
* @param varid Variable ID
* @param grid The grid from which to construct \c start and \c count variables to forward to \c nc_put_vara
* @param data data to be written to the netcdf file
* @copydoc hide_parallel_param
* @copydoc hide_master_comment
*/
template<class host_vector, class Topology>
void put_var( int ncid, int varid, const Topology& grid,
    const host_vector& data, bool parallel = false)
{
    file::NC_Error_Handle err;
    err = detail::put_var_T( ncid, varid, data.data());
}

/**
* @brief Write an array to netcdf file
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
* @tparam host_vector For shared Topology: Type with \c data() member that
* returns pointer to first element in CPU (host) adress space, meaning it
* cannot be a GPU vector. For MPI Topology: must be \c MPI_Vector
* @param ncid NetCDF file or group ID
* @param varid Variable ID
* @param slice The number of the time-slice to write (first element of the \c startp array in \c nc_put_vara)
* @param grid The grid from which to construct \c start and \c count variables to forward to \c nc_put_vara
* @param data data to be written to the netcdf file
* @copydoc hide_parallel_param
* @copydoc hide_master_comment
*/
template<class host_vector, class Topology>
void put_vara( int ncid, int varid, unsigned slice, const Topology& grid,
    const host_vector& data, bool parallel = false)
{
    file::NC_Error_Handle err;
    auto shape = dg::shape( grid);
    std::vector<size_t> count( shape.begin(), shape.end());
    std::reverse( count.begin(), count.end());
    count.insert( count.begin(), 1);
    std::vector<size_t> start( count.size(), 0);
    start[0] = slice;
    err = detail::put_vara_T( ncid, varid, &start[0], &count[0], data.data());
}
// scalar data

/**
 * @brief Write a scalar to the netcdf file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to write
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID (Note that in netcdf variables without dimensions are scalars)
 * @param grid a Tag to signify scalar ouput (and help the compiler choose this function over the array output function). Can be of type <tt> dg::RealMPIGrid<real_type> </tt>
 * @param data The (single) datum to write.
 * @param parallel This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes can call this function but only the master thread writes.
 * @copydoc hide_master_comment
 */
template<class T, class real_type>
void put_var( int ncid, int varid, const RealGrid0d<real_type>& grid,
    T data, bool parallel = false)
{
    file::NC_Error_Handle err;
    err = detail::put_var_T( ncid, varid, &data);
}
/**
 * @brief Write a scalar to the netcdf file
 *
 * @note This function throws a \c dg::file::NC_Error if an error occurs
 * @tparam T Determines data type to write
 * @tparam real_type ignored
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID (Note that in netcdf variables without dimensions are scalars)
 * @param slice The number of the time-slice to write (first element of the \c startp array in \c nc_put_vara)
 * @param grid a Tag to signify scalar ouput (and help the compiler choose this function over the array output function). Can be of type <tt> dg::RealMPIGrid<real_type> </tt>
 * @param data The (single) datum to write.
 * @param parallel This parameter is ignored in both serial and MPI versions.
 * In an MPI program all processes can call this function but only the master thread writes.
 * @copydoc hide_master_comment
 */
template<class T, class real_type>
void put_vara( int ncid, int varid, unsigned slice, const RealGrid0d<real_type>& grid,
    T data, bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t count = 1;
    size_t start = slice; // convert to size_t
    err = detail::put_vara_T( ncid, varid, &start, &count, &data);
}


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
    detail::put_vara_detail( ncid, varid, 0, grid, data, true, parallel);
}
// scalar data

template<class T, class real_type>
void put_var( int ncid, int varid, const RealMPIGrid0d<real_type>& grid,
    T data, bool parallel = false)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0)
        put_var( ncid, varid, dg::RealGrid0d<real_type>(), data, parallel);
}
template<class T, class real_type>
void put_vara( int ncid, int varid, unsigned slice, const RealMPIGrid0d<real_type>& grid,
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

///@}
}//namespace file
}//namespace dg
