#pragma once

#include <exception>
#include <netcdf.h>
#include "dg/topology/grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "dg/topology/mpi_grid.h"
#endif //MPI_VERSION

/*!@file
 *
 * Contains Error handling class
 */

namespace dg
{
namespace file
{
/**
 * @defgroup netcdf NetCDF utilities
 * \#include "dg/file/nc_utilities.h" (link -lnetcdf -lhdf5[_serial] -lhdf5[_serial]_hl)
 *
 * @addtogroup netcdf
 * @{
 */

/**
 * @brief Class thrown by the NC_Error_Handle
 */
struct NC_Error : public std::exception
{

    /**
     * @brief Construct from error code
     *
     * @param error netcdf error code
     */
    NC_Error( int error): error_( error) {}
    /**
     * @brief What string
     *
     * @return string netcdf error message generated from error code
     */
    char const* what() const throw(){
        return nc_strerror(error_);}
  private:
    int error_;
};

/**
 * @brief Empty utitlity class that handles return values of netcdf
 * functions and throws NC_Error(status) if( status != NC_NOERR)
 *
 * For example
 * @code
 * file::NC_Error_Handle err;
 * int ncid = -1;
 * try{
 *      err = nc_open( "file.nc", NC_WRITE, &ncid);
 * //throws if for example "file.nc" does not exist
 * } catch ( std::exception& e)
 * {
 *      //log the error and exit
 *      std::cerr << "An error occured opening file.nc !\n"<<e.what()<<std::endl;
 *      exit( EXIT_FAILURE);
 * }
 * @endcode
 *
 * This allows for a C++ style error handling of netcdf errors in that the program either terminates if the NC_Error is not caught or does something graceful in a try catch statement.
 */
struct NC_Error_Handle
{
    /**
     * @brief Construct from error code
     *
     * throws an NC_Error if err is not a success
     * @param err netcdf error code
     *
     * @return Newly instantiated object
     */
    NC_Error_Handle operator=( int err)
    {
        NC_Error_Handle h;
        return h(err);
    }
    /**
     * @brief Construct from error code
     *
     * throws an NC_Error if err is not a success
     * @param err netcdf error code
     *
     * @return Newly instantiated object
     */
    NC_Error_Handle operator()( int err)
    {
        if( err)
            throw NC_Error( err);
        return *this;
    }
};

/**
* @brief Convenience wrapper around \c nc_put_vara_double()
*
* The purpose of this function is mainly to simplify output in an MPI environment and to provide
* the same interface also in a shared memory system for uniform programming.
* This version is for a time-independent variable,
* i.e. writes a single variable in one go and is actually equivalent
* to \c nc_put_var_double. The dimensionality is given by the grid.
* @note This function throws a \c dg::file::NC_Error if an error occurs
* @tparam host_vector Type with \c data() member that returns pointer to first element in CPU (host) adress space, meaning it cannot be a GPU vector
* @param ncid Forwarded to \c nc_put_vara_double
* @param varid  Forwarded to \c nc_put_vara_double
* @param grid The grid from which to construct \c start and \c count variables to forward to \c nc_put_vara_double
* @param data data is forwarded to \c nc_put_vara_double
* @param parallel This parameter is ignored in the serial version.
* In the MPI version this parameter indicates whether each process
* writes to the file independently in parallel (\c true)
* or each process funnels its data through the master rank (\c false),
* which involves communication but may be faster than the former method.
* @attention In the MPI version (i) all processes must call this function and (ii) if \c parallel==true a **parallel netcdf and hdf5** must be
* linked, the file opened with the \c NC_MPIIO flag from the \c netcdf_par.h header and the variable be marked with \c NC_COLLECTIVE access while if \c parallel==false we need **serial netcdf and hdf5** and only the master thread needs to open and access the file.
* Note that serious performance penalties have been observed on some platforms for parallel netcdf.
*/
template<class host_vector>
void put_var_double(int ncid, int varid, const dg::aTopology2d& grid,
    const host_vector& data, bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[2] = {0,0}, count[2];
    count[0] = grid.n()*grid.Ny();
    count[1] = grid.n()*grid.Nx();
    err = nc_put_vara_double( ncid, varid, start, count, data.data());
}

/**
* @brief Convenience wrapper around \c nc_put_vara_double()
*
* The purpose of this function is mainly to simplify output in an MPI environment and to provide
* the same interface also in a shared memory system for uniform programming.
* This version is for a time-dependent variable,
* i.e. writes a single time-slice into the file.
* The dimensionality is given by the grid.
* @note This function throws a \c dg::file::NC_Error if an error occurs
* @tparam host_vector Type with \c data() member that returns pointer to first element in CPU (host) adress space, meaning it cannot be a GPU vector
* @param ncid Forwarded to \c nc_put_vara_double
* @param varid  Forwarded to \c nc_put_vara_double
* @param slice The number of the time-slice to write (first element of the \c startp array in \c nc_put_vara_double)
* @param grid The grid from which to construct \c start and \c count variables to forward to \c nc_put_vara_double
* @param data data is forwarded to \c nc_put_vara_double, 
* @param parallel This parameter is ignored in the serial version.
* In the MPI version this parameter indicates whether each process
* writes to the file independently in parallel (\c true)
* or each process funnels its data through the master rank (\c false),
* which involves communication but may be faster than the former method.
* @attention In the MPI version (i) all processes must call this function and (ii) if \c parallel==true a **parallel netcdf and hdf5** must be
* linked, the file opened with the \c NC_MPIIO flag from the \c netcdf_par.h
* header and the variable be marked with \c NC_COLLECTIVE access while if \c
* parallel==false we need **serial netcdf and hdf5** and only the master thread
* needs to open and access the file.  Note that serious performance penalties
* have been observed on some platforms for parallel netcdf.
*/
template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
    const dg::aTopology2d& grid, const host_vector& data, bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[3] = {slice,0,0}, count[3];
    count[0] = 1;
    count[1] = grid.n()*grid.Ny();
    count[2] = grid.n()*grid.Nx();
    err = nc_put_vara_double( ncid, varid, start, count, data.data());
}

///@copydoc put_var_double()
template<class host_vector>
void put_var_double(int ncid, int varid, const dg::aTopology3d& grid,
    const host_vector& data, bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[3] = {0,0,0}, count[3];
    count[0] = grid.Nz();
    count[1] = grid.n()*grid.Ny();
    count[2] = grid.n()*grid.Nx();
    err = nc_put_vara_double( ncid, varid, start, count, data.data());
}

///@copydoc put_vara_double()
template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
    const dg::aTopology3d& grid, const host_vector& data, bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[4] = {slice, 0,0,0}, count[4];
    count[0] = 1;
    count[1] = grid.Nz();
    count[2] = grid.n()*grid.Ny();
    count[3] = grid.n()*grid.Nx();
    err = nc_put_vara_double( ncid, varid, start, count, data.data());
}

#ifdef MPI_VERSION
///@copydoc put_var_double()
template<class host_vector>
void put_var_double(int ncid, int varid, const dg::aMPITopology2d& grid,
    const dg::MPI_Vector<host_vector>& data, bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[3] = {0,0}, count[2];
    count[0] = grid.n()*grid.local().Ny();
    count[1] = grid.n()*grid.local().Nx();
    int rank, size;
    MPI_Comm comm = grid.communicator();
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    if( !parallel)
    {
        MPI_Status status;
        size_t local_size = grid.local().size();
        std::vector<int> coords( size*2);
        for( int rrank=0; rrank<size; rrank++)
            MPI_Cart_coords( comm, rrank, 2, &coords[2*rrank]);
        if(rank==0)
        {
            host_vector receive( data.data());
            for( int rrank=0; rrank<size; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
                          rrank, rrank, comm, &status);
                start[0] = coords[2*rrank+1]*count[0],
                start[1] = coords[2*rrank+0]*count[1],
                err = nc_put_vara_double( ncid, varid, start, count,
                    receive.data());
            }
        }
        else
            MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
                      0, rank, comm);
        MPI_Barrier( comm);
    }
    else
    {
        int coords[2];
        MPI_Cart_coords( comm, rank, 2, coords);
        start[0] = coords[1]*count[0],
        start[1] = coords[0]*count[1],
        err = nc_put_vara_double( ncid, varid, start, count,
            data.data().data());
    }
}

///@copydoc put_vara_double()
template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
    const dg::aMPITopology2d& grid, const dg::MPI_Vector<host_vector>& data,
    bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[3] = {slice, 0,0}, count[3];
    count[0] = 1;
    count[1] = grid.n()*grid.local().Ny();
    count[2] = grid.n()*grid.local().Nx();
    int rank, size;
    MPI_Comm comm = grid.communicator();
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    if( parallel)
    {
        int coords[2];
        MPI_Cart_coords( comm, rank, 2, coords);
        start[1] = coords[1]*count[1],
        start[2] = coords[0]*count[2],
        err = nc_put_vara_double( ncid, varid, start, count,
            data.data().data());
    }
    else
    {
        MPI_Status status;
        size_t local_size = grid.local().size();
        std::vector<int> coords( size*2);
        for( int rrank=0; rrank<size; rrank++)
            MPI_Cart_coords( comm, rrank, 2, &coords[2*rrank]);
        if(rank==0)
        {
            host_vector receive( data.data());
            for( int rrank=0; rrank<size; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
                          rrank, rrank, comm, &status);
                start[1] = coords[2*rrank+1]*count[1],
                start[2] = coords[2*rrank+0]*count[2],
                err = nc_put_vara_double( ncid, varid, start, count,
                    receive.data());
            }
        }
        else
            MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
                      0, rank, comm);
        MPI_Barrier( comm);
    }
}

///@copydoc put_var_double()
template<class host_vector>
void put_var_double(int ncid, int varid,
    const dg::aMPITopology3d& grid, const dg::MPI_Vector<host_vector>& data,
    bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[3] = {0,0,0}, count[3];
    count[0] = grid.local().Nz();
    count[1] = grid.n()*grid.local().Ny();
    count[2] = grid.n()*grid.local().Nx();
    int rank, size;
    MPI_Comm comm = grid.communicator();
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    if( !parallel)
    {
        MPI_Status status;
        size_t local_size = grid.local().size();
        std::vector<int> coords( size*3);
        for( int rrank=0; rrank<size; rrank++)
            MPI_Cart_coords( comm, rrank, 3, &coords[3*rrank]);
        if(rank==0)
        {
            host_vector receive( data.data());
            for( int rrank=0; rrank<size; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
                          rrank, rrank, comm, &status);
                start[0] = coords[3*rrank+2]*count[0],
                start[1] = coords[3*rrank+1]*count[1],
                start[2] = coords[3*rrank+0]*count[2];
                err = nc_put_vara_double( ncid, varid, start, count,
                    receive.data());
            }
        }
        else
            MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
                      0, rank, comm);
        MPI_Barrier( comm);
    }
    else
    {
        int coords[3];
        MPI_Cart_coords( comm, rank, 3, coords);
        start[0] = coords[2]*count[0],
        start[1] = coords[1]*count[1],
        start[2] = coords[0]*count[2];
        err = nc_put_vara_double( ncid, varid, start, count,
            data.data().data());
    }
}

///@copydoc put_vara_double()
template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
    const dg::aMPITopology3d& grid, const dg::MPI_Vector<host_vector>& data,
    bool parallel = false)
{
    file::NC_Error_Handle err;
    size_t start[4] = {slice, 0,0,0}, count[4];
    count[0] = 1;
    count[1] = grid.local().Nz();
    count[2] = grid.n()*grid.local().Ny();
    count[3] = grid.n()*grid.local().Nx();
    int rank, size;
    MPI_Comm comm = grid.communicator();
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    if( parallel)
    {
        int coords[3];
        MPI_Cart_coords( comm, rank, 3, coords);
        start[1] = coords[2]*count[1],
        start[2] = coords[1]*count[2],
        start[3] = coords[0]*count[3];
        err = nc_put_vara_double( ncid, varid, start, count,
            data.data().data());
    }
    else
    {
        MPI_Status status;
        size_t local_size = grid.local().size();
        std::vector<int> coords( size*3);
        for( int rrank=0; rrank<size; rrank++)
            MPI_Cart_coords( comm, rrank, 3, &coords[3*rrank]);
        if(rank==0)
        {
            host_vector receive( data.data());
            for( int rrank=0; rrank<size; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
                          rrank, rrank, comm, &status);
                start[1] = coords[3*rrank+2]*count[1],
                start[2] = coords[3*rrank+1]*count[2],
                start[3] = coords[3*rrank+0]*count[3];
                err = nc_put_vara_double( ncid, varid, start, count,
                    receive.data());
            }
        }
        else
            MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
                      0, rank, comm);
        MPI_Barrier( comm);
    }
}
#endif //MPI_VERSION

///@}
}//namespace file
}//namespace dg
