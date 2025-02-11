#pragma once

#include <exception>
#include <netcdf.h>

/*!@file
 *
 * Contains Error handling class
 */
namespace dg
{
namespace file
{

/**
 * @brief Class thrown by the NC_Error_Handle
 * @ingroup netcdf
 */
struct NC_Error : public std::exception
{
    /**
     * @brief Construct from error code
     *
     * @param error netcdf error code
     */
    NC_Error( int error): m_error( error) {}

    int error() const { return m_error;}
    int& error() { return m_error;}
    /**
     * @brief What string
     *
     * @return string netcdf error message generated from error code
     */
    char const* what() const noexcept{
        if ( m_error == 1000)
            return "NC ERROR Cannot operate on closed file!\n";
        else if( m_error == 1001)
            return "NC ERROR Slab dimension does not match variable dimension!\n";
        else if( m_error == 1002)
            return "NC ERROR Cannot open file. File already open!\n";
        return nc_strerror(m_error);}
  private:
    int m_error;
};

/**
 * @brief DEPRECATED Empty utitlity class that handles return values of netcdf
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
 * @ingroup legacy
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

}//namespace file
}//namespace dg
