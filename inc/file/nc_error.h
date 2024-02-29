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
 * @ingroup netcdf
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
