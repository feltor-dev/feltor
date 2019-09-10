#pragma once
#ifndef _FILE_INCLUDED_BY_DG_
#pragma message( "The inclusion of file/nc_utilities.h is deprecated. Please use dg/file/nc_utilities.h")
#endif //_INCLUDED_BY_DG_

#include <netcdf.h>
#include "thrust/host_vector.h"

#include "dg/topology/grid.h"
#include "dg/topology/evaluation.h"

#include "easy_output.h"

/*!@file
 *
 * Contains Error handling class and the define_dimensions functions
 */


/**
* @brief Namespace for netcdf output related classes and functions following the
 <a href="http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html">CF-conventions</a>
*/
namespace file
{

/**
 * @brief Define an unlimited time dimension and variable following
  <a href="http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html">CF-conventions</a>
 *
 * The conventions dictate that the units attribute must be defined for a time variable: we give it the value "time since start"
 * @param ncid file ID
 * @param name Name of time variable (variable names are not standardized)
 * @param dimID time-dimension ID
 * @param tvarID time-variable ID
 *
 * @return netcdf error code if any
 */
static inline int define_time( int ncid, const char* name, int* dimID, int* tvarID)
{
    int retval;
    if( (retval = nc_def_dim( ncid, name, NC_UNLIMITED, dimID)) ){ return retval;}
    if( (retval = nc_def_var( ncid, name, NC_DOUBLE, 1, dimID, tvarID))){return retval;}
    std::string t = "time since start"; //needed for paraview to recognize timeaxis
    if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
    return retval;
}

/**
 * @brief Define a limited time dimension and variable following
  <a href="http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html">CF-conventions</a>
 *
 * The conventions dictate that the units attribute must be defined for a time variable: we give it the value "time since start"
 * @param ncid file ID
 * @param name Name of the time variable (usually "time")
 * @param size The number of timesteps
 * @param dimID time-dimension ID
 * @param tvarID time-variable ID
 *
 * @return netcdf error code if any
 */
static inline int define_limited_time( int ncid, const char* name, int size, int* dimID, int* tvarID)
{
    int retval;
    if( (retval = nc_def_dim( ncid, name, size, dimID)) ){ return retval;}
    if( (retval = nc_def_var( ncid, name, NC_DOUBLE, 1, dimID, tvarID))){return retval;}
    std::string t = "time since start"; //needed for paraview to recognize timeaxis
    if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
    return retval;
}

/**
 * @brief Define a 1d dimension and create a coordinate variable together with its data points in a netcdf file
 *
 * By netcdf conventions a variable with the same name as a dimension is called a coordinate variable.
 * @param ncid file ID
 * @param name Name of dimension (input)
 * @param dimID dimension ID (output)
 * @param points pointer to data (input)
 * @param size size of data points (input)
 *
 * @return netcdf error code if any
 */
static inline int define_dimension( int ncid, const char* name, int* dimID, const double * points, int size)
{
    int retval;
    if( (retval = nc_def_dim( ncid, name, size, dimID)) ) { return retval;}
    int varID;
    if( (retval = nc_def_var( ncid, name, NC_DOUBLE, 1, dimID, &varID))){return retval;}
    if( (retval = nc_enddef(ncid)) ) {return retval;} //not necessary for NetCDF4 files
    if( (retval = nc_put_var_double( ncid, varID, points)) ){ return retval;}
    if( (retval = nc_redef(ncid))) {return retval;} //not necessary for NetCDF4 files
    return retval;
}
/**
 * @brief Define a 1d dimension and create a coordinate variable together with its data points in a netcdf file
 *
 * By netcdf conventions a variable with the same name as a dimension is called a coordinate variable.
 * @param ncid file ID
 * @param name Name of dimension (input)
 * @param dimID dimension ID (output)
 * @param g The 1d DG grid from which data points are generated (input)
 *
 * @return netcdf error code if any
 */
static inline int define_dimension( int ncid, const char* name, int* dimID, const dg::Grid1d& g)
{
    thrust::host_vector<double> points = dg::create::abscissas( g);
    return define_dimension( ncid, name, dimID, points.data(), points.size());
}

///@cond
namespace detail{
static inline int define_x_dimension( int ncid, int* dimID, const dg::Grid1d& g)
{
    int retval;
    std::string long_name = "x-coordinate in Computational coordinate system";
    if( (retval = define_dimension( ncid, "x", dimID, g))){ return retval;}
    retval = nc_put_att_text( ncid, *dimID, "axis", 1, "X");
    retval = nc_put_att_text( ncid, *dimID, "long_name", long_name.size(), long_name.data());
    return retval;
}
static inline int define_y_dimension( int ncid, int* dimID, const dg::Grid1d& g)
{
    int retval;
    std::string long_name = "y-coordinate in Computational coordinate system";
    if( (retval = define_dimension( ncid, "y", dimID, g))){ return retval;}
    retval = nc_put_att_text( ncid, *dimID, "axis", 1, "Y");
    retval = nc_put_att_text( ncid, *dimID, "long_name", long_name.size(), long_name.data());
    return retval;
}
static inline int define_z_dimension( int ncid, int* dimID, const dg::Grid1d& g)
{
    int retval;
    std::string long_name = "z-coordinate in Computational coordinate system";
    if( (retval = define_dimension( ncid, "z", dimID, g))){ return retval;}
    retval = nc_put_att_text( ncid, *dimID, "axis", 1, "Z");
    retval = nc_put_att_text( ncid, *dimID, "long_name", long_name.size(), long_name.data());
    return retval;
}
}//namespace detail
///@endcond

/**
 * @brief Define a 1d time-dependent dimension variable together with its data points
 *
 * Dimensions are named x, and time
 * @param ncid file ID
 * @param dimsIDs dimension IDs (time, x)
 * @param tvarID time variable ID
 * @param g The 1d DG grid from which data points are generated
 *
 * @return netcdf error code if any
 */
static inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::Grid1d& g)
{
    int retval;
    if( (retval = detail::define_x_dimension( ncid, &dimsIDs[1], g))){ return retval;}
    if( (retval = define_time( ncid, "time", &dimsIDs[0], tvarID)) ){ return retval;}

    return retval;
}
/**
 * @brief Define 2d dimensions and associate values in NetCDF-file
 *
 * Dimensions are named y, x
 * @param ncid file ID
 * @param dimsIDs (write - only) 2D array of dimension IDs (y,x)
 * @param g The 2d grid from which to derive the dimensions
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, const dg::aTopology2d& g)
{
    dg::Grid1d gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d gy( g.y0(), g.y1(), g.n(), g.Ny());
    int retval;
    if( (retval = detail::define_x_dimension( ncid, &dimsIDs[1], gx))){ return retval;}
    if( (retval = detail::define_y_dimension( ncid, &dimsIDs[0], gy))){ return retval;}

    return retval;
}
/**
 * @brief Define 2d time-dependent dimensions and associate values in NetCDF-file
 *
 * Dimensions are named time, y and x
 * @param ncid file ID
 * @param dimsIDs (write - only) 3D array of dimension IDs (time, y,x)
 * @param tvarID (write - only) The ID of the time variable
 * @param g The 2d grid from which to derive the dimensions
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aTopology2d& g)
{
    dg::Grid1d gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d gy( g.y0(), g.y1(), g.n(), g.Ny());
    int retval;
    if( (retval = detail::define_x_dimension( ncid, &dimsIDs[2], gx))){ return retval;}
    if( (retval = detail::define_y_dimension( ncid, &dimsIDs[1], gy))){ return retval;}
    if( (retval = define_time( ncid, "time", &dimsIDs[0], tvarID)) ){ return retval;}

    return retval;
}

/**
 * @brief Define 2d time-dependent (limited) dimensions and associate values in NetCDF-file
 *
 * Dimensions are named x, y, and time (limited)
 * @param ncid file ID
 * @param dimsIDs (write - only) 3D array of dimension IDs (time, y,x)
 * @param size The size of the time variable
 * @param tvarID (write - only) The ID of the time variable
 * @param g The 2d grid from which to derive the dimensions
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_limtime_xy( int ncid, int* dimsIDs, int size, int* tvarID, const dg::aTopology2d& g)
{
    dg::Grid1d gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d gy( g.y0(), g.y1(), g.n(), g.Ny());
    int retval;
    if( (retval = detail::define_x_dimension( ncid, &dimsIDs[2], gx)));
    if( (retval = detail::define_y_dimension( ncid, &dimsIDs[1], gy)));
    if( (retval = define_limited_time( ncid, "time", size, &dimsIDs[0], tvarID)) ){ return retval;}

    return retval;
}
/**
 * @brief Define 3d dimensions and associate values in NetCDF-file
 *
 * Dimensions are named x, y, z
 * @param ncid file ID
 * @param dimsIDs (write - only) 3D array of dimension IDs (z,y,x)
 * @param g The grid from which to derive the dimensions
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, const dg::aTopology3d& g)
{
    dg::Grid1d gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d gy( g.y0(), g.y1(), g.n(), g.Ny());
    dg::Grid1d gz( g.z0(), g.z1(), 1, g.Nz());
    int retval;
    if( (retval = detail::define_x_dimension( ncid, &dimsIDs[2], gx)));
    if( (retval = detail::define_y_dimension( ncid, &dimsIDs[1], gy)));
    if( (retval = detail::define_z_dimension( ncid, &dimsIDs[0], gz)));
    return retval;
}

/**
 * @brief Define 3d time-dependent dimensions and associate values in NetCDF-file
 *
 * Dimensions are named x, y, z, and time
 * @param ncid file ID
 * @param dimsIDs (write - only) 4D array of dimension IDs (time, z,y,x)
 * @param tvarID (write - only) The ID of the time variable
 * @param g The grid from which to derive the dimensions
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aTopology3d& g)
{
    int retval;
    if( (retval = define_dimensions( ncid, &dimsIDs[1], g)) ){ return retval;}
    if( (retval = define_time( ncid, "time", &dimsIDs[0], tvarID)) ){ return retval;}
    return retval;
}



} //namespace file
