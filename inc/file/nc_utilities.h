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
 * @note By netcdf conventions a variable with the same name as a dimension is called a coordinate variable.
 * @param ncid file ID
 * @param dimID dimension ID (output)
 * @param g The 1d DG grid from which data points are generated (input)
 * @param name_dim Name of dimension (input)
 * @param axis The axis attribute (input), ("X", "Y" or "Z")
 *
 * @return netcdf error code if any
 */
static inline int define_dimension( int ncid, int* dimID, const dg::Grid1d& g, std::string name_dim = "x", std::string axis = "X")
{
    int retval;
    std::string long_name = name_dim+"-coordinate in Computational coordinate system";
    thrust::host_vector<double> points = dg::create::abscissas( g);
    if( (retval = nc_def_dim( ncid, name_dim.data(), points.size(), dimID)) ) { return retval;}
    int varID;
    if( (retval = nc_def_var( ncid, name_dim.data(), NC_DOUBLE, 1, dimID, &varID))){return retval;}
    if( (retval = nc_enddef(ncid)) ) {return retval;} //not necessary for NetCDF4 files
    if( (retval = nc_put_var_double( ncid, varID, points.data())) ){ return retval;}
    if( (retval = nc_redef(ncid))) {return retval;} //not necessary for NetCDF4 files
    retval = nc_put_att_text( ncid, *dimID, "axis", axis.size(), axis.data());
    retval = nc_put_att_text( ncid, *dimID, "long_name", long_name.size(), long_name.data());
    return retval;
}

/**
 * @brief Define a 1d time-dependent dimension variable together with its data points
 *
 * Dimensions have attribute of (time, X)
 * @param ncid file ID
 * @param dimsIDs dimension IDs (time, X)
 * @param tvarID time variable ID (unlimited)
 * @param g The 1d DG grid from which data points are generated
 * @param name_dims Names for the dimension variables
 *
 * @return netcdf error code if any
 */
static inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::Grid1d& g, std::array<std::string,2> name_dims = {"time","x"})
{
    int retval;
    retval = define_time( ncid, name_dims[0].data(), &dimsIDs[0], tvarID);
    if(retval)
        return retval;
    return define_dimension( ncid, &dimsIDs[1], g, name_dims[1], "X");
}
/**
 * @brief Define 2d dimensions and associate values in NetCDF-file
 *
 * Dimensions have attributes of (Y, X)
 * @param ncid file ID
 * @param dimsIDs (write - only) 2D array of dimension IDs (Y,X)
 * @param g The 2d grid from which to derive the dimensions
 * @param name_dims Names for the dimension variables
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, const dg::aTopology2d& g, std::array<std::string,2> name_dims = {"y", "x"})
{
    dg::Grid1d gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d gy( g.y0(), g.y1(), g.n(), g.Ny());
    int retval;
    retval = define_dimension( ncid, &dimsIDs[0], gy, name_dims[0], "Y");
    if(retval)
        return retval;
    return define_dimension( ncid, &dimsIDs[1], gx, name_dims[1], "X");
}
/**
 * @brief Define 2d time-dependent dimensions and associate values in NetCDF-file
 *
 * Dimensions have attributes of (time, Y, X)
 * @param ncid file ID
 * @param dimsIDs (write - only) 3D array of dimension IDs (time, Y,X)
 * @param tvarID (write - only) The ID of the time variable ( unlimited)
 * @param g The 2d grid from which to derive the dimensions
 * @param name_dims Names for the dimension variables ( time, Y, X)
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aTopology2d& g, std::array<std::string,3> name_dims = {"time", "y", "x"})
{
    int retval;
    retval = define_time( ncid, name_dims[0].data(), &dimsIDs[0], tvarID);
    if(retval)
        return retval;
    return define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2]});
}

/**
 * @brief Define 2d time-dependent (limited) dimensions and associate values in NetCDF-file
 *
 * Dimensions have attributes of (time, Y, X)
 * @param ncid file ID
 * @param dimsIDs (write - only) 3D array of dimension IDs (time, Y,X)
 * @param size The size of the time variable
 * @param tvarID (write - only) The ID of the time variable (limited)
 * @param g The 2d grid from which to derive the dimensions
 * @param name_dims Names for the dimension variables (time, Y, X)
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_limtime_xy( int ncid, int* dimsIDs, int size, int* tvarID, const dg::aTopology2d& g, std::array<std::string, 3> name_dims = {"time", "y", "x"})
{
    int retval;
    retval = define_limited_time( ncid, name_dims[0].data(), size, &dimsIDs[0], tvarID);
    if(retval)
        return retval;
    return define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2]});
}
/**
 * @brief Define 3d dimensions and associate values in NetCDF-file
 *
 * Dimensions have attributes ( Z, Y, X)
 * @param ncid file ID
 * @param dimsIDs (write - only) 3D array of dimension IDs (Z,Y,X)
 * @param g The grid from which to derive the dimensions
 * @param name_dims Names for the dimension variables ( Z, Y, X)
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, const dg::aTopology3d& g, std::array<std::string, 3> name_dims = {"z", "y", "x"})
{
    dg::Grid1d gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d gy( g.y0(), g.y1(), g.n(), g.Ny());
    dg::Grid1d gz( g.z0(), g.z1(), 1, g.Nz());
    int retval;
    retval = define_dimension( ncid, &dimsIDs[0], gz, name_dims[0], "Z");
    if(retval)
        return retval;
    retval = define_dimension( ncid, &dimsIDs[1], gy, name_dims[1], "Y");
    if(retval)
        return retval;
    return define_dimension( ncid, &dimsIDs[2], gx, name_dims[2], "X");
}

/**
 * @brief Define 3d time-dependent dimensions and associate values in NetCDF-file
 *
 * Dimensions have attributes ( time, Z, Y, X)
 * @param ncid file ID
 * @param dimsIDs (write - only) 4D array of dimension IDs (time, Z,Y,X)
 * @param tvarID (write - only) The ID of the time variable ( unlimited)
 * @param g The grid from which to derive the dimensions
 * @param name_dims Names for the dimension variables ( time, Z, Y, X)
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
static inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aTopology3d& g, std::array<std::string, 4> name_dims = {"time", "z", "y", "x"})
{
    int retval;
    retval = define_time( ncid, "time", &dimsIDs[0], tvarID);
    if(retval)
        return retval;
    return define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2], name_dims[3]});
}



} //namespace file
