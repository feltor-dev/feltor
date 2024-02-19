#pragma once
#ifndef _FILE_INCLUDED_BY_DG_
#pragma message( "The inclusion of file/nc_utilities.h is deprecated. Please use dg/file/nc_utilities.h")
#endif //_INCLUDED_BY_DG_

#include <netcdf.h>
#include "nc_error.h"
#include "easy_dims.h"
#include "easy_output.h"
#include "easy_input.h"
//#include "writer.h"

/*!@file
 *
 * Meta-file to gather all netcdf input files
 *
 * @defgroup netcdf NetCDF utilities
 * \#include "dg/file/nc_utilities.h" (link -lnetcdf -lhdf5[_serial] -lhdf5[_serial]_hl)
 * @{
 *      @defgroup Dimensions Dimension utilities
 *      @defgroup Attributes Json as Attributes utilities
 *      @defgroup Input Read variable utilities
 *      @defgroup Output Write variable utilities
 * @}
 *
 *
 */


namespace dg
{
/**
* @brief Namespace for netCDF output related classes and functions following the
 <a href="http://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html">CF-conventions</a>
 and
 <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">netCDF conventions</a>
 @sa @ref json and @ref netcdf
*/
namespace file
{

} //namespace file
} //namespace dg
