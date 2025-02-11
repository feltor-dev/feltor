#pragma once

#ifndef _FILE_INCLUDED_BY_DG_
#pragma message( "The inclusion of file/file.h is deprecated. Please use dg/file/file.h")
#endif //_INCLUDED_BY_DG_

#include "json_utilities.h"
#include "nc_utilities.h"
#include "probes.h"

/*!@file
 *
 * Combined json and netcdf utilities
 *
 * @defgroup section Json utilities
 * \#include "dg/file/json_utilities.h" (link -ljsoncpp )
 * @{
 *  @defgroup wrapper The WrappedJsonValue class
 *  @defgroup json Utility functions
 * @}
 *
 * If the Macro \c DG_USE_JSONHPP is defined before inclusion then the (header-only) nlohmann-json library is used
 * instead of jsoncpp
 * @defgroup high A high level C++ interface to NetCDF-4
 * \#include "dg/file/nc_utilities.h" (link -lnetcdf -lhdf5[_serial] -lhdf5[_serial]_hl)
 * @{
 *  @defgroup ncfile The dg::file::NcFile
 *  @{
 *      @defgroup netcdf Utilitiy classes and functions
 *  @}
 *  @defgroup probes The dg::file::Probes module
 * @}
 *
 * @defgroup legacy Legacy NetCDF-C utility
 */

namespace dg
{
/**
* @brief Namespace for Json and NetCDF I/O related classes and functions
*
* The NetCDF files follow the
 <a href="http://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html">CF-conventions</a>
 and
 <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">netCDF conventions</a>
 @sa @ref json and @ref netcdf
*/
namespace file
{

} //namespace file
} //namespace dg
