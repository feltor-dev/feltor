#pragma once
#ifndef _FILE_NC_INCLUDED_BY_DG_
#pragma message( "The inclusion of file/nc_utilities.h is deprecated. Please use dg/file/nc_utilities.h")
#endif //_INCLUDED_BY_DG_

#include <netcdf.h>
#include "nc_error.h"
#include "easy_dims.h"
#include "easy_output.h"
#include "easy_input.h"
#ifdef MPI_VERSION
#include "nc_mpi_file.h"
#endif //MPI_VERSION
#include "nc_file.h"

/*!@file
 *
 * Meta-file to gather all netcdf input files
 *
 */
