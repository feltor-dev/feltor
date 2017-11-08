#pragma once
#ifndef MPI_VERSION
#include "backend/average.cuh"
#endif //NOT THE MPI_VERSION
#ifdef MPI_VERSION
#include "backend/average.h"
#endif //MPI_VERSION

/*!@file 
 *
 * @brief This file includes the appropriate headers for parallel derivatives
 */
