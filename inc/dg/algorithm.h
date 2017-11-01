#pragma once

/*! @file
 * Includes all container independent headers of the dg library.
 *
 * @note include <mpi.h> before this header to activate mpi support
 */
#include "blas.h"
#include "geometry/geometry.h"
#include "helmholtz.h"
#include "cg.h"
#include "functors.h"
#include "multistep.h"
#include "elliptic.h"
#include "runge_kutta.h"
#include "multigrid.h"
#include "refined_elliptic.h"
#include "backend/timer.cuh"
#include "backend/split_and_join.h"
#include "backend/transpose.h"
#include "backend/xspacelib.cuh"
#include "backend/evaluationX.cuh"
#include "backend/derivativesX.h"
#include "backend/weightsX.cuh"
#ifdef MPI_VERSION
#include "arakawa.h"
#include "backend/mpi_init.h"
#endif
