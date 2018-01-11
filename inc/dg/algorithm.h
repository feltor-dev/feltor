#pragma once

/*! @file
 * Includes all container independent headers of the dg library.
 *
 * @note include <mpi.h> before this header to activate mpi support
 */
#include "backend/timer.cuh"
#include "backend/transpose.h"
#include "geometry/split_and_join.h"
#include "geometry/xspacelib.cuh"
#include "geometry/evaluationX.cuh"
#include "geometry/derivativesX.h"
#include "geometry/weightsX.cuh"
#include "geometry/interpolationX.cuh"
#include "geometry/projectionX.h"
#include "geometry/geometry.h"
#include "blas.h"
#include "helmholtz.h"
#include "cg.h"
#include "functors.h"
#include "multistep.h"
#include "elliptic.h"
#include "runge_kutta.h"
#include "multigrid.h"
#include "refined_elliptic.h"
#include "arakawa.h"
#include "poisson.h"
#include "backend/average.cuh"
#ifdef MPI_VERSION
#include "backend/average.h"
#include "backend/mpi_init.h"
#endif
