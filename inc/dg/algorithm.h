#pragma once

/*! @file
 * Includes all container independent headers of the dg library.
 *
 * @note include <mpi.h> before this header to activate mpi support
 */
#include "backend/timer.h"
#include "backend/transpose.h"
#include "topology/split_and_join.h"
#include "topology/xspacelib.h"
#include "topology/evaluationX.h"
#include "topology/derivativesX.h"
#include "topology/weightsX.h"
#include "topology/interpolationX.h"
#include "topology/projectionX.h"
#include "topology/geometry.h"
#include "blas.h"
#include "helmholtz.h"
#include "cg.h"
#include "bicgstabl.h"
#include "lgmres.h"
#include "functors.h"
#include "multistep.h"
#include "elliptic.h"
#include "runge_kutta.h"
#include "adaptive.h"
#include "multigrid.h"
#include "refined_elliptic.h"
#include "arakawa.h"
#include "advection.h"
#include "poisson.h"
#include "simpsons.h"
#include "topology/average.h"
#ifdef MPI_VERSION
#include "topology/average_mpi.h"
#include "backend/mpi_init.h"
#endif
