#pragma once

#include <cassert>
#include "thrust/host_vector.h"
#include "evaluation.h"
#include "weights.h"
#include "filter.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "mpi_evaluation.h"
#include "mpi_weights.h"
#endif//MPI_VERSION
#include "base_geometry.h"
#include "base_geometryX.h"
#include "refined_gridX.h"
//#include "cartesianX.h"
#ifdef MPI_VERSION
#include "mpi_base_geometry.h"
#endif//MPI_VERSION
#include "tensor.h"
#include "transform.h"
#include "multiply.h"
// finite element method
#include "fem.h"
