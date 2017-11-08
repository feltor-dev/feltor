#pragma once

#include <cassert>
#include "thrust/host_vector.h"
#include "../backend/evaluation.cuh"
#include "../backend/weights.cuh"
#ifdef MPI_VERSION
#include "../backend/mpi_vector.h"
#include "../backend/mpi_evaluation.h"
#include "../backend/mpi_precon.h"
#endif//MPI_VERSION
#include "base_geometry.h"
//#include "cartesianX.h"
#ifdef MPI_VERSION
#include "mpi_base.h"
#endif//MPI_VERSION
#include "tensor.h"
#include "transform.h"
#include "multiply.h"
