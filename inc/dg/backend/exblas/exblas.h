#pragma once

#include "exdot_serial.h"
#include "fpedot_serial.h"
#include "thrust/device_vector.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "exdot_cuda.cuh" // accumulate.cuh , config.h, mylibm.cuh
#include "fpedot_cuda.cuh"
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include "exdot_omp.h" //accumulate.h, mylibm.hpp
#include "fpedot_omp.h"
#endif

#ifdef MPI_VERSION
#include "mpi_accumulate.h"
#endif //MPI_VERSION
