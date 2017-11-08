#pragma once
#ifndef MPI_VERSION
#include "backend/average.cuh"
#endif //NOT THE MPI_VERSION
#ifdef MPI_VERSION
#include "backend/average.h"
#endif //MPI_VERSION
///@deprecated This header is deprecated in favour of algorithm.h
