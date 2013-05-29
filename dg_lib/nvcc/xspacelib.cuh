#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>

//functions for evaluation
#include "grid.cuh"
#include "arrvec2d.cuh"
#include "functors.cuh"
#include "dlt.h"
#include "evaluation.cuh"


//creational functions
#include "derivatives.cuh"
#include "arakawa.cuh"
#include "polarisation.cuh"

//integral functions
#include "preconditioner.cuh"

#include "typedefs.cuh"

#endif // _DG_XSPACELIB_CUH_
