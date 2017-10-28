#pragma once

//include grids
#include "curvilinear.h"
#ifdef MPI_VERSION
#include "mpi_curvilinear.h"
#endif

//include grid generators
#include "simple_orthogonal.h"
#include "ribeiro.h"
#include "flux.h"
#include "hector.h"
#include "polar.h"

//include magnetic field geometries
#include "solovev.h"
#include "guenther.h"
#include "toroidal.h"

#include "init.h"
#include "fluxfunctions.h"
#include "magnetic_field.h"
#include "adaption.h"

//include average
#include "average.h"
//include ds and fieldaligned
#include "ds.h"
