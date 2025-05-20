#pragma once
#ifndef _GEOMETRIES_INCLUDED_BY_DG_
#pragma message( "The inclusion of geometries/geometries.h is deprecated. Please use dg/geometries/geometries.h")
#endif //_INCLUDED_BY_DG_

//include grid generators
#include "simple_orthogonal.h"
#include "separatrix_orthogonal.h"
#include "ribeiro.h"
#include "flux.h"
#include "hector.h"
#include "polar.h"
#include "ribeiroX.h"
#include "ds_generator.h"
//include grids
#include "curvilinear.h"
#include "curvilinearX.h"
#include "refined_curvilinearX.h"
#ifdef MPI_VERSION
#include "mpi_curvilinear.h"
#endif

//include magnetic field geometries
#include "solovev.h"
#include "guenter.h"
#include "toroidal.h"
#include "polynomial.h"
#include "taylor.h"
#include "make_field.h"

#include "fluxfunctions.h"
#include "magnetic_field.h"
#include "adaption.h"
#include "sheath.h"

//include average
#include "average.h"
//include ds and fieldaligned
#include "ds.h"
