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
namespace dg
{

//should there be a utility for W2D?
/*
//are these really necessary?
template< class Vector, size_t n>
typename Vector::value_type dot( const Vector& x, const Vector& y, const Grid<typename Vector::value_type, n>& g)
{
    return blas2::dot( x, W2D<typename Vector::value_type, n>(g.hx(), g.hy()), y);
}
template< class Vector, size_t n>
typename Vector::value_type nrml2( const Vector& x, const Grid<typename Vector::value_type, n>& g)
{
    return sqrt(blas2::dot( W2D<typename Vector::value_type, n>(g.hx(), g.hy()), x));
}
template< class Vector, size_t n>
typename Vector::value_type integ( const Vector& x, const Grid<typename Vector::value_type, n>& g)
{
    Vector one(x.size(), 1.);
    return dot( x, one, g);
}
*/


}//namespace dg

#endif // _DG_XSPACELIB_CUH_
