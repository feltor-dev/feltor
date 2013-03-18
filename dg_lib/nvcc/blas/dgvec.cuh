#ifndef _DG_BLAS_DGVEC_
#define _DG_BLAS_DGVEC_

#include "../blas.h"
#include "thrust_vector.cuh"

namespace dg{

//??
template<> 
template<class ThrustView>
double BLAS1<ThrustView>::ddot( const Vector& x, const Vector& y)
{
    return BLAS1< ThrustView::Vector>::ddot( x, y);
}
template<> 
template<class ThrustView>
void BLAS1<ThrustView>::daxpby( double alpha, const Vector& x, double beta, Vector& y)
{
    BLAS1< ThrustView::Vector>::daxpby( alpha, x, beta, y);
}

} //namespace dg

#endif // _DG_BLAS_DGVEC
