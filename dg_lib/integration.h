#ifndef _DG_INTEGRATION_
#define _DG_INTEGRATION_

#include <vector>

#include "evaluation.h"
#include "blas.h"
#include "laplace.h"
#include "dlt.h"

namespace dg
{

template< size_t n>
struct RHS
{
    typedef typename std::vector<std::array<double, n>> Vector;
    RHS(double h, double D):h(h),D(D),lap(h){}
    void operator()( const Vector& y, Vector& yp);
    private:
    double h;
    double D;
    Laplace<n> lap;
};

template<size_t n>
void RHS<n>::operator()( const Vector& y, Vector& yp)
{
    BLAS2<Laplace<n>, Vector>::dsymv( -1., lap, y, 0., yp);
    BLAS2<T, Vector>::dsymv( D, T(h), yp, 0., yp);
}
} // namespace dg

#endif //_DG_INTEGRATION_
