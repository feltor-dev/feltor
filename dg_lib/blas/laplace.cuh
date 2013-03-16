#ifndef _DG_BLAS_LAPLACE_CUH
#define _DG_BLAS_LAPLACE_CUH

#include <cusp/multiply.h>

#include "thrust_vector.cuh"

#include "../blas.h"
#include "../laplace.cuh"
#include "../array.cuh"

namespace dg{

template <size_t n>
struct BLAS2< Laplace<n>, thrust::device_vector<Array<double,n> > >
{
    typedef Laplace<n> Matrix;
    typedef thrust::device_vector<Array<double,n>> Vector;
    static void dsymv( const Matrix& m, const Vector& x, Vector& y)
    {
        assert( &x != &y); 
        const m::DMatrix& dm = m.get_m();
        cusp::multiply( dm, x, y);

    }
    static void dsymv( const Matrix& m, const Vector& x, Vector& y)
    {
        dsymv( 1., m, x, 0., y);
    }
};

} //namespace dg

#endif //_DG_BLAS_LAPLACE_CUH
