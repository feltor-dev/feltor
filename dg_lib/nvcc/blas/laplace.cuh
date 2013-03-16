#ifndef _DG_BLAS_LAPLACE_CUH
#define _DG_BLAS_LAPLACE_CUH

#include <cusp/multiply.h>
#include <cusp/array1d.h>

#include "thrust_vector.cuh"

#include "../blas.h"
#include "../laplace.cuh"

namespace dg{

template <size_t n>
struct BLAS2< Laplace<n>, thrust::device_vector<double> >
{
    typedef Laplace<n> Matrix;
    typedef thrust::device_vector<double> Vector;
    typedef cusp::array1d<double, cusp::device_memory> CuspVector;
    static void dsymv( const Matrix& m, const Vector& x, Vector& y)
    {
        //assert( &x != &y); 
        cusp::array1d_view< Vector::const_iterator> cx( x.cbegin(), x.cend());
        cusp::array1d_view< Vector::iterator> cy( y.begin(), y.end());
        const typename Matrix::DMatrix& dm = m.get_m();
        cusp::multiply( dm, cx, cy);
    }
};

} //namespace dg

#endif //_DG_BLAS_LAPLACE_CUH
