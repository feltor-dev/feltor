#ifndef _DG_BLAS_LAPLACE_CUH
#define _DG_BLAS_LAPLACE_CUH

#include <cusp/multiply.h>
#include <cusp/array1d.h>

#include "../blas.h"
#include "../laplace.cuh"

namespace dg{

    
    /*
template <size_t n>
struct BLAS2< Laplace<n>, thrust::device_vector<double> >
{
    typedef Laplace<n> Matrix;
    typedef thrust::device_vector<double> Vector;
    typedef cusp::array1d<double, cusp::device_memory> CuspVector;
    static void dsymv( const Matrix& m, const Vector& x, Vector& y)
    {
        //assert( &x != &y); //does cusp do this assert?
        cusp::array1d_view< Vector::const_iterator> cx( x.cbegin(), x.cend());
        cusp::array1d_view< Vector::iterator> cy( y.begin(), y.end());
        const typename Matrix::DMatrix& dm = m.data();
        cusp::multiply( dm, cx, cy);
    }
};
*/

template< class CuspMatrix, class ThrustVector>
void BLAS2<CuspMatrix, ThrustVector>::dsymv( const Matrix& m, const Vector&x, Vector& y)
{
    cusp::array1d_view< typename Vector::const_iterator> cx( x.cbegin(), x.cend());
    cusp::array1d_view< typename Vector::iterator> cy( y.begin(), y.end());
    cusp::multiply( m, cx, cy);
}

} //namespace dg

#endif //_DG_BLAS_LAPLACE_CUH
