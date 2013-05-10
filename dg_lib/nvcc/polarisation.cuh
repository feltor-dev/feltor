#ifndef _DG_POLARISATION_CUH
#define _DG_POLARISATION_CUH

#include <cusp/coo_matrix.h>

#include "functions.h"
#include "operator.cuh"
#include "creation.cuh"
#include "dx.cuh"
#include "dlt.h"

namespace dg
{

//as far as i see in the source code cusp only supports coo - coo matrix
//-matrix multiplication
template< class T, size_t n, class Memory>
struct Polarisation
{
    typedef cusp::coo_matrix<int, T, Memory> Martrix;
    typedef cusp::array1d<T, Memory> Vector;
    Polarisation( unsigned N, T h, int bc);
    DMatrix create( const Vector& );
  private:
    DMatrix left, middle, right;
    cusp::array1d<int, Memory> I, J;
    unsigned N;

};
template <class T, size_t n, class Memory>
Polarisation<T,n>::Polarisation( unsigned N, T h, int bc): I(N, 1), J(I), N(N)
{
    right = create::dx_asymm_mt( N, h, bc); //create and transfer to device
    Operator<T, n> backward( DLT<n>::backward);
    middle = tensor<T,n>( N, backward);
    cusp::multiply( middle, right, right);
    cusp::transpose( right, left); 
    Operator<T,n> weights(0);
    for( unsigned i=0; i<n; i++)
        weights(i,i) = -DLT<n>::weight[i]; //one minus because left should be -right^T
    middle = tensor<T,n>( N, weights*backward);
    cusp::multiply( left, middle, left); 

}

template< class T, size_t n, class Memory>
DMatrix Polarisation<T,n>::create( const Vector& n)
{
    DMatrix laplace;
    cusp::coo_matrix_view<int, T, Memory> n_view( N, N, N, I, J, n);
    cusp::multiply( n_view, right, laplace);
    cusp::multiply( left, laplace, laplace);
    return laplace;
}

} //namespace dg

#endif // _DG_POLARISATION_CUH
