#ifndef _DG_POLARISATION_CUH
#define _DG_POLARISATION_CUH

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>

#include "functions.h"
#include "operator.cuh"
#include "operator_matrix.cuh"
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
    typedef cusp::coo_matrix<int, T, Memory> Matrix;
    typedef cusp::array1d<T, Memory> Vector;
    Polarisation( unsigned N, T h, int bc);
    Matrix create( const Vector& );
  private:
    typedef cusp::array1d<int, Memory> Array;
    Matrix left, middle, right, jump;
    cusp::array1d<int, Memory> I, J;
    Vector xspace;
    unsigned N;

};
template <class T, size_t n, class Memory>
Polarisation<T,n, Memory>::Polarisation( unsigned N, T h, int bc): I(n*N), J(I), xspace( n*N), N(N)
{
    for( unsigned i=0; i<n*N; i++)
        I[i] = J[i] = i;
    right = create::dx_asymm_mt<T,n>( N, h, bc); //create and transfer to device
    Operator<T, n> backward( DLT<n>::backward);
    middle = tensor<T,n>( N, backward);
    cusp::multiply( middle, right, right);
    cusp::transpose( right, left); 
    Operator<T,n> weights(0.);
    for( unsigned i=0; i<n; i++)
        weights(i,i) = DLT<n>::weight[i]*h/2.; // normalisation because F is invariant
    middle = tensor<T,n>( N, weights*backward);
    jump = create::jump_ot<T,n>( N, bc); //without jump cg is unstable

}

template< class T, size_t n, class Memory>
cusp::coo_matrix<int, T, Memory> Polarisation<T,n, Memory>::create( const Vector& chi)
{
    Matrix laplace;
    cusp::multiply( middle, chi, xspace);
    cusp::coo_matrix_view<Array, Array, Vector,  int, T, Memory> chi_view( n*N, n*N, n*N, I, J, xspace);
    cusp::multiply( chi_view, right, laplace);
    cusp::multiply( left, laplace, laplace);
    cusp::add( laplace, jump, laplace);
    return laplace;
}

} //namespace dg

#endif // _DG_POLARISATION_CUH
