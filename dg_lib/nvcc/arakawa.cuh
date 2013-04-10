#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"
#include "vector_traits.h"

#include "functions.h"
#include "tensor.cuh"
#include "operator_matrix.cuh"
#include "dx.cuh"

namespace dg
{

template< class T, size_t n, class container=thrust::device_vector<T>, class MemorySpace = cusp::device_memory>
struct Arakawa
{
    Arakawa( unsigned Nx, unsigned Ny, double hx, double hy, container test = container());

    void operator()( const container& lhs, const container& rhs, container& result);
  private:
    typedef T value_type;
    //typedef typename VectorTraits< Vector>::value_type value_type;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Matrix bdx, bdy, forward;
    container dxlhs, dylhs, dxrhs, dyrhs;
};

template< class T, size_t n, class container, class MemorySpace>
Arakawa<T, n, container, MemorySpace>::Arakawa( unsigned Nx, unsigned Ny, double hx, double hy, container test): dxlhs( n*n*Nx*Ny), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs)
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> HMatrix;

    //create forward dlt matrix
    Operator<value_type, n> forward1d( DLT<n>::forward);
    Operator<value_type, n*n> forward2d = tensor( forward1d, forward1d);
    forward = tensor( Nx*Ny, forward2d);

    //create backward dlt matrix
    Operator<value_type, n> backward1d( DLT<n>::backward);
    Operator<value_type, n*n> backward2d = tensor( backward1d, backward1d);
    HMatrix backward = tensor( Nx*Ny, backward2d);

    //create derivatives
    HMatrix dx = dgtensor<T,n>( tensor<T,n>( Ny, delta), create::dx_per<value_type,n>( Nx, hx));
    HMatrix dy = dgtensor<T,n>( create::dx_per<value_type,n>( Ny, hy), tensor<T,n>(Nx, delta));
    HMatrix bdx_(dx), bdy_(dy);
    cusp::multiply( backward, dx, bdx_);
    cusp::multiply( backward, dy, bdy_);
    bdx = bdx_;
    bdy = bdy_;

    /*
    //test is really the same
    ArrVec2d<T, n, container> direct( test, Nx), indirect1(direct), indirect2(direct);
    blas2::symv( dy, test, direct.data());
    std::cout << "Direct\n"<<direct;
    blas2::symv( bdy, test, indirect1.data());
    blas2::symv( bdx, test, indirect2.data());
    std::cout << "InDirect1\n"<<indirect1;
    std::cout << "InDirect2\n"<<indirect2;
    blas1::pointwiseDot( indirect1.data(), indirect2.data(), indirect2.data());
    std::cout << "InDirect12\n"<<indirect2;
    blas2::symv( forward, indirect2.data(), test);
    indirect2.data() = test;
    std::cout << "InDirect\n"<<indirect2;
    */
}

template< class T, size_t n, class container, class MemorySpace>
void Arakawa<T, n, container, MemorySpace>::operator()( const container& lhs, const container& rhs, container& result)
{
    //probably not consistent
    blas2::symv( bdx, lhs, dxlhs);
    blas2::symv( bdy, lhs, dylhs);
    blas2::symv( bdx, rhs, dxrhs);
    blas2::symv( bdy, rhs, dyrhs);
    cudaThreadSynchronize();
    blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    blas1::pointwiseDot( dxrhs, dylhs, dylhs);
    cudaThreadSynchronize();
    blas1::axpby( 1., dyrhs, -1., dylhs);
    cudaThreadSynchronize();
    blas2::symv( forward, dylhs, result);
    cudaThreadSynchronize();
}

}//namespace dg

#endif //_DG_ARAKAWA_CUH
