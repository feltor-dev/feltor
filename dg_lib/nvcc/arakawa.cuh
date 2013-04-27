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
    Arakawa( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy);

    void operator()( const container& lhs, const container& rhs, container& result);
  private:
    typedef T value_type;
    //typedef typename VectorTraits< Vector>::value_type value_type;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Matrix bdx, bdy, dxf, dyf, forward, backward;
    container dxlhs, dylhs, dxrhs, dyrhs, blhs, brhs;
};

template< class T, size_t n, class container, class MemorySpace>
Arakawa<T, n, container, MemorySpace>::Arakawa( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy): dxlhs( n*n*Nx*Ny), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), blhs( n*n*Nx*Ny), brhs( blhs)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;

    //create forward dlt matrix
    Operator<value_type, n> forward1d( DLT<n>::forward);
    Operator<value_type, n*n> forward2d = tensor( forward1d, forward1d);
    forward = tensor( Nx*Ny, forward2d);

    //create backward dlt matrix
    Operator<value_type, n> backward1d( DLT<n>::backward);
    Operator<value_type, n*n> backward2d = tensor( backward1d, backward1d);
    backward = tensor( Nx*Ny, backward2d);

    //create derivatives
    HMatrix dx = dgtensor<T,n>( tensor<T,n>( Ny, delta), create::dx_symm<value_type,n>( Nx, hx, bcx));
    HMatrix dy = dgtensor<T,n>( create::dx_symm<value_type,n>( Ny, hy, bcy), tensor<T,n>(Nx, delta));

    HMatrix bdx_(dx), bdy_(dy);
    cusp::multiply( backward, dx, bdx_);
    cusp::multiply( backward, dy, bdy_);
    HMatrix dxf_(dx), dyf_(dy);
    cusp::multiply( dx, forward, dxf_);
    cusp::multiply( dy, forward, dyf_);
    bdx = bdx_;
    bdy = bdy_;
    dxf = dxf_; 
    dyf = dyf_;

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
    //compute derivatives in x-space
    blas2::symv( bdx, lhs, dxlhs);
    blas2::symv( bdy, lhs, dylhs);
    blas2::symv( bdx, rhs, dxrhs);
    blas2::symv( bdy, rhs, dyrhs);
    //transform to x-space
    blas2::symv( backward, lhs, blhs);
    blas2::symv( backward, rhs, brhs);
    cudaThreadSynchronize();

    // order is important now
    // +x (1) -> result und (2) -> blhs
    blas1::pointwiseDot( blhs, dyrhs, result);
    blas1::pointwiseDot( blhs, dxrhs, blhs);
    cudaThreadSynchronize();

    // ++ (1) -> dyrhs and (2) -> dxrhs
    blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    blas1::pointwiseDot( dylhs, dxrhs, dxrhs);
    cudaThreadSynchronize();

    // x+ (1) -> dxlhs and (2) -> dylhs
    blas1::pointwiseDot( dxlhs, brhs, dxlhs);
    blas1::pointwiseDot( dylhs, brhs, dylhs);
    cudaThreadSynchronize();

    blas1::axpby( 1./3., dyrhs, -1./3., dxrhs);  //dxl*dyr - dyl*dxr -> dxrhs
    //everything which needs a dx 
    blas1::axpby( 1./3., dxlhs, -1./3., blhs);   //dxl*r - l*dxr     -> blhs 
    //everything which needs a dy
    blas1::axpby( 1./3., result, -1./3., dylhs); //l*dyr - dyl*r     -> dylhs

    //blas1::axpby( 1., dyrhs,  -1., dxrhs);
    ////for testing purposes (note that you need to set criss-cross)
    //blas1::axpby( 0., dxlhs,  -0., blhs);
    //blas1::axpby( 0., result, -0., dylhs);

    cudaThreadSynchronize();
    blas2::symv( forward, dxrhs, dyrhs);  //back ++                 -> dyrhs
    blas2::symv( dyf, blhs, result);      //dy*back*(dxl*r - l*dxr) -> result
    blas2::symv( dxf, dylhs, dxlhs);      //dx*back*(l*dyr - dyl*r) -> dxlhs
    //now sum everything up
    cudaThreadSynchronize();
    blas1::axpby( 1., dxlhs, 1., result); //result + dxlhs -> result
    cudaThreadSynchronize();
    blas1::axpby( 1., dyrhs, 1., result); //result + dyrhs -> result
}

}//namespace dg

#endif //_DG_ARAKAWA_CUH
