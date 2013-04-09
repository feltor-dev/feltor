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
    Arakawa( unsigned Nx, unsigned Ny, double hx, double hy);

    void operator()( const container& lhs, const container& rhs, container& result);
  private:
    typedef T value_type;
    //typedef typename VectorTraits< Vector>::value_type value_type;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Matrix bdx, bdy, forward;
    container dxlhs, dylhs, dxrhs, dyrhs;
};

template< class T, size_t n, class container, class MemorySpace>
Arakawa<T, n, container, MemorySpace>::Arakawa( unsigned Nx, unsigned Ny, double hx, double hy): dxlhs( n*n*Nx*Ny), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs)
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> HMatrix;
    Operator<value_type, n> backward1d( DLT<n>::backward);
    Operator<value_type, n> forward1d( DLT<n>::forward);
    Operator<value_type, n*n> backward2d = tensor( backward1d, backward1d);
    Operator<value_type, n*n> forw = tensor( forward1d, forward1d);
    std::cout << "ping\n";

    HMatrix b = tensor( Nx*Ny, backward2d);
    forward = tensor<T,n>( tensor(Ny, forw),  tensor( Nx, forw));
    std::cout << "ping\n";
    HMatrix dx = tensor<T,n>( tensor<T,n>( Ny, delta), create::dx_per<value_type,n>( Nx, hx));
    HMatrix dy = tensor<T,n>( create::dx_per<value_type,n>( Ny, hy), tensor<T,n>(Nx, delta));
    HMatrix bdx_(dx), bdy_(dy);
    std::cout << dx.num_rows << "\n";
    std::cout << dy.num_rows << "\n";
    std::cout << b.num_rows << "\n";
    std::cout << "Hello ping\n";
    cusp::multiply( b, dx, bdx_);
    std::cout << "Hello ping\n";
    cusp::multiply( b, dy, bdy_);
    bdx = bdx_;
    bdy = bdy_;
    std::cout << "ping\n";
}

template< class T, size_t n, class container, class MemorySpace>
void Arakawa<T, n, container, MemorySpace>::operator()( const container& lhs, const container& rhs, container& result)
{
    std::cout<<"A1\n";
    blas2::symv( bdx, lhs, dxlhs);
    blas2::symv( bdy, lhs, dylhs);
    blas2::symv( bdx, rhs, dxrhs);
    blas2::symv( bdy, rhs, dyrhs);
    std::cout<<"A2\n";
    cudaThreadSynchronize();
    blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    blas1::pointwiseDot( dxrhs, dylhs, dylhs);
    cudaThreadSynchronize();
    std::cout<<"A3\n";
    blas1::axpby( 1., dyrhs, -1., dylhs);
    cudaThreadSynchronize();
    std::cout<<"A4\n";
    blas2::symv( forward, dylhs, result);
    cudaThreadSynchronize();
    std::cout<<"A5\n";
}

}//namespace dg

#endif //_DG_ARAKAWA_CUH
