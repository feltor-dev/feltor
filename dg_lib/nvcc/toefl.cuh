#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"
#include "vector_traits.h"

#include "arakawa.cuh"
#include "laplace.cuh"
#include "functions.h"
#include "tensor.cuh"
#include "operator_matrix.cuh"
#include "dx.cuh"
#include "cg.cuh"

namespace dg
{

template< class T, size_t n, class container=thrust::device_vector<T>, class MemorySpace = cusp::device_memory>
struct Toefl
{
    typedef std::vector<container> Vector;
    Toefl( unsigned Nx, unsigned Ny, double hx, double hy, double a, double mu, double kappa, double d, double g, double eps = 1e-6);

    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    typedef T value_type;
    //typedef typename VectorTraits< Vector>::value_type value_type;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Matrix dy;
    Matrix laplace;
    container rho, phi, dyne, dyphi;
    Arakawa<T, n, container, MemorySpace> arakawa; 
    CG<Matrix, container, dg::T2D<T, n> > pcg;
    
    double hx, hy;
    double a, mu, kappa, d, g;
    double eps; 
};

template< class T, size_t n, class container, class MemorySpace>
Toefl<T, n, container, MemorySpace>::Toefl( unsigned Nx, unsigned Ny, double hx, double hy,
        double a, double mu, double kappa, double d, double g, double eps): 
    rho( n*n*Nx*Ny, 0.), phi(rho), dyne(rho), dyphi(rho), 
    arakawa( Nx, Ny, hx, hy), 
    pcg( rho, n*n*Nx),// n*n*Nx*Ny),
    hx( hx), hy(hy), a(a), mu(mu), kappa(kappa), d(d), g(g), eps(eps)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    HMatrix A = dg::dgtensor<double, n>
                             ( dg::create::laplace1d_per<double, n>( Ny, hy), 
                               dg::S1D<double, n>( hx),
                               dg::S1D<double, n>( hy),
                               dg::create::laplace1d_per<double, n>( Nx, hx)); 
    laplace = A;
    //create derivatives
    HMatrix dy_ = dgtensor<T,n>( create::dx_per<value_type,n>( Ny, hy), tensor<T,n>(Nx, delta));
    dy = dy_;

}

template< class T, size_t n, class container, class MemorySpace>
void Toefl<T, n, container, MemorySpace>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //rho = (ni-ne)/a_imu_i
    blas1::axpby( -1., y[0], 0., rho);
    cudaThreadSynchronize();
    blas1::axpby( 1./a/mu, y[1], 1./a/mu, rho);
    cudaThreadSynchronize();
    //compute S rho
    blas2::symv( S2D<double, n>(hx, hy), rho, rho);
    ArrVec2d<T, n, thrust::host_vector<T> > view(rho, 10);
    std::cout << view <<std::endl;
    ArrVec2d<T, n, thrust::host_vector<T> > view2(phi, 10);
    std::cout << view2 <<std::endl;
    cudaThreadSynchronize();
    std::cout << "Number of pcg iterations "<< pcg( laplace, phi, rho, T2D<double, n>(hx, hy), eps)<<std::endl;
    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi, yp[i]);

    cudaThreadSynchronize();
    blas2::symv( dy, phi, dyphi);
    blas2::symv( dy, y[0], dyne);
    cudaThreadSynchronize();
    blas1::axpby( kappa+g, dyphi, 1., yp[0]);
    blas1::axpby( kappa+g, dyphi, 1., yp[1]);

    blas1::axpby( -kappa, dyne, 1., yp[0]);
    cudaThreadSynchronize();
    blas1::axpby( d, phi, 1., yp[0]);
    cudaThreadSynchronize();
    blas1::axpby( -d, y[0], 1., yp[0]);

}

}//namespace dg

#endif //_DG_TOEFL_CUH
