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

    //Garcia equations with switched x <-> y  and phi -> -phi
template< class T, size_t n, class container=thrust::device_vector<T>, class MemorySpace = cusp::device_memory>
struct Toefl
{
    typedef std::vector<container> Vector;
    Toefl( unsigned Nx, unsigned Ny, double hx, double hy, double R, double P, double eps = 1e-6);

    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    typedef T value_type;
    //typedef typename VectorTraits< Vector>::value_type value_type;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Matrix dy;
    Matrix laplace;
    container omega, phi, dxtheta, dxphi;
    Arakawa<T, n, container, MemorySpace> arakawa; 
    CG<Matrix, container, dg::T2D<T, n> > pcg;
    
    double hx, hy;
    double Ra, Pr;
    double eps; 
};

template< class T, size_t n, class container, class MemorySpace>
Toefl<T, n, container, MemorySpace>::Toefl( unsigned Nx, unsigned Ny, double hx, double hy,
        double R, double P, double eps): 
    omega( n*n*Nx*Ny, 0.), phi(omega), dxtheta(omega), dxphi(omega), 
    arakawa( Nx, Ny, hx, hy, -1, 0), 
    pcg( omega, n*n*Nx*Ny),
    hx( hx), hy(hy), Ra (R), Pr(P), eps(eps)
{
    //typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    laplace = dg::dgtensor<double, n>
                             ( dg::create::laplace1d_dir<double, n>( Ny, hy), 
                               dg::S1D<double, n>( hx),
                               dg::S1D<double, n>( hy),
                               dg::create::laplace1d_per<double, n>( Nx, hx)); 
    //laplace = A;
    //create derivatives
    dx = dgtensor<T,n>( tensor<T,n>(Ny, delta), create::dx_symm<value_type,n>( Nx, hx, -1));
    //dy = dy_;

}

template< class T, size_t n, class container, class MemorySpace>
void Toefl<T, n, container, MemorySpace>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //omega 
    blas1::axpby( 1., y[1], 0., omega);
    cudaThreadSynchronize();
    //compute S omega 
    blas2::symv( S2D<double, n>(hx, hy), omega, omega);
    /*
    ArrVec2d<T, n, thrust::host_vector<T> > view(rho, 10);
    std::cout << view <<std::endl;
    ArrVec2d<T, n, thrust::host_vector<T> > view2(phi, 10);
    std::cout << view2 <<std::endl;
    */
    cudaThreadSynchronize();
    std::cout << "Number of pcg iterations "<< pcg( laplace, phi, omega, T2D<double, n>(hx, hy), eps)<<std::endl;
    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi, yp[i]);

    // dx terms
    cudaThreadSynchronize();
    blas2::symv( dx, phi, dxphi);
    blas2::symv( dx, y[1], dxtheta);
    cudaThreadSynchronize();
    blas1::axpby( -1, dxphi, 1., yp[0]);
    blas1::axpby( -Pr*Ra, dxtheta, 1., yp[1]);

    //laplace terms
    blas2::symv( laplace, y[1], dxphi);
    blas2::symv( -Pr, dg::T2D<T,n>(hx, hy), dxphi, 1., yp[1]); 
    cudaThreadSynchronize();
    blas2::symv( laplace, y[0], dxphi);
    blas2::symv( -1., dg::T2D<T,n>(hx, hy), dxphi, 1., yp[0]); 


}

}//namespace dg

#endif //_DG_TOEFL_CUH
