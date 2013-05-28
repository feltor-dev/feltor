#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"

#include "cusp_eigen.h"
#include "arakawa.cuh"
#include "laplace.cuh"
#include "functions.h"
#include "tensor.cuh"
#include "operator_matrix.cuh"
#include "dx.cuh"
#include "cg.cuh"

namespace dg
{

template< class T, size_t n, class container=thrust::device_vector<T> >
struct Shu 
{
    typedef T value_type;
    typedef container Vector;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Shu( unsigned Nx, unsigned Ny, double hx, double hy, double D, double eps = 1e-6);

    Matrix& lap() { return laplace;}
    const container& potential( ) {return phi;}

    void operator()( const Vector& y, Vector& yp);
  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    Matrix laplace;
    container omega, phi, phi_old;
    Arakawa<T, n, container> arakawa; 
    CG<Matrix, container, dg::T2D<T, n> > pcg;
    SimplicialCholesky cholesky;
    thrust::host_vector<double> x,b;
    
    double hx, hy;
    double D;
    double eps; 
};

template< class T, size_t n, class container>
Shu<T, n, container>::Shu( unsigned Nx, unsigned Ny, double hx, double hy,
        double D, double eps): 
    omega( n*n*Nx*Ny, 0.), phi(omega), phi_old(phi),
    arakawa( Nx, Ny, hx, hy, -1, -1), 
    pcg( omega, n*n*Nx*Ny), x( n*n*Nx*Ny), b( n*n*Nx*Ny),
    hx( hx), hy(hy), D(D), eps(eps)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    HMatrix A = dg::dgtensor<double, n>
                             ( dg::create::laplace1d_per<double, n>( Ny, hy), 
                               dg::S1D<double, n>( hx),
                               dg::S1D<double, n>( hy),
                               dg::create::laplace1d_per<double, n>( Nx, hx)); 
    laplace = A;
    cholesky.compute( A);
}

template< class T, size_t n, class container>
void Shu<T, n, container>::operator()( const Vector& y, Vector& yp)
{
    dg::blas2::symv( laplace, y, yp);
    dg::blas2::symv( -D, dg::T2D<T,n>(hx, hy), yp, 0., yp); //laplace is unnormalized -laplace
    cudaThreadSynchronize();
    //compute S omega
    blas2::symv( S2D<double, n>(hx, hy), y, omega);
    cudaThreadSynchronize();
    //blas1::axpby( 2., phi, -1.,  phi_old);
    //thrust::swap( phi, phi_old);
    //unsigned number = pcg( laplace, phi, omega, T2D<double, n>(hx, hy), eps);
    //std::cout << "Number of pcg iterations "<< number<<"\n"; 
    b = omega; //copy data to host
    cholesky.solve( x.data(), b.data(), b.size()); //solve on host
    phi = x; //copy data back to device
    arakawa( y, phi, omega); //A(y,phi)-> omega
    cudaThreadSynchronize();
    blas1::axpby( 1., omega, 1., yp);
    cudaThreadSynchronize();

}

}//namespace dg

#endif //_DG_SHU_CUH
