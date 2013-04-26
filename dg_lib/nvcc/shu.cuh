#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"

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
struct Shu 
{
    typedef T value_type;
    typedef container Vector;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Shu( unsigned Nx, unsigned Ny, double hx, double hy, double D, double eps = 1e-6);

    Matrix& lap() { return laplace;}
    void operator()( const Vector& y, Vector& yp);
  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    Matrix laplace;
    container rho, phi;
    Arakawa<T, n, container, MemorySpace> arakawa; 
    CG<Matrix, container, dg::T2D<T, n> > pcg;
    
    double hx, hy;
    double D;
    double eps; 
};

template< class T, size_t n, class container, class MemorySpace>
Shu<T, n, container, MemorySpace>::Shu( unsigned Nx, unsigned Ny, double hx, double hy,
        double D, double eps): 
    rho( n*n*Nx*Ny, 0.), phi(rho),
    arakawa( Nx, Ny, hx, hy, 0, 0), 
    pcg( rho, n*n*Nx*Ny),
    hx( hx), hy(hy), D(D), eps(eps)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    HMatrix A = dg::dgtensor<double, n>
                             ( dg::create::laplace1d_dir<double, n>( Ny, hy), 
                               dg::S1D<double, n>( hx),
                               dg::S1D<double, n>( hy),
                               dg::create::laplace1d_dir<double, n>( Nx, hx)); 
    laplace = A;
}

template< class T, size_t n, class container, class MemorySpace>
void Shu<T, n, container, MemorySpace>::operator()( const Vector& y, Vector& yp)
{
    dg::blas2::symv( laplace, y, yp);
    //laplace is unnormalized -laplace
    dg::blas2::symv( -D, dg::T2D<T,n>(hx, hy), yp, 0., yp); 

    //rho = y;
    cudaThreadSynchronize();
    blas1::axpby( 1., y, 0, rho);
    cudaThreadSynchronize();
    //compute S rho
    blas2::symv( S2D<double, n>(hx, hy), rho, rho);
    cudaThreadSynchronize();
    blas1::axpby( 0., phi, 0., phi);
    unsigned number = pcg( laplace, phi, rho, T2D<double, n>(hx, hy), eps);
    std::cout << "Number of pcg iterations "<< number<<"\n"; 
    for( unsigned i=0; i<y.size(); i++)
        arakawa( y, phi, rho); //A(y,phi)-> rho
    cudaThreadSynchronize();
    blas1::axpby( 1., rho, 1., yp);
    cudaThreadSynchronize();

}

}//namespace dg

#endif //_DG_SHU_CUH
