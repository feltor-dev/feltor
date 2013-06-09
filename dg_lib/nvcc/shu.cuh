#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"

#include "cusp_eigen.h"
#include "arakawa.cuh"
#include "derivatives.cuh"
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

    Shu( const Grid<T,n>& grid, double D, double eps = 1e-4);

    Matrix& lap() { return laplace;}
    const container& potential( ) {return phi;}
    void operator()( const Vector& y, Vector& yp);
  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    Matrix laplace;
    container omega, phi, phi_old;
    Arakawa<T, n, container> arakawa; 
    CG< container > pcg;
    SimplicialCholesky cholesky;
    thrust::host_vector<double> x,b;
    T2D<T,n> t2d;
    S2D<T,n> s2d;
    
    double hx, hy;
    double D;
    double eps; 
};

template< class T, size_t n, class container>
Shu<T, n, container>::Shu( const Grid<T, n>& g, double D, double eps): 
    omega( n*n*g.Nx()*g.Ny(), 0.), phi(omega), phi_old(phi),
    arakawa( g), 
    pcg( omega, g.size()), x( phi), b( x),
    t2d( g.hx(), g.hy()), s2d( g.hx(), g.hy()), D(D), eps(eps)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    HMatrix A = dg::create::laplacianM( g, not_normed, LSPACE);
    laplace = A;
    cholesky.compute( A);
}

template< class T, size_t n, class container>
void Shu<T, n, container>::operator()( const Vector& y, Vector& yp)
{
    dg::blas2::symv( laplace, y, yp);
    dg::blas2::symv( -D, t2d, yp, 0., yp); //laplace is unnormalized -laplace
    cudaThreadSynchronize();
    //compute S omega
    blas2::symv( s2d, y, omega);
    cudaThreadSynchronize();
    blas1::axpby( 2., phi, -1.,  phi_old);
    phi.swap( phi_old);
    unsigned number = pcg( laplace, phi, omega, t2d, eps);
    //std::cout << "Number of pcg iterations "<< number<<"\n"; 
    //b = omega; //copy data to host
    //cholesky.solve( x.data(), b.data(), b.size()); //solve on host
    //phi = x; //copy data back to device
    arakawa( y, phi, omega); //A(y,phi)-> omega
    cudaThreadSynchronize();
    blas1::axpby( 1., omega, 1., yp);
    cudaThreadSynchronize();

}

}//namespace dg

#endif //_DG_SHU_CUH
