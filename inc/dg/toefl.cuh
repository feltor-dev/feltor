#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"

#include "arakawa.cuh"
#include "derivatives.cuh"
#include "cg.cuh"

namespace dg
{

//Garcia equations with switched x <-> y  and phi -> -phi
template< class T, size_t n, class container=thrust::device_vector<T> >
struct Toefl
{
    typedef std::vector<container> Vector;
    typedef T value_type;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Toefl( const Grid<T,n>& ,  double R, double P, double eps);

    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    Matrix laplaceM;
    container omega, phi, phi_old, dxtheta, dxphi;
    ArakawaX<T, n, container> arakawaX; 
    CG<container > pcg;
    V2D<T,n> v2d;
    W2D<T,n> w2d;

    double Ra, Pr;
    double eps; 
};

template< class T, size_t n, class container>
Toefl<T, n, container>::Toefl( const Grid<T,n>& grid, double R, double P, double eps): 
    omega( grid.size(), 0.), phi(omega), phi_old( phi), dxtheta(omega), dxphi(omega), 
    arakawaX( grid), 
    pcg( omega, grid.size()),
    v2d( grid), w2d( grid), Ra (R), Pr(P), eps(eps)
{
    laplaceM = dg::create::laplacianM( grid, not_normed, XSPACE);
}

template< class T, size_t n, class container>
void Toefl<T, n, container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //omega 
    blas1::axpby( 1., y[1], 0., omega);
    cudaThreadSynchronize();
    //compute S omega 
    blas2::symv( w2d, omega, omega);
    blas1::axpby( 2., phi, -1.,  phi_old);
    phi.swap( phi_old);
    cudaThreadSynchronize();
    unsigned number = pcg( laplaceM, phi, omega, v2d, eps);
#ifdef DG_BENHMARK
    std::cout << "Number of pcg iterations "<<  number << "\n";
#endif

    for( unsigned i=0; i<y.size(); i++)
        arakawaX( y[i], phi, yp[i]);

    // dx terms
    cudaThreadSynchronize();
    blas2::symv( arakawaX.dx(), phi, dxphi);
    blas2::symv( arakawaX.dx(), y[0], dxtheta);
    cudaThreadSynchronize();
    blas1::axpby( 1, dxphi, 1., yp[0]);
    blas1::axpby( -Pr*Ra, dxtheta, 1., yp[1]);

    //laplace terms
    blas2::symv( laplaceM, y[0], dxphi);
    blas2::symv( -1., v2d, dxphi, 1., yp[0]); 
    cudaThreadSynchronize();
    blas2::symv( laplaceM, y[1], dxphi);
    blas2::symv( -Pr, v2d, dxphi, 1., yp[1]); 


}

}//namespace dg

#endif //_DG_TOEFL_CUH
