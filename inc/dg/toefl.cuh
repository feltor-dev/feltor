#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.cuh"

#include "arakawa.cuh"
#include "derivatives.cuh"
#include "cg.cuh"

namespace dg
{

//Garcia equations with switched x <-> y  and phi -> -phi
template< class container=thrust::device_vector<double> >
struct Toefl
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Toefl( const Grid<value_type>& ,  double R, double P, double eps);

    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    Matrix laplaceM;
    container omega, phi, phi_old, dxtheta, dxphi;
    ArakawaX<container> arakawaX; 
    CG<container > pcg;
    container w2d, v2d;

    double Ra, Pr;
    double eps; 
};

template< class container>
Toefl<container>::Toefl( const Grid<value_type>& grid, double R, double P, double eps): 
    omega( grid.size(), 0.), phi(omega), phi_old( phi), dxtheta(omega), dxphi(omega), 
    arakawaX( grid), 
    pcg( omega, grid.size()),
    v2d( create::v2d(grid)), w2d( create::w2d(grid)), Ra (R), Pr(P), eps(eps)
{
    laplaceM = dg::create::laplacianM( grid, not_normed, XSPACE);
}

template< class container>
void Toefl< container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //omega 
    blas1::axpby( 1., y[1], 0., omega);
    //compute S omega 
    blas2::symv( w2d, omega, omega);
    blas1::axpby( 2., phi, -1.,  phi_old);
    phi.swap( phi_old);
    unsigned number = pcg( laplaceM, phi, omega, v2d, eps);
    number +=0; //avoid warning
#ifdef DG_BENHMARK
    std::cout << "Number of pcg iterations "<<  number << "\n";
#endif

    for( unsigned i=0; i<y.size(); i++)
        arakawaX( y[i], phi, yp[i]);

    // dx terms
    blas2::symv( arakawaX.dx(), phi, dxphi);
    blas2::symv( arakawaX.dx(), y[0], dxtheta);
    blas1::axpby( 1, dxphi, 1., yp[0]);
    blas1::axpby( -Pr*Ra, dxtheta, 1., yp[1]);

    //laplace terms
    blas2::symv( laplaceM, y[0], dxphi);
    blas2::symv( -1., v2d, dxphi, 1., yp[0]); 
    blas2::symv( laplaceM, y[1], dxphi);
    blas2::symv( -Pr, v2d, dxphi, 1., yp[1]); 


}

}//namespace dg

#endif //_DG_TOEFL_CUH
