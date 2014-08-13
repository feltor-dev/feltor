#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH

#include "algorithm.h"
#include "blas.h"
#include "arakawa.h"
#include "cg.h"

namespace dg
{

//Garcia equations with switched x <-> y  and phi -> -phi
template<class Matrix, class container, class Preconditioner >
struct Toefl
{
    template<class Grid>
    Toefl( const Grid& ,  double R, double P, double eps);

    void operator()( std::vector<container>& y, std::vector<container>& yp);
  private:
    Matrix laplaceM;
    container omega, phi, phi_old, dxtheta, dxphi;
    ArakawaX<Matrix, container> arakawaX; 
    Invert<container > pcg;
    Preconditioner w2d, v2d;

    double Ra, Pr;
};

template< class Matrix, class container, class Prec>
template< class Grid>
Toefl<Matrix, container, Prec>::Toefl( const Grid& grid, double R, double P, double eps): 
    laplaceM( dg::create::laplacianM( grid, not_normed, dg::symmetric)),
    omega( dg::evaluate(one, grid) ), phi(omega), phi_old( phi), 
    dxtheta(omega), dxphi(omega), 
    arakawaX( grid), 
    pcg( omega, grid.size(), eps),
    v2d( dg::create::precond(grid)), w2d( dg::create::weights(grid)), Ra (R), Pr(P)
{
}

template< class Matrix, class container, class P>
void Toefl< Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //omega 
    blas1::axpby( 1., y[1], 0., omega);
    unsigned number = pcg( laplaceM, phi, omega, w2d, v2d);
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
