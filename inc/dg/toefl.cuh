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
    Elliptic<Matrix, container, Preconditioner> laplaceM;
    container omega, phi, phi_old, dxtheta, dxphi;
    ArakawaX<Matrix, container> arakawaX; 
    Invert<container > invert;

    double Ra, Pr;
};

template< class Matrix, class container, class Prec>
template< class Grid>
Toefl<Matrix, container, Prec>::Toefl( const Grid& grid, double R, double P, double eps): 
    laplaceM( grid, not_normed, dg::centered),
    omega( dg::evaluate(one, grid) ), phi(omega), phi_old( phi), 
    dxtheta(omega), dxphi(omega), 
    arakawaX( grid), 
    invert( omega, grid.size(), eps), Ra(R), Pr(P)
{ }

template< class Matrix, class container, class P>
void Toefl< Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //omega 
    blas1::axpby( 1., y[1], 0., omega);
    unsigned number = invert( laplaceM, phi, omega);
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
    blas2::symv( -1., laplaceM.precond(), dxphi, 1., yp[0]); 
    blas2::symv( laplaceM, y[1], dxphi);
    blas2::symv( -Pr, laplaceM.precond(), dxphi, 1., yp[1]); 


}

}//namespace dg

#endif //_DG_TOEFL_CUH
