#pragma once

#include <exception>

#include "dg/algorithm.h"

///@note This is an old copy of the toefl project and shouldn't be taken as a basis for a new project

namespace mima
{
template< class Matrix, class container>
struct Diffusion
{
    Diffusion( const dg::CartesianGrid2d& g, double nu): nu_(nu),
        w2d(dg::create::weights( g)), v2d( dg::create::inv_weights(g)), temp( g.size()), LaplacianM( g, dg::normed, dg::centered) {
        }
    void operator()( double t, const container& x, container& y)
    {
        dg::blas2::gemv( LaplacianM, x, temp);
        dg::blas2::gemv( LaplacianM, temp, y);
        dg::blas1::scal( y, -nu_);
    }
    const container& weights(){return w2d;}
    const container& inv_weights(){return v2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    container temp;
    dg::Elliptic<dg::CartesianGrid2d, Matrix, container> LaplacianM;
};


template< class Matrix, class container=thrust::device_vector<double> >
struct Mima
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    /**
     * @brief Construct a Mima solver object
     *
     * @param g The grid on which to operate
     * @param kappa The curvature
     * @param nu The artificial viscosity
     * @param tau The ion temperature
     * @param eps_pol stopping criterion for polarisation equation
     * @param eps_gamma stopping criterion for Gamma operator
     * @param global local or global computation
     */
    Mima( const dg::CartesianGrid2d& g, double kappa, double alpha, double eps, bool global);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const container& potential( ) const { return phi;}


    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( double t, const container& y, container& yp);


  private:
    double kappa, alpha;
    bool global;
    container phi, dxphi, dyphi, omega, lambda, chi, nGinv;
    container dxxphi, dxyphi;

    //matrices and solvers
    dg::Elliptic<dg::CartesianGrid2d, Matrix, container> laplaceM;
    dg::ArakawaX<dg::CartesianGrid2d, Matrix, container> arakawa; 
    const container w2d, v2d;
    dg::Invert<container> invert;
    dg::Helmholtz<dg::CartesianGrid2d, Matrix, container> helmholtz;



};

template< class M, class container>
Mima< M, container>::Mima( const dg::CartesianGrid2d& grid, double kappa, double alpha, double eps, bool global ):
    kappa( kappa), global(global),
    phi( grid.size(), 0.), dxphi( phi), dyphi( phi), omega(phi), lambda(phi), chi(phi),
    nGinv(dg::evaluate(dg::ExpProfX(1.0, 0.0,kappa),grid)),
    dxxphi( phi), dxyphi(phi),
    laplaceM( grid, dg::normed, dg::centered),
    arakawa( grid),
    w2d( dg::create::weights(grid)), v2d( dg::create::inv_weights(grid)),
    invert( phi, grid.size(), eps),
    helmholtz( grid, -1)
{
}

template<class M, class container>
void Mima< M, container>::operator()( double t, const container& y, container& yp)
{
    invert( helmholtz, phi, y);
    dg::blas1::axpby( 1., phi, -1., y, chi); //chi = lap \phi


    arakawa( phi, chi, yp);
    //compute derivatives
    dg::blas2::gemv( arakawa.dx(), phi, dxphi);
    dg::blas2::gemv( arakawa.dy(), phi, dyphi);
    dg::blas2::gemv( arakawa.dx(), dxphi, dxxphi);
    dg::blas2::gemv( arakawa.dy(), dxphi, dxyphi);
    //gradient terms
    dg::blas1::axpby( -1, dyphi, 1., yp);


    //full-F terms (NOB) correction terms
    if( global )
    {
        //0.5*(nabla phi)^2
        dg::blas1::pointwiseDot( 0.5, dxphi, dxphi, 0.5, dyphi, dyphi, 0., omega);
        dg::blas2::gemv( arakawa.dy(), omega, dyphi); //d_y 0.5*(nabla phi)^2
        dg::blas1::axpby( +kappa, dyphi, 1., yp);     //kappa* d_y 0.5*(nabla phi)^2

        arakawa( omega,phi, lambda);        // [0.5*(nabla phi)^2, phi]
        dg::blas1::axpby( -kappa, lambda, 1., yp);  // -kappa* [0.5*(nabla phi)^2, phi]

        arakawa( omega, dxphi, lambda);         // [0.5*(nabla phi)^2, d_x phi]
        dg::blas1::axpby( +kappa*kappa, lambda, 1., yp); //kappa^2* [0.5*(nabla phi)^2, d_x phi]

        dg::blas1::pointwiseDot(y,dxphi,omega);   //omega = (phi - lap phi) d_x phi
        dg::blas1::pointwiseDivide(omega,nGinv,omega);   //omega = e^(kappa*x)*(phi - lap phi)*d_x phi
        dg::blas1::axpby( -kappa*alpha, omega, 1., yp);  // -kappa*alpha*e^(kappa*x)*(phi - lap phi)*d_x phi
    }
}


}//namespace mima

