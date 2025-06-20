#pragma once

#include <exception>

#include "dg/algorithm.h"

///@note This is an old copy of the toefl project and shouldn't be taken as a basis for a new project

namespace mima
{

template< class Geometry, class Matrix, class Container >
struct Mima
{
    using Vector =  std::vector<Container>;
    typedef typename Container::value_type value_type;

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
    Mima( const Geometry& g, double kappa, double alpha, double eps, double nu, bool global);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const Container& potential( ) const { return phi;}


    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( double t, const Container& y, Container& yp);


  private:
    double kappa, alpha;
    bool global;
    Container phi, dxphi, dyphi, omega, lambda, chi, nGinv;
    Container dxxphi, dxyphi;

    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, Container> m_laplaceM;
    dg::ArakawaX< Geometry, Matrix, Container> m_arakawa;
    dg::PCG<Container> m_pcg;
    dg::Extrapolation<Container> m_extra;
    double m_eps, m_nu;
    dg::Helmholtz<Geometry, Matrix, Container> m_helmholtz;
};

template< class G, class M, class Container>
Mima< G, M, Container>::Mima( const G& grid, double kappa, double alpha, double eps, double nu, bool global ):
    kappa( kappa), alpha(alpha), global(global),
    phi( grid.size(), 0.), dxphi( phi), dyphi( phi), omega(phi), lambda(phi),
    chi(phi), nGinv(dg::evaluate(dg::ExpProfX(1.0, 0.0,kappa),grid)),
    dxxphi( phi), dxyphi(phi),
    m_laplaceM( grid,  dg::centered),
    m_arakawa( grid),
    m_pcg( phi, grid.size()),
    m_extra( 2, phi),
    m_eps(eps), m_nu(nu),
    m_helmholtz( -1., {grid})
{
}

template<class G, class M, class Container>
void Mima< G, M, Container>::operator()( double t, const Container& y, Container& yp)
{
    m_extra.extrapolate( t, phi);
    m_pcg.solve(m_helmholtz, phi, y,
        m_helmholtz.precond(), m_helmholtz.weights(), m_eps );
    m_extra.update( t, phi);
    dg::blas1::axpby( 1., phi, -1., y, chi); //chi = lap \phi

    m_arakawa( phi, chi, yp);
    //compute derivatives
    dg::blas2::gemv( m_arakawa.dx(), phi, dxphi);
    dg::blas2::gemv( m_arakawa.dy(), phi, dyphi);
    dg::blas2::gemv( m_arakawa.dx(), dxphi, dxxphi);
    dg::blas2::gemv( m_arakawa.dy(), dxphi, dxyphi);
    //gradient terms
    dg::blas1::axpby( -1, dyphi, 1., yp);


    //full-F terms (NOB) correction terms
    if( global )
    {
        //0.5*(nabla phi)^2
        dg::blas1::pointwiseDot( 0.5, dxphi, dxphi, 0.5, dyphi, dyphi, 0., omega);
        dg::blas2::gemv( m_arakawa.dy(), omega, dyphi); //d_y 0.5*(nabla phi)^2
        dg::blas1::axpby( +kappa, dyphi, 1., yp);     //kappa* d_y 0.5*(nabla phi)^2

        m_arakawa( omega,phi, lambda);        // [0.5*(nabla phi)^2, phi]
        dg::blas1::axpby( -kappa, lambda, 1., yp);  // -kappa* [0.5*(nabla phi)^2, phi]

        m_arakawa( omega, dxphi, lambda);         // [0.5*(nabla phi)^2, d_x phi]
        dg::blas1::axpby( +kappa*kappa, lambda, 1., yp); //kappa^2* [0.5*(nabla phi)^2, d_x phi]

        dg::blas1::pointwiseDot(y,dxphi,omega);   //omega = (phi - lap phi) d_x phi
        dg::blas1::pointwiseDivide(omega,nGinv,omega);   //omega = e^(kappa*x)*(phi - lap phi)*d_x phi
        dg::blas1::axpby( -kappa*alpha, omega, 1., yp);  // -kappa*alpha*e^(kappa*x)*(phi - lap phi)*d_x phi
    }
    // add diffusion
    dg::blas2::gemv( -m_nu, m_laplaceM, y, 1., yp);
}


}//namespace mima

