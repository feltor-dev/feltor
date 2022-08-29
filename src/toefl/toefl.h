#pragma once
#include <exception>

#include "dg/algorithm.h"
#include "parameters.h"

namespace toefl
{

template< class Geometry,  class Matrix, class Container >
struct Explicit
{
    Explicit( const Geometry& g, const Parameters& p );

    const Container& phi(unsigned i ) const { return m_phi[i];}
    const Container& var(unsigned i ) const { return m_ype[i];}
    const Container& uE2() const { return m_uE2;}

    dg::Elliptic<Geometry, Matrix, Container>& laplacianM( ) {
        return m_laplaceM;
    }

    dg::Helmholtz<Geometry, Matrix, Container >&  gamma_inv() {
        return m_multi_gamma1[0];
    }
    unsigned ncalls() const{ return m_ncalls;}

    void operator()( double t, const std::array<Container,2>& y,
            std::array<Container,2>& yp);

    void compute_psi( double t);
    void polarisation( double t, const std::array<Container,2>& y);
  private:
    //use chi and omega as helpers to compute square velocity in uE2
    Container m_chi, m_omega, m_uE2;
    const Container m_binv; //magnetic field

    std::array<Container,2> m_phi, m_dxphi, m_dyphi, m_ype;
    std::array<Container,2> m_lapy, m_v;
    Container m_gamma_n;

    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, Container> m_laplaceM;
    std::vector<dg::Elliptic<Geometry, Matrix, Container> > m_multi_pol;
    std::vector<dg::Helmholtz<Geometry,  Matrix, Container> > m_multi_gamma1;
    std::array<Matrix,2> m_centered;
    dg::Advection< Geometry, Matrix, Container> m_adv;
    dg::ArakawaX< Geometry, Matrix, Container> m_arakawa;

    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    dg::Extrapolation<Container> m_old_phi, m_old_psi, m_old_gammaN;
    std::vector<Container> m_multi_chi;

    Parameters m_p;

    unsigned m_ncalls = 0;

};

template< class Geometry, class M, class Container>
Explicit< Geometry, M, Container>::Explicit( const Geometry& grid, const Parameters& p ):
    m_chi( evaluate( dg::zero, grid)), m_omega(m_chi), m_uE2(m_chi),
    m_binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)),
    m_phi( {m_chi, m_chi}), m_dxphi(m_phi), m_dyphi( m_phi), m_ype(m_phi),
    m_lapy(m_phi), m_v(m_phi),
    m_gamma_n(m_chi),
    m_laplaceM( grid,  p.diff_dir),
    m_adv( grid), m_arakawa(grid),
    m_multigrid( grid, p.num_stages),
    m_old_phi( 2, m_chi), m_old_psi( 2, m_chi), m_old_gammaN( 2, m_chi),
    m_p(p)
{
    m_multi_chi= m_multigrid.project( m_chi);
    for( unsigned u=0; u<p.num_stages; u++)
    {
        m_multi_pol.push_back({ m_multigrid.grid(u),  p.pol_dir, 1.});
        m_multi_gamma1.push_back({-0.5*p.tau, { m_multigrid.grid(u), p.pol_dir}});
    }
    m_centered = {dg::create::dx( grid, m_p.bcx),
                  dg::create::dy( grid, m_p.bcy)};
}

template< class G, class M, class Container>
void Explicit<G, M, Container>::compute_psi( double t)
{
    if(m_p.model == "gravity_local")
        return;
    //in gyrofluid invert Gamma operator
    if( m_p.model == "local" || m_p.model == "global")
    {
        if (m_p.tau == 0.) {
            dg::blas1::axpby( 1.,m_phi[0], 0.,m_phi[1]); //chi = N_i - 1
        }
        else {
            m_old_psi.extrapolate( t, m_phi[1]);
            m_multigrid.set_benchmark( true, "Gamma Phi   ");
            m_multigrid.solve( m_multi_gamma1, m_phi[1], m_phi[0], m_p.eps_gamma);
            m_old_psi.update( t, m_phi[1]);
        }
    }
    //compute (nabla phi)^2
    m_multi_pol[0].variation(m_phi[0], m_uE2);
    //compute psi
    if(m_p.model == "global")
    {
        dg::blas1::pointwiseDot( 1., m_binv, m_binv, m_uE2, 0., m_uE2);
        dg::blas1::axpby( -0.5, m_uE2, 1., m_phi[1]);
    }
    else if ( m_p.model == "drift_global")
    {
        dg::blas1::pointwiseDot( 1., m_binv, m_binv, m_uE2, 0., m_uE2);
        dg::blas1::axpby( 0.5, m_uE2, 0., m_phi[1]);
    }
    else if( m_p.model == "gravity_global" )
        dg::blas1::axpby( 0.5, m_omega, 0., m_phi[1]);
}


//computes and modifies expy!!
template<class G, class M, class Container>
void Explicit<G, M, Container>::polarisation( double t,
        const std::array<Container,2>& y)
{
    //compute chi
    if(m_p.model == "global" )
    {
        dg::blas1::evaluate( m_chi, dg::equals(), []DG_DEVICE
                ( double nt, double binv){
                    return (nt+1.)*binv*binv;
                }, y[1], m_binv);
        if( !m_p.boussinesq)
        {
            m_multigrid.project( m_chi, m_multi_chi);
            for( unsigned u=0; u<3; u++)
                m_multi_pol[u].set_chi( m_multi_chi[u]);
        }
    }
    else if(m_p.model == "gravity_global" )
    {
        dg::blas1::evaluate( m_chi, dg::equals(), []DG_DEVICE
                ( double nt){ return (nt+1.); }, y[1]);
        if( !m_p.boussinesq)
        {
            m_multigrid.project( m_chi, m_multi_chi);
            for( unsigned u=0; u<3; u++)
                m_multi_pol[u].set_chi( m_multi_chi[u]);
        }
    }
    else if( m_p.model == "drift_global" )
    {
        dg::blas1::evaluate( m_chi, dg::equals(), []DG_DEVICE
                ( double nt, double binv){
                    return (nt+1.)*binv*binv;
                }, y[0], m_binv);
        if( !m_p.boussinesq)
        {
            m_multigrid.project( m_chi, m_multi_chi);
            for( unsigned u=0; u<3; u++)
                m_multi_pol[u].set_chi( m_multi_chi[u]);
        }
    }
    //compute polarisation
    if( m_p.model == "local" || m_p.model == "global")
    {
        if (m_p.tau == 0.) {
            dg::blas1::axpby( 1., y[1], 0., m_gamma_n); //chi = N_i - 1
        }
        else {
            m_old_gammaN.extrapolate(t, m_gamma_n);
            m_multigrid.set_benchmark( true, "Gamma N     ");
            m_multigrid.solve( m_multi_gamma1, m_gamma_n, y[1], m_p.eps_gamma);
            m_old_gammaN.update(t, m_gamma_n);
        }
        dg::blas1::axpby( -1., y[0], 1., m_gamma_n, m_omega); //omega = a_i\Gamma n_i - n_e
    }
    else
        dg::blas1::axpby( -1., y[1], 0., m_omega);
    if( m_p.model == "global" || m_p.model == "gravity_global"
            || m_p.model == "drift_global")
        if( m_p.boussinesq)
            dg::blas1::pointwiseDivide( m_omega, m_chi, m_omega);
    //invert

    m_old_phi.extrapolate(t, m_phi[0]);
    m_multigrid.set_benchmark( true, "Polarisation");
    m_multigrid.solve( m_multi_pol, m_phi[0], m_omega, m_p.eps_pol);
    m_old_phi.update( t, m_phi[0]);
}

template< class G, class M, class Container>
void Explicit<G, M, Container>::operator()( double t,
        const std::array<Container,2>& y, std::array<Container,2>& yp)
{
    m_ncalls ++ ;
    //y[0] = N_e - 1
    //y[1] = N_i - 1 || y[1] = Omega

    polarisation( t, y);
    compute_psi( t);

    ///////////////////////////////////////////////////////////////////////
    if( m_p.model == "gravity_global")
    {
        dg::blas1::transform( y[0], m_ype[0], dg::PLUS<double>(1.));
        dg::blas1::copy( y[1], m_ype[1]);

        // ExB advection with updwind scheme
        dg::blas2::symv( m_centered[0], m_phi[0], m_dxphi[0]);
        dg::blas2::symv( m_centered[1], m_phi[0], m_dyphi[0]);
        dg::blas1::pointwiseDot( -1., m_binv, m_dyphi[0], 0., m_v[0]);
        dg::blas1::pointwiseDot( +1., m_binv, m_dxphi[0], 0., m_v[1]);
        for( unsigned u=0; u<2; u++)
        {
            m_adv.upwind( -1., m_v[0], m_v[1], y[u], 0., yp[u]);
        }
        m_arakawa(1., y[0], m_phi[1], 1., yp[1]);
        // diamagnetic compression
        dg::blas2::gemv( -1., m_centered[1], y[0], 1., yp[1]);
        // friction
        dg::blas1::axpby( -m_p.friction, y[1], 1., yp[1]);

    }
    else if( m_p.model == "gravity_local")
    {
        dg::blas1::copy( y, m_ype);
        for( unsigned u=0; u<2; u++)
        {
            // ExB advection with updwind scheme
            dg::blas2::symv(  1., m_centered[0], m_phi[u], 0., m_v[1]);
            dg::blas2::symv( -1., m_centered[1], m_phi[u], 0., m_v[0]);
            m_adv.upwind( -1., m_v[0], m_v[1], y[u], 0., yp[u]);
        }
        // diamagnetic compression
        dg::blas2::gemv( -1., m_centered[1], y[0], 1., yp[1]);
        // friction
        dg::blas1::axpby( -m_p.friction, y[1], 1., yp[1]);
    }
    else if( m_p.model == "drift_global")
    {
        dg::blas1::transform( y[0], m_ype[0], dg::PLUS<double>(1.));
        dg::blas1::copy( y[1], m_ype[1]);
        for( unsigned u=0; u<2; u++)
        {
            // ExB + Curv advection with updwind scheme
            dg::blas2::symv( m_centered[0], m_phi[u], m_dxphi[u]);
            dg::blas2::symv( m_centered[1], m_phi[u], m_dyphi[u]);
            // only phi is advecting not psi
            dg::blas1::pointwiseDot( -1., m_binv, m_dyphi[0], 0., m_v[0]);
            dg::blas1::pointwiseDot( +1., m_binv, m_dxphi[0], 0., m_v[1]);
            m_adv.upwind( -1., m_v[0], m_v[1], y[u], 0., yp[u]);
            // Div ExB velocity
            dg::blas1::pointwiseDot( m_p.kappa, m_ype[u], m_dyphi[u], 1., yp[u]);
        }
        m_arakawa(y[0], m_phi[1], m_omega);
        dg::blas1::pointwiseDot( 1., m_binv, m_omega, 1., yp[1]);

        dg::blas1::pointwiseDot( m_p.kappa, y[1], m_dyphi[0], 1.,  yp[1]);
        // diamagnetic compression
        dg::blas2::gemv( -m_p.kappa, m_centered[1], y[0], 1., yp[1]);
    }
    else if ( m_p.model == "global")
    {
        dg::blas1::transform( y, m_ype, dg::PLUS<double>(1.));
        std::array<double, 2> tau = {-1., m_p.tau};
        for( unsigned u=0; u<2; u++)
        {
            // ExB + Curv advection with updwind scheme
            dg::blas2::symv( m_centered[0], m_phi[u], m_dxphi[u]);
            dg::blas2::symv( m_centered[1], m_phi[u], m_dyphi[u]);
            dg::blas1::pointwiseDot( -1., m_binv, m_dyphi[u], 0., m_v[0]);
            dg::blas1::pointwiseDot( +1., m_binv, m_dxphi[u], 0., m_v[1]);
            dg::blas1::plus( m_v[1], -tau[u]*m_p.kappa);
            m_adv.upwind( -1., m_v[0], m_v[1], y[u], 0., yp[u]);
            // Div ExB velocity
            dg::blas1::pointwiseDot( m_p.kappa, m_ype[u], m_dyphi[u], 1., yp[u]);
        }
    }
    else if ( m_p.model == "local")
    {
        dg::blas1::copy( y, m_ype);
        std::array<double, 2> tau = {-1., m_p.tau};
        for( unsigned u=0; u<2; u++)
        {
            // ExB + Curv advection with updwind scheme
            dg::blas2::symv( m_centered[0], m_phi[u], m_dxphi[u]);
            dg::blas2::symv( m_centered[1], m_phi[u], m_dyphi[u]);
            dg::blas1::axpby( -1., m_dyphi[u], 0., m_v[0]);
            dg::blas1::axpby( +1., m_dxphi[u], 0., m_v[1]);
            dg::blas1::plus( m_v[1], -tau[u]*m_p.kappa);
            m_adv.upwind( -1., m_v[0], m_v[1], y[u], 0., yp[u]);
            // Div ExB velocity
            dg::blas1::axpby( m_p.kappa, m_dyphi[u], 1., yp[u]);
        }
    }

    for( unsigned u=0; u<2; u++)
    {
        dg::blas2::symv( -1., m_laplaceM, y[u], 0., m_lapy[u]);
        dg::blas1::axpby( m_p.nu, m_lapy[u], 1., yp[u]);
    }
    return;
}

}//namespace dg
