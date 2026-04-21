#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/matrix/matrix.h"
#include "dg/geometries/geometries.h"
#include "../feltor/common.h"


namespace thermal
{

// Class to hold solvers for phi and Gamma
// Assume Product Space Grid where varphi component decouples from perp grids

template< class Geometry, class Matrix, class Container >
struct Solvers
{
    Solvers() = default;
    Solvers( const Geometry&, thermal::Parameters p,
        dg::geo::TokamakMagneticField mag, dg::file::WrappedJsonValue);
    void compute_phi(
        double time, const std::vector<Container>& density,
        const std::vector<Container>& pperp,
        Container& phi,
        bool penalize_wall, const Container& wall,
        bool penalize_sheath, const Container& sheath
    );

    void compute_aparST( double time, const std::vector<Container>& densityST,
        const std::vector<Container>& wST, Container& aparST,
        // update determines if old solution is updated (or start iteration from 0)
        bool update);

    // Only valid directly after compute_phi, because it stores tridiag from phi
    void compute_psi(
        double time,
        const Container& phi,
        std::vector<Container>& psi0,
        std::vector<Container>& psi1
    );

    const Geometry& grid() const {
        return m_multigrid.grid(0);
    }
    const Container& uE2() const {return m_uE2;}
    //volume with dG weights
    const Container& vol3d() const { return m_laplaceM.weights();}
    const Container& weights() const { return m_laplaceM.weights();}

    // 1/B_varphi
    const Container& Btorinv() const {return m_Btorinv;}
    // 1/R
    const Container& rinv() const {return m_Rinv;}
    void compute_lapMperp( const Container& f, Container& lapMf) const {
        dg::blas2::symv( m_laplaceM, f, lapMf);
    }
    void compute_lapMperpP( const Container& phi, Container& lapMphi) const {
        dg::blas2::symv( m_laplaceMphi, phi, lapMphi);
    }

    private:
    thermal::Parameters m_p;
    Container m_temp0, m_temp1, m_uE2, m_omega, m_Btorinv, m_Rinv;
    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    std::vector<Container> m_multi_chi;

    std::vector<dg::Elliptic2d< Geometry, Matrix, Container> > m_multi_pol;
    std::vector<dg::Helmholtz2d<Geometry, Matrix, Container> >
        m_multi_ampere;

    dg::Elliptic2d<Geometry, Matrix, Container> m_laplaceM, m_laplaceMphi;

    dg::Extrapolation<Container> m_old_phi, m_old_aparST;
};

template<class Geometry, class Matrix, class Container>
Solvers<Geometry, Matrix, Container>::Solvers( const Geometry& g,
    thermal::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue
    ): m_p(p),
    m_multigrid( g, p.stages),
    m_old_phi( 2, dg::evaluate( dg::zero, g)), m_old_aparST( m_old_phi)
{
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_uE2 = m_omega = m_temp1 = m_temp0;
    dg::assign(  dg::pullback(dg::geo::InvBtor(mag), g), m_Btorinv);
    dg::assign(  dg::pullback(dg::cooX3d, g), m_Rinv);
    dg::blas1::pointwiseDivide( 1., m_Rinv, m_Rinv);

    //Set a hard code limit on the maximum number of iteration to avoid
    //endless iteration in case of failure
    m_multigrid.set_max_iter( 1e5);
    /////////////////////////init elliptic and helmholtz operators/////////
    m_multi_chi = m_multigrid.project( m_temp0);
    m_multi_pol.resize(p.stages);
    m_multi_ampere.resize(p.stages);
    for( unsigned u=0; u<p.stages; u++)
    {
        m_multi_pol[u].construct( m_multigrid.grid(u),
            p.bcxP, p.bcyP,
            p.pol_dir, p.jfactor);
        m_multi_ampere[u] = {  -1.,
                {m_multigrid.grid(u), p.bcxA, p.bcyA, p.pol_dir}};
        Container Rinv = dg::pullback( dg::cooX3d, m_multigrid.grid(u));
        dg::blas1::pointwiseDivide( 1., Rinv, Rinv);
        dg::blas1::pointwiseDot( Rinv, Rinv, Rinv); // = 1/R^2
        m_multi_ampere[u].matrix().set_chi( Rinv);
    }
    m_laplaceM.construct( g, p.bcx, p.bcy, p.pol_dir, p.jfactor);
    m_laplaceMphi.construct( g, p.bcxP, p.bcyP, p.pol_dir, p.jfactor);
}

template<class Geometry, class Matrix, class Container>
void Solvers<Geometry, Matrix, Container>::compute_phi(
    double time, const std::vector<Container>& density,
    const std::vector<Container>& pperp,
    Container& phi,
    bool penalize_wall, const Container& wall,
    bool penalize_sheath, const Container& sheath
    )
{
    //----------Compute and set chi----------------------------//
    dg::blas1::copy( 0., m_temp0);
    // The first species is the electron species where mass is neglected
    for( unsigned s = 1; s<m_p.num_species; s++)
    {
        dg::blas1::pointwiseDot( m_p.mu[s], density[s], m_Btorinv, m_Btorinv, 1., m_temp0);
    }
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_pol[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    // Electrons
    dg::blas1::axpby( m_p.z[0], density[0], 0., m_temp0);
    for( unsigned s = 1; s<m_p.num_species; s++)
    {
        dg::blas1::axpby( m_p.z[s], density[s], 1., m_temp0);
        dg::blas1::pointwiseDot(m_p.mu[s]/2./m_p.z[s]/m_p.z[s], pperp[s], m_Btorinv, m_Btorinv, 0., m_omega);
        dg::blas2::symv( -m_p.z[s], m_laplaceM, m_omega, 1., m_temp0);
    }
    // Add penalization method
    common::multiply_rhs_penalization( m_temp0, penalize_wall, wall, penalize_sheath, sheath );

    //----------Invert polarisation----------------------------//
    m_old_phi.extrapolate( time, phi);
    m_multigrid.set_benchmark( true, "Polarisation");
    std::vector<unsigned> number = m_multigrid.solve(
        m_multi_pol, phi, m_temp0, m_p.eps_pol);
    m_old_phi.update( time, phi);

}

template<class Geometry, class Matrix, class Container>
void Solvers<Geometry, Matrix, Container>::compute_aparST(
    double time, const std::vector<Container>& densityST,
    const std::vector<Container>& wST, Container& aparST,
    // update determines if old solution is updated (or start iteration from 0)
    bool update)
{
    // beta is nonzero when this function is called
    //----------Compute and set chi----------------------------//
    dg::blas1::copy( 0, m_temp0);
    for( unsigned s=0; s<m_p.num_species; s++)
    {
        dg::blas1::axpby(  m_p.beta*m_p.z[s]*m_p.z[s]/m_p.mu[s],
            densityST[s], 1., m_temp0);
    }
    dg::blas1::pointwiseDot( 1., m_temp0, m_Rinv, m_Rinv, 0., m_temp0);
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_ampere[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    dg::blas1::copy( 0, m_temp0);
    for( unsigned s=0; s<m_p.num_species; s++)
    {
        dg::blas1::pointwiseDot(  m_p.beta*m_p.z[s], densityST[s], wST[s],
                                  1., m_temp0);
    }
    dg::blas1::pointwiseDot( m_temp0, m_Rinv, m_temp0);
    //----------Invert Induction Eq----------------------------//
    if( update)
        m_old_aparST.extrapolate( time, aparST);
    m_multigrid.set_benchmark( true, "Apar        ");
    std::vector<unsigned> number = m_multigrid.solve(
        m_multi_ampere, aparST, m_temp0, m_p.eps_ampere);
    if( update)
        m_old_aparST.update( time, aparST);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_ampere);
}

template<class Geometry, class Matrix, class Container>
void Solvers<Geometry, Matrix, Container>::compute_psi(
    double,
    const Container& phi,
    std::vector<Container>& psi0,
    std::vector<Container>& psi1
    )
{
    // s == 0
    dg::blas1::copy( phi, psi0[0]);
    dg::blas1::copy( 0,   psi1[0]);
    // u_E^2
    m_laplaceMphi.variation( m_Btorinv, phi, m_uE2);
    for( unsigned s = 1; s<m_p.num_species; s++)
    {
        dg::blas1::axpby( 1., phi, -0.5*m_p.mu[s]/m_p.z[s], m_uE2, psi0[s]);
        dg::blas2::symv( m_laplaceMphi, phi, m_temp0);
        dg::blas1::pointwiseDot( -1., m_temp0, m_Btorinv, 0., psi1[s]);
    }
}

}//namespace thermal
