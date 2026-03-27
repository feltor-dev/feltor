#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/geometries/geometries.h"
#include "common.h"

#ifdef WRITE_POL_FILE
int counter = 0;
dg::file::NcFile pol_file;
#endif // WRITE_POL_FILE

namespace feltor
{

// Class to hold solvers for phi and Gamma
// Assume Product Space Grid where varphi component decouples from perp grids

template< class Geometry, class Matrix, class Container >
struct Solvers
{
    Solvers() = default;
    Solvers( const Geometry&, feltor::Parameters p,
        dg::geo::TokamakMagneticField mag, dg::file::WrappedJsonValue);
    void compute_phi(
        double time, const std::array<Container,2>& density,
        Container& phi,
        bool penalize_wall, const Container& wall,
        bool penalize_sheath, const Container& sheath
    );

    void compute_aparST( double time, const std::array<Container,2>& densityST,
        std::array<Container,2>& velocityST, Container& aparST,
        // update determines if old solution is updated (or start iteration from 0)
        bool update);

    // Only valid directly after compute_phi, because it stores tridiag from phi
    void compute_psi(
        double time,
        const Container& phi,
        Container& psi
    );

    const Geometry& grid() const {
        return m_multigrid.grid(0);
    }
    const Container& uE2() const {return m_uE2;}
    //volume with dG weights
    //const Container& vol3d() const { return m_laplaceM.weights();}
    //const Container& weights() const { return m_laplaceM.weights();}
    //
    const Container&  old_gammaN_head() const{
        return m_old_gammaN.head();
    }
    const Container&  old_psi_head() const{
        return m_old_psi.head();
    }

    // 1/B_varphi
    const Container& binv( ) const { return m_binv; }
    // 1/R
    const Container& rinv() const {return m_Rinv;}
    //void compute_lapMperp( const Container& f, Container& lapMf) const {
    //    dg::blas2::symv( m_laplaceM, f, lapMf);
    //}
    //void compute_lapMperpP( const Container& phi, Container& lapMphi) const {
    //    dg::blas2::symv( m_laplaceMphi, phi, lapMphi);
    //}

    void invert_gammaN( const Container& src, Container& target)
    {
        // ne-nbc = Gamma (ni-nbc)
        dg::blas1::transform(src, m_temp0, dg::PLUS<double>(-m_p.nbc));
        dg::blas1::copy( m_temp0, target);
        m_multigrid.set_benchmark( true, "Gamma N     ");
        std::vector<unsigned> number = m_multigrid.solve(
            m_multi_invgammaN, target, m_temp0, m_p.eps_gamma);
        dg::blas1::plus(target, +m_p.nbc);
    }
    void compute_pol( double alpha, const Container& density, const Container& phi, Container& temp, double beta, Container& result)
    {
        // polarisation term
        dg::blas1::pointwiseDot( m_p.mu[1], density, m_binv, m_binv, 0., temp);
        m_multi_pol[0].set_chi( temp);
        dg::blas2::symv( -alpha, m_multi_pol[0], phi, beta, result);
    }

    private:
    feltor::Parameters m_p;
    Container m_temp0, m_temp1, m_uE2, m_binv, m_Rinv;
    Matrix m_dx_P, m_dy_P, m_dz;
    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    std::vector<Container> m_multi_chi;

    std::vector<dg::Elliptic3d< Geometry, Matrix, Container> > m_multi_pol;
    std::vector<dg::Helmholtz3d<Geometry, Matrix, Container> > m_multi_invgammaP,
        m_multi_invgammaN, m_multi_ampere;

    //dg::Elliptic3d<Geometry, Matrix, Container> m_laplaceM, m_laplaceMphi;

    dg::Extrapolation<Container> m_old_phi, m_old_psi, m_old_gammaN, m_old_aparST;
    //dg::Extrapolation<Container> m_old_phiST, m_old_psiST, m_old_gammaNST;
};

template<class Geometry, class Matrix, class Container>
Solvers<Geometry, Matrix, Container>::Solvers( const Geometry& g,
    feltor::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue
    ): m_p(p),
    m_dx_P(  dg::create::dx( g, p.bcxP, p.pol_dir) ),
    m_dy_P(  dg::create::dy( g, p.bcyP, p.pol_dir) ),
    m_dz( dg::create::dz( g, dg::PER) ),
    m_multigrid( g, p.stages),
    m_old_phi( 2, dg::evaluate( dg::zero, g)),
    m_old_psi( m_old_phi), m_old_gammaN( m_old_phi),
    m_old_aparST( m_old_phi)
    //m_old_phiST( 2, dg::evaluate( dg::zero, g)),
    //m_old_psiST( m_old_phi), m_old_gammaNST( m_old_phi),
{
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_uE2 =  m_temp1 = m_temp0;
    if( m_p.curvmode == "flutemode")
    {
        dg::assign(  dg::pullback(dg::geo::InvBtor(mag), g), m_binv);
        dg::assign(  dg::pullback(dg::cooX3d, g), m_Rinv);
        dg::blas1::pointwiseDivide( 1., m_Rinv, m_Rinv);
    }
    else
        dg::assign(  dg::pullback(dg::geo::InvB(mag), g), m_binv);

    //Set a hard code limit on the maximum number of iteration to avoid
    //endless iteration in case of failure
    m_multigrid.set_max_iter( 1e5);
    /////////////////////////init elliptic and helmholtz operators/////////
    auto bhat = dg::geo::createEPhi(+1); //bhat = ephi except when "true"
    if( p.curvmode == "true")
        bhat = dg::geo::createBHat( mag);
    m_multi_chi = m_multigrid.project( m_temp0);
    m_multi_pol.resize(p.stages);
    m_multi_invgammaP.resize(p.stages);
    m_multi_invgammaN.resize(p.stages);
    m_multi_ampere.resize(p.stages);
    for( unsigned u=0; u<p.stages; u++)
    {
        m_multi_pol[u].construct( m_multigrid.grid(u),
            p.bcxP, p.bcyP, dg::PER,
            p.pol_dir, p.jfactor);
        m_multi_invgammaP[u] = { -0.5*p.tau[1]*p.mu[1],
                {m_multigrid.grid(u), p.bcxP, p.bcyP, dg::PER, p.pol_dir}};
        m_multi_invgammaN[u] = { -0.5*p.tau[1]*p.mu[1],
                {m_multigrid.grid(u), p.bcxN, p.bcyN, dg::PER, p.pol_dir}};
        m_multi_ampere[u] = {  -1.,
                {m_multigrid.grid(u), p.bcxA, p.bcyA, dg::PER, p.pol_dir}};

        dg::SparseTensor<Container> hh = dg::geo::createProjectionTensor(
            bhat, m_multigrid.grid(u));
        m_multi_pol[u].set_chi( hh);
        m_multi_invgammaP[u].matrix().set_chi( hh);
        m_multi_invgammaN[u].matrix().set_chi( hh);
        m_multi_ampere[u].matrix().set_chi( hh);
        if( p.curvmode == "flutemode")
        {
            Container Rinv = dg::pullback( dg::cooX3d, m_multigrid.grid(u));
            dg::blas1::pointwiseDivide( 1., Rinv, Rinv);
            dg::blas1::pointwiseDot( Rinv, Rinv, Rinv); // = 1/R^2
            m_multi_ampere[u].matrix().set_chi( Rinv);
        }
        if( !((p.curvmode == "true") && (p.symmetric == false))){
            m_multi_pol[u].set_compute_in_2d( true);
            m_multi_invgammaP[u].matrix().set_compute_in_2d( true);
            m_multi_invgammaN[u].matrix().set_compute_in_2d( true);
            m_multi_ampere[u].matrix().set_compute_in_2d( true);
        }
    }
}

template<class Geometry, class Matrix, class Container>
void Solvers<Geometry, Matrix, Container>::compute_phi(
    double time, const std::array<Container,2>& density,
    Container& phi,
    bool penalize_wall, const Container& wall,
    bool penalize_sheath, const Container& sheath
    )
{
    //density[0]:= n_e
    //density[1]:= N_i
    //----------Compute and set chi----------------------------//
    dg::blas1::pointwiseDot( m_p.mu[1], density[1], m_binv, m_binv, 0., m_temp0);
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_pol[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    if (m_p.tau[1] == 0.) {
        //compute N_i - n_e
        dg::blas1::axpby( 1., density[1], -1., density[0], m_temp0);
    }
    else
    {
        dg::blas1::transform( density[1], m_temp1, dg::PLUS<double>(-m_p.nbc));
        //compute Gamma N_i - n_e
        //if( staggered)
        //    m_old_gammaNST.extrapolate( time, m_temp0);
        //else
            m_old_gammaN.extrapolate( time, m_temp0);
        m_multigrid.set_benchmark( true, "Gamma N     ");
        std::vector<unsigned> numberG = m_multigrid.solve(
            m_multi_invgammaN, m_temp0, m_temp1, m_p.eps_gamma);
        //if( staggered)
        //    m_old_gammaNST.update( time, m_temp0); // store N - nbc
        //else
            m_old_gammaN.update( time, m_temp0); // store N - nbc
        dg::blas1::transform( density[0], m_temp1, dg::PLUS<double>(-m_p.nbc));
        dg::blas1::axpby( -1., m_temp1, 1., m_temp0, m_temp0);
    }
    // Add penalization method
    common::multiply_rhs_penalization( m_temp0, penalize_wall, wall,
                    penalize_sheath, sheath); // F*(1-chi_w-chi_s)
    //----------Invert polarisation----------------------------//
    //if( staggered)
    //    m_old_phiST.extrapolate( time, phi);
    //else
        m_old_phi.extrapolate( time, phi);
    m_multigrid.set_benchmark( true, "Polarisation");
    std::vector<unsigned> number = m_multigrid.solve(
        m_multi_pol, phi, m_temp0, m_p.eps_pol);
#ifdef WRITE_POL_FILE
    //if( number[0] > 1000)
        counter++;
    if( counter >= 10 && number.back() > 100) // choose a somewhat difficult timestep
    {
        typename dg::file::NcFile::Hyperslab slab( m_multigrid.grid(0));
        pol_file.defput_var( "chi",  {"z","y","x"}, {}, slab, m_multi_chi[0]);
        pol_file.defput_var( "sol",  {"z","y","x"}, {}, slab, phi);
        pol_file.defput_var( "rhs",  {"z","y","x"}, {}, slab, m_temp0);
        pol_file.defput_var( "ne",   {"z","y","x"}, {}, slab, density[0]);
        pol_file.defput_var( "Ni",   {"z","y","x"}, {}, slab, density[1]);
        pol_file.defput_var( "phiH", {"z","y","x"}, {}, slab, m_old_phi.head());
        m_old_phi.extrapolate( time, phi);
        pol_file.defput_var( "phi0",  {"z","y","x"}, {}, slab, phi);
        pol_file.close();
        dg::abort_program();
    }
#endif // WRITE_POL_FILE
    //if( staggered)
    //    m_old_phiST.update( time, phi);
    //else
        m_old_phi.update( time, phi);

}

template<class Geometry, class Matrix, class Container>
void Solvers<Geometry, Matrix, Container>::compute_aparST(
    double time, const std::array<Container,2>& densityST,
    std::array<Container,2>& velocityST, Container& aparST,
    // update determines if old solution is updated (or start iteration from 0)
    bool update)
{
    //on input
    //densityST[0] = n_e, velocityST[0]:= w_e
    //densityST[1] = N_i, velocityST[1]:= W_i
    //
    // beta is nonzero when this function is called
    //----------Compute and set chi----------------------------//
    dg::blas1::axpby(  m_p.beta/m_p.mu[1], densityST[1],
                      -m_p.beta/m_p.mu[0], densityST[0], m_temp0);
    if( m_p.curvmode == "flutemode")
        dg::blas1::pointwiseDot( 1., m_temp0, m_Rinv, m_Rinv, 1., m_temp0);
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_ampere[u].set_chi( m_multi_chi[u]);
    //----------Compute right hand side------------------------//
    dg::blas1::pointwiseDot(  m_p.beta, densityST[1], velocityST[1],
                             -m_p.beta, densityST[0], velocityST[0],
                              0., m_temp0);
    if( m_p.curvmode == "flutemode")
        dg::blas1::pointwiseDot(  m_temp0, m_Rinv, m_temp0);
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
    if( m_p.curvmode == "flutemode")
        dg::blas1::pointwiseDot( aparST, m_Rinv, aparST); // Aparallel = Avarphi / R
    //----------Compute Velocities-----------------------------//
    dg::blas1::axpby( 1., velocityST[0], -1./m_p.mu[0], aparST, velocityST[0]);
    dg::blas1::axpby( 1., velocityST[1], -1./m_p.mu[1], aparST, velocityST[1]);
}

template<class Geometry, class Matrix, class Container>
void Solvers<Geometry, Matrix, Container>::compute_psi(
    double time,
    const Container& phi,
    Container& psi
    )
{
    //-----------Solve for Gamma Phi---------------------------//
    if (m_p.tau[1] == 0.) {
        dg::blas1::copy( phi, psi);
    } else {
        //if( staggered)
        //    m_old_psiST.extrapolate( time, psi);
        //else
            m_old_psi.extrapolate( time, psi);
        m_multigrid.set_benchmark( true, "Gamma Phi   ");
        std::vector<unsigned> number = m_multigrid.solve(
            m_multi_invgammaP, psi, phi, m_p.eps_gamma);
        //if( staggered)
        //    m_old_psiST.update( time, psi);
        //else
            m_old_psi.update( time, psi);
    }
    //-------Compute Psi and derivatives
    m_multi_invgammaP[0].matrix().variation( m_binv, phi, m_uE2);
    dg::blas1::axpby( -0.5, m_uE2, 1., psi);
}

}//namespace feltor
