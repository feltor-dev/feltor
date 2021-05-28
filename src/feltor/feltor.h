#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/geometries/geometries.h"
#ifndef DG_MANUFACTURED
#define FELTORPARALLEL 1
#define FELTORPERP 1
#endif


//Latest measurement: m = 10.000 per step

namespace feltor
{

template< class Geometry, class IMatrix, class Matrix, class Container >
struct Explicit
{
    using vector = std::array<std::array<Container,2>,2>;
    using container = Container;
    Explicit( const Geometry& g, feltor::Parameters p,
        dg::geo::TokamakMagneticField mag );

    //Given N_i-nbc initialize n_e-nbc such that phi=0
    void initializene( const Container& ni, Container& ne, std::string initphi);
    //Given n_e-nbc initialize N_i-nbc such that phi=0
    void initializeni( const Container& ne, Container& ni, std::string initphi);

    void operator()( double t,
        const std::array<std::array<Container,2>,2>& y,
        std::array<std::array<Container,2>,2>& yp);

    const std::array<std::array<Container,2>,2>& fields() const{
        return m_fields;
    }
    const std::array<Container,2>& potentials() const{
        return m_phi;
    }
    const std::array<std::array<Container,2>,2>& sources() const{
        return m_s;
    }

    /// ///////////////////DIAGNOSTIC MEMBERS //////////////////////
    const Geometry& grid() const {
        return m_multigrid.grid(0);
    }
    //potential[0]: electron potential, potential[1]: ion potential
    const Container& uE2() const {
        return m_UE2;
    }
    const Container& density(int i)const{
        return m_fields[0][i];
    }
    const Container& density_source(int i)const{
        return m_s[0][i];
    }
    const Container& velocity(int i)const{
        return m_fields[1][i];
    }
    const Container& velocity_source(int i)const{
        return m_s[1][i];
    }
    const Container& potential(int i) const {
        return m_phi[i];
    }
    const Container& aparallel() const {
        return m_apar;
    }
    const std::array<Container, 3> & gradN (int i) const {
        return m_dFN[i];
    }
    const std::array<Container, 3> & gradU (int i) const {
        return m_dFU[i];
    }
    const std::array<Container, 3> & gradP (int i) const {
        return m_dP[i];
    }
    const std::array<Container, 3> & gradA () const {
        return m_dA;
    }
    void compute_dsN (int i, Container& dsN, Container& tmp) const {
        // plusN and minusN were computed using y0 (not fields)
        dg::blas1::transform( m_fields[0][i], tmp, dg::PLUS<double>( -m_p.nbc));
        if( m_p.fci_bc == "along_field")
            dg::geo::ds_centered_bc_along_field( m_fa_N, 1., m_minusN[i],
                    tmp, m_plusN[i], 0., dsN, m_p.bcxN, {0,0});
        else
            dg::geo::ds_centered( m_fa_N, 1., m_minusN[i],
                    tmp, m_plusN[i], 0., dsN);
    }
    void compute_dsU (int i, Container& dsU) const {
        if( m_p.fci_bc == "along_field")
            dg::geo::ds_centered_bc_along_field( m_fa_N, 1., m_minusU[i],
                    m_fields[1][i], m_plusU[i], 0., dsU, m_p.bcxU, {0,0});
        else
            dg::geo::ds_centered( m_fa_U, 1., m_minusU[i],
                    m_fields[1][i], m_plusU[i], 0., dsU);
    }
    void compute_dsP (int i, Container& dsP) const {
        if( m_p.fci_bc == "along_field")
            dg::geo::ds_centered_bc_along_field( m_fa_N, 1., m_minusP[i],
                    m_phi[i], m_plusP[i], 0.0, dsP, m_p.bcxP, {0,0});
        else
            dg::geo::ds_centered( m_fa_P, 1., m_minusP[i],
                    m_phi[i], m_plusP[i], 0.0, dsP);
    }
    void compute_dssU(int i, Container& dssU) {
        if( m_p.fci_bc == "along_field")
            dg::geo::dss_centered_bc_along_field( m_fa_N, 1., m_minusU[i],
                m_fields[1][i], m_plusU[i], 0., dssU, m_p.bcxU, {0,0});
        else
            dg::geo::dss_centered( m_fa_U, 1., m_minusU[i],
                m_fields[1][i], m_plusU[i], 0., dssU);
    }
    void compute_gradSN( int i, std::array<Container,3>& gradS) const{
        // MW: don't like this function, if we need more gradients we might
        // want a more flexible solution
        // grad S_ne and grad S_ni
        dg::blas2::symv( m_dxF_N, m_s[0][i], gradS[0]);
        dg::blas2::symv( m_dyF_N, m_s[0][i], gradS[1]);
        if(!m_p.symmetric)dg::blas2::symv( m_dz, m_s[0][i], gradS[2]);
    }
    void compute_dot_aparallel( Container& tmp) const {
        m_old_apar.derive( tmp);
    }
    const dg::SparseTensor<Container>& projection() const{
        return m_hh;
    }
    const std::array<Container, 3> & curv () const {
        return m_curv;
    }
    const std::array<Container, 3> & curvKappa () const {
        return m_curvKappa;
    }
    const Container& divCurvKappa() const {
        return m_divCurvKappa;
    }
    const Container& bphi( ) const { return m_bphi; }
    const Container& binv( ) const { return m_binv; }
    const Container& divb( ) const { return m_divb; }
    //volume with dG weights
    const Container& vol3d() const { return m_lapperpN.weights();}
    const Container& weights() const { return m_lapperpN.weights();}
    //bhat / sqrt{g} / B
    const std::array<Container, 3> & bhatgB () const {
        return m_b;
    }
    const Container& lapMperpP (int i)
    {
        m_lapperpP.set_chi( 1.);
        dg::blas2::gemv( m_lapperpP, m_phi[i], m_temp1);
        return m_temp1;
    }
    const Container& lapMperpA ()
    {
        dg::blas2::gemv( m_lapperpU, m_apar, m_temp1);
        return m_temp1;
    }
    const Container& get_source() const{
        return m_source;
    }
    const Container& get_source_prof() const{
        return m_profne;
    }
    const Container& get_wall() const{
        return m_wall;
    }
    const Container& get_sheath() const{
        return m_sheath;
    }
    const Container& get_sheath_coordinate() const{
        return m_sheath_coordinate;
    }
    /// //////////////////////DIAGNOSTICS END////////////////////////////////
    void compute_lapParN( double alpha, unsigned i,
            double beta, Container& result, Container& temp0 )
    {
        // the temp0 is there because the function is also called internally
        // and then it is important to see which temporary is used
        // result = alpha Delta_par N + beta result
        dg::blas1::transform( m_fields[0][i], temp0, dg::PLUS<double>( -m_p.nbc));
        if( m_p.fci_bc == "along_field")
            dg::geo::dssd_centered_bc_along_field( m_divb, m_fa_N,
                    alpha, m_minusN[i], temp0, m_plusN[i],
                    beta, result, m_p.bcxN, {0,0});
        else
            dg::geo::dssd_centered( m_divb, m_fa_U, alpha,
                    m_minusN[i], temp0, m_plusN[i], beta, result);
    }
    void compute_lapParU( double alpha, unsigned i, double beta,
            Container& result)
    {
        // result = alpha Delta_par U + beta result
        if( m_p.fci_bc == "along_field")
            dg::geo::dssd_centered_bc_along_field( m_divb, m_fa_N,
                    alpha, m_minusU[i], m_fields[1][i], m_plusU[i],
                    beta, result, m_p.bcxU, {0,0});
        else
            dg::geo::dssd_centered( m_divb, m_fa_U, alpha,
                    m_minusU[i], m_fields[1][i], m_plusU[i], beta, result);
    }
    void compute_perp_diffusiveN( double alpha, const Container& density,
            Container& temp0, Container& temp1, double beta, Container& result )
    {
        // density = full N
        // result = alpha Lambda_N + beta result
        if( m_p.nu_perp_n > 0)
        {
            dg::blas1::transform( density, temp0, dg::PLUS<double>(-m_p.nbc));
            for( unsigned s=0; s<m_p.diff_order; s++)
            {
                using std::swap;
                swap( temp0, temp1);
                dg::blas2::symv( 1., m_lapperpN, temp1, 0., temp0);
            }
            dg::blas1::axpby( -alpha*m_p.nu_perp_n, temp0, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
    }
    void compute_perp_diffusiveU( double alpha, const Container& velocity,
            Container& temp0, Container& temp1, double beta, Container& result)
    {
        // density = full N
        // result = alpha Lambda_U + beta result
        if( m_p.nu_perp_u > 0)
        {
            dg::blas1::copy( velocity, temp0);
            for( unsigned s=0; s<m_p.diff_order; s++)
            {
                using std::swap;
                swap( temp0, temp1);
                dg::blas2::symv( 1., m_lapperpU, temp1, 0., temp0);
            }
            dg::blas1::axpby( -alpha*m_p.nu_perp_u, temp0, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
    }

    //source strength, profile - 1
    void set_source( bool fixed_profile, Container profile, double source_rate, Container source, double minne, double minrate, double minalpha)
    {
        m_fixed_profile = fixed_profile;
        m_profne = profile;
        m_source_rate = source_rate;
        m_source = source;
        m_minne = minne;
        m_minrate = minrate;
        m_minalpha = minalpha;
    }
    void set_wall(double wall_rate, const Container& wall, double nwall, double uwall)
    {
        m_wall_rate = wall_rate;
        dg::blas1::copy( wall, m_wall);
        m_nwall = nwall;
        m_uwall = uwall;
    }
    void set_sheath(double sheath_rate, const Container& sheath,
            const Container& sheath_coordinate)
    {
        m_sheath_rate = sheath_rate;
        dg::blas1::copy( sheath, m_sheath);
        dg::blas1::copy( sheath_coordinate, m_sheath_coordinate);
    }
    void compute_apar( double t, std::array<std::array<Container,2>,2>& fields);
    void compute_phi( double t, const std::array<Container,2>& y);
    void compute_psi( double t);
    void compute_perp( double t,
        const std::array<std::array<Container,2>,2>& y,
        const std::array<std::array<Container,2>,2>& fields,
        std::array<std::array<Container,2>,2>& yp);
    void compute_parallel( double t,
        const std::array<std::array<Container,2>,2>& y,
        const std::array<std::array<Container,2>,2>& fields,
        std::array<std::array<Container,2>,2>& yp);
    void add_source_terms( double t,
        const std::array<std::array<Container,2>,2>& y,
        const std::array<std::array<Container,2>,2>& fields,
        std::array<std::array<Container,2>,2>& yp);
    void add_rhs_penalization( std::array<std::array<Container,2>,2>& yp);
    void add_wall_and_sheath_terms( double t,
        const std::array<std::array<Container,2>,2>& fields,
        std::array<std::array<Container,2>,2>& yp);
  private:
    void construct_mag( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);
    void construct_bhat( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);
    void construct_invert( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);

    Container m_UE2;
    Container m_temp0, m_temp1, m_temp2, m_temp3;//helper variables
#ifdef DG_MANUFACTURED
    Container m_R, m_Z, m_P; //coordinates
#endif //DG_MANUFACTURED

    //these should be considered const // m_curv is full curvature
    std::array<Container,3> m_curv, m_curvKappa, m_b; //m_b is bhat/ sqrt(g) / B
    Container m_divCurvKappa;
    Container m_bphi, m_binv, m_divb;
    Container m_source, m_profne, m_sheath_coordinate;
    Container m_wall, m_sheath;

    Container m_apar;
    std::array<Container,2> m_phi;
    std::array<Container,2> m_plusN, m_minusN, m_plusU, m_minusU, m_plusP, m_minusP;
    std::array<Container,3> m_dA;
    std::array<std::array<Container,3>,2> m_dP, m_dFN, m_dBN, m_dFU, m_dBU;
    std::array<std::array<Container,2>,2> m_fields, m_s; //fields, sources

    std::vector<Container> m_multi_chi;

    //matrices and solvers
    Matrix m_dxF_N, m_dxB_N, m_dxF_U, m_dxB_U, m_dx_P, m_dx_A;
    Matrix m_dyF_N, m_dyB_N, m_dyF_U, m_dyB_U, m_dy_P, m_dy_A, m_dz;
    dg::geo::Fieldaligned<Geometry, IMatrix, Container> m_fa_N, m_fa_U, m_fa_P;
    dg::Elliptic3d< Geometry, Matrix, Container> m_lapperpN, m_lapperpU, m_lapperpP;
    std::vector<dg::Elliptic3d< Geometry, Matrix, Container> > m_multi_pol;
    std::vector<dg::Helmholtz3d<Geometry, Matrix, Container> > m_multi_invgammaP,
        m_multi_invgammaN, m_multi_ampere;

    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    dg::Extrapolation<Container> m_old_phi, m_old_psi, m_old_gammaN, m_old_apar;

    dg::SparseTensor<Container> m_hh;

    const feltor::Parameters m_p;
    double m_source_rate = 0., m_sheath_rate = 0., m_wall_rate = 0.;
    double m_minne = 0., m_minrate  = 0., m_minalpha = 0.;
    double m_nwall = 0., m_uwall = 0.;
    bool m_fixed_profile = true, m_reversed_field = false;

};

template<class Grid, class IMatrix, class Matrix, class Container>
void Explicit<Grid, IMatrix, Matrix, Container>::construct_mag(
    const Grid& g, feltor::Parameters p, dg::geo::TokamakMagneticField mag)
{
    //due to the various approximations bhat and mag not always correspond
    dg::geo::CylindricalVectorLvl0 curvNabla, curvKappa;
    m_reversed_field = false;
    if( mag.ipol()( g.x0(), g.y0()) < 0)
        m_reversed_field = true;
    if( p.curvmode == "true" )
    {
        curvNabla = dg::geo::createTrueCurvatureNablaB(mag);
        curvKappa = dg::geo::createTrueCurvatureKappa(mag);
        dg::assign(  dg::pullback(dg::geo::TrueDivCurvatureKappa(mag), g),
            m_divCurvKappa);
    }
    else if( p.curvmode == "low beta")
    {
        if( m_reversed_field)
            curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag, -1);
        else
            curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag, +1);
        dg::assign( dg::evaluate(dg::zero, g), m_divCurvKappa);
    }
    else if( p.curvmode == "toroidal")
    {
        if( m_reversed_field)
        {
            curvNabla = dg::geo::createCurvatureNablaB(mag, -1);
            curvKappa = dg::geo::createCurvatureKappa(mag, -1);
            dg::assign(  dg::pullback(dg::geo::DivCurvatureKappa(mag, -1), g),
                m_divCurvKappa);
        }
        else
        {
            curvNabla = dg::geo::createCurvatureNablaB(mag, +1);
            curvKappa = dg::geo::createCurvatureKappa(mag, +1);
            dg::assign(  dg::pullback(dg::geo::DivCurvatureKappa(mag, +1), g),
                m_divCurvKappa);
        }
    }
    else
        throw std::runtime_error( "Warning! curvmode value '"+p.curvmode+"' not recognized!! I don't know what to do! I exit!\n");
    dg::pushForward(curvNabla.x(), curvNabla.y(), curvNabla.z(),
        m_curv[0], m_curv[1], m_curv[2], g);
    dg::pushForward(curvKappa.x(), curvKappa.y(), curvKappa.z(),
        m_curvKappa[0], m_curvKappa[1], m_curvKappa[2], g);
    dg::blas1::axpby( 1., m_curvKappa, 1., m_curv);
    dg::assign(  dg::pullback(dg::geo::InvB(mag), g), m_binv);
    dg::assign(  dg::pullback(dg::geo::Divb(mag), g), m_divb);

}
template<class Grid, class IMatrix, class Matrix, class Container>
void Explicit<Grid, IMatrix, Matrix, Container>::construct_bhat(
    const Grid& g, feltor::Parameters p, dg::geo::TokamakMagneticField mag)
{
    //in DS we take the true bhat
    auto bhat = dg::geo::createBHat( mag);
    // do not construct FCI if we just want to calibrate
    if( !p.calibrate )
    {
        m_fa_N.construct( bhat, g, p.bcxN, p.bcyN, dg::geo::NoLimiter(),
            p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz );
        if ( p.fci_bc == "perp")
        {
            if( p.bcxU == p.bcxN && p.bcyU == p.bcyN)
                m_fa_U.construct( m_fa_N);
            else
                m_fa_U.construct( bhat, g, p.bcxU, p.bcyU, dg::geo::NoLimiter(),
                    p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz);
            if( p.bcxP == p.bcxN && p.bcyP == p.bcyN)
                m_fa_P.construct( m_fa_N);
            else if( p.bcxP == p.bcxU && p.bcyP == p.bcyU)
                m_fa_P.construct( m_fa_U);
            else
                m_fa_P.construct( bhat, g, p.bcxP, p.bcyP, dg::geo::NoLimiter(),
                     p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz);
        }
    }

    // in Poisson we take EPhi except for the true curvmode
    bhat = dg::geo::createEPhi(+1);
    if( p.curvmode == "true")
        bhat = dg::geo::createBHat(mag);
    else if( m_reversed_field)
        bhat = dg::geo::createEPhi(-1);
    dg::pushForward(bhat.x(), bhat.y(), bhat.z(), m_b[0], m_b[1], m_b[2], g);
    dg::SparseTensor<Container> metric = g.metric();
    dg::tensor::inv_multiply3d( metric, m_b[0], m_b[1], m_b[2],
                                        m_b[0], m_b[1], m_b[2]);
    dg::assign( m_b[2], m_bphi); //save bphi for momentum conservation
    Container detg = dg::tensor::volume( metric);
    dg::blas1::pointwiseDivide( m_binv, detg, m_temp0); //1/B/detg
    for( int i=0; i<3; i++)
        dg::blas1::pointwiseDot( m_temp0, m_b[i], m_b[i]); //b_i/detg/B
    m_hh = dg::geo::createProjectionTensor( bhat, g);
    m_lapperpN.construct ( g, p.bcxN, p.bcyN, dg::PER, dg::normed, m_p.diff_dir),
    m_lapperpU.construct ( g, p.bcxU, p.bcyU, dg::PER, dg::normed, m_p.diff_dir),
    m_lapperpP.construct ( g, p.bcxP, p.bcyP, dg::PER, dg::normed, dg::centered),
    m_lapperpN.set_chi( m_hh);
    m_lapperpU.set_chi( m_hh);
    m_lapperpP.set_chi( m_hh);
    if( p.curvmode != "true")
    {
        m_lapperpN.set_compute_in_2d(true);
        m_lapperpU.set_compute_in_2d(true);
        m_lapperpP.set_compute_in_2d(true);
    }
    m_lapperpP.set_jfactor(0); //we don't want jump terms in source
}
template<class Grid, class IMatrix, class Matrix, class Container>
void Explicit<Grid, IMatrix, Matrix, Container>::construct_invert(
    const Grid& g, feltor::Parameters p, dg::geo::TokamakMagneticField mag)
{
    //Set a hard code limit on the maximum number of iteration to avoid
    //endless iteration in case of failure
    m_multigrid.set_max_iter( 1e5);
    /////////////////////////init elliptic and helmholtz operators/////////
    auto bhat = dg::geo::createEPhi(+1); //bhat = ephi except when "true"
    if( p.curvmode == "true")
        bhat = dg::geo::createBHat( mag);
    else if( m_reversed_field)
        bhat = dg::geo::createEPhi(-1);
    m_multi_chi = m_multigrid.project( m_temp0);
    m_multi_pol.resize(p.stages);
    m_multi_invgammaP.resize(p.stages);
    m_multi_invgammaN.resize(p.stages);
    m_multi_ampere.resize(p.stages);
    for( unsigned u=0; u<p.stages; u++)
    {
        m_multi_pol[u].construct( m_multigrid.grid(u),
            p.bcxP, p.bcyP, dg::PER, dg::not_normed,
            p.pol_dir, p.jfactor);
        m_multi_invgammaP[u].construct(  m_multigrid.grid(u),
            p.bcxP, p.bcyP, dg::PER, -0.5*p.tau[1]*p.mu[1], p.pol_dir);
        m_multi_invgammaN[u].construct(  m_multigrid.grid(u),
            p.bcxN, p.bcyN, dg::PER, -0.5*p.tau[1]*p.mu[1], p.pol_dir);
        m_multi_ampere[u].construct(  m_multigrid.grid(u),
            p.bcxA, p.bcyA, dg::PER, -1., p.pol_dir);

        dg::SparseTensor<Container> hh = dg::geo::createProjectionTensor(
            bhat, m_multigrid.grid(u));
        m_multi_pol[u].set_chi( hh);
        m_multi_invgammaP[u].elliptic().set_chi( hh);
        m_multi_invgammaN[u].elliptic().set_chi( hh);
        m_multi_ampere[u].elliptic().set_chi( hh);
        if(p.curvmode != "true"){
            m_multi_pol[u].set_compute_in_2d( true);
            m_multi_invgammaP[u].elliptic().set_compute_in_2d( true);
            m_multi_invgammaN[u].elliptic().set_compute_in_2d( true);
            m_multi_ampere[u].elliptic().set_compute_in_2d( true);
        }
    }
}
template<class Grid, class IMatrix, class Matrix, class Container>
Explicit<Grid, IMatrix, Matrix, Container>::Explicit( const Grid& g,
    feltor::Parameters p, dg::geo::TokamakMagneticField mag):
#ifdef DG_MANUFACTURED
    m_R( dg::pullback( dg::cooX3d, g)),
    m_Z( dg::pullback( dg::cooY3d, g)),
    m_P( dg::pullback( dg::cooZ3d, g)),
#endif //DG_MANUFACTURED
    m_dxF_N( dg::create::dx( g, p.bcxN, dg::forward) ),
    m_dxB_N( dg::create::dx( g, p.bcxN, dg::backward) ),
    m_dxF_U( dg::create::dx( g, p.bcxU, dg::forward) ),
    m_dxB_U( dg::create::dx( g, p.bcxU, dg::backward) ),
    m_dx_P(  dg::create::dx( g, p.bcxP, dg::centered) ),
    m_dx_A(  dg::create::dx( g, p.bcxA, dg::centered) ),
    m_dyF_N( dg::create::dy( g, p.bcyN, dg::forward) ),
    m_dyB_N( dg::create::dy( g, p.bcyN, dg::backward) ),
    m_dyF_U( dg::create::dy( g, p.bcyU, dg::forward) ),
    m_dyB_U( dg::create::dy( g, p.bcyU, dg::backward) ),
    m_dy_P(  dg::create::dy( g, p.bcyP, dg::centered) ),
    m_dy_A(  dg::create::dy( g, p.bcyA, dg::centered) ),
    m_dz( dg::create::dz( g, dg::PER) ),
    m_multigrid( g, p.stages),
    m_old_phi( 2, dg::evaluate( dg::zero, g)),
    m_old_psi( m_old_phi), m_old_gammaN( m_old_phi), m_old_apar( m_old_phi),
    m_p(p)
{
    //--------------------------init vectors to 0-----------------//
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_source = m_sheath_coordinate = m_UE2 = m_temp3 = m_temp2 = m_temp1 = m_temp0;
    m_apar = m_profne = m_wall = m_sheath = m_temp0;

    m_phi[0] = m_phi[1] = m_temp0;
    m_plusN = m_minusN = m_minusU = m_plusU = m_minusP = m_plusP = m_phi;
    m_dA[0] = m_dA[1] = m_dA[2] = m_temp0;
    m_dP[0] = m_dP[1] = m_dA;
    m_dFN = m_dBN = m_dFU = m_dBU = m_dP;
    m_fields[0] = m_fields[1] = m_phi;
    m_s = m_fields;

    //--------------------------Construct-------------------------//
    construct_mag( g, p, mag);
    construct_bhat( g, p, mag);
    construct_invert( g, p, mag);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::initializene(
    const Container& src, Container& target, std::string initphi)
{
    // ne - nbc = Ni - nbc
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        if( initphi == "zero")
        {
            // ne-nbc = Gamma (ni-nbc)
            std::vector<unsigned> number = m_multigrid.direct_solve(
                m_multi_invgammaN, target, src, m_p.eps_gamma);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma);
        }
        else if( initphi == "balance")
        {
            //add FLR correction -0.5*tau*mu*Delta n_e
            dg::blas2::symv( 0.5*m_p.tau[1]*m_p.mu[1],
                m_lapperpN, src, 1.0, target);
            //wird stark negativ falls alpha klein!!
        }
        else if( !(initphi == "zero_pol"))
        {
            throw dg::Error(dg::Message(_ping_)<<"Warning! initphi value '"<<initphi<<"' not recognized. I have tau = "<<m_p.tau[1]<<" ! I don't know what to do! I exit!\n");
        }
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::initializeni(
    const Container& src, Container& target, std::string initphi)
{
    //According to Markus we should actually always invert
    //so we should reconsider this function
    // Ni - nbc = ne - nbc
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        if( initphi == "zero")
        {
            //add FLR correction -0.5*tau*mu*Delta n_e
            dg::blas2::symv( 0.5*m_p.tau[1]*m_p.mu[1],
                m_lapperpN, src, 1.0, target);
            //wird stark negativ falls alpha klein!!
        }
        else if( initphi == "balance")
            //add FLR correction +0.5*tau*mu*Delta n_e
            dg::blas2::symv( -0.5*m_p.tau[1]*m_p.mu[1],
                m_lapperpN, src, 1.0, target);
            //wird stark negativ falls alpha klein!!
        else if( !(initphi == "zero_pol"))
        {
            throw dg::Error(dg::Message(_ping_)<<"Warning! initphi value '"<<initphi<<"' not recognized. I have tau = "<<m_p.tau[1]<<" ! I don't know what to do! I exit!\n");
        }
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_phi(
    double time, const std::array<Container,2>& y)
{
    //y[0]:= n_e - nbc
    //y[1]:= N_i - nbc
    //----------Compute and set chi----------------------------//
    double nbc = m_p.nbc;
    dg::blas1::subroutine(
        [nbc] DG_DEVICE ( double& chi, double tilde_Ni, double binv,
        double mu_i) {
            chi = mu_i*(tilde_Ni+nbc)*binv*binv;
        },
        m_temp0, y[1], m_binv, m_p.mu[1]);
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_pol[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    if (m_p.tau[1] == 0.) {
        //compute N_i - n_e
        dg::blas1::axpby( 1., y[1], -1., y[0], m_temp0);
    }
    else
    {
        //compute Gamma N_i - n_e
        m_old_gammaN.extrapolate( time, m_temp0);
#ifdef DG_MANUFACTURED
        dg::blas1::copy( y[1], m_temp1);
        dg::blas1::evaluate( m_temp1, dg::plus_equals(), manufactured::SGammaNi{
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
            m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
        std::vector<unsigned> numberG = m_multigrid.direct_solve(
            m_multi_invgammaN, m_temp0, m_temp1, m_p.eps_gamma);
#else
        std::vector<unsigned> numberG = m_multigrid.direct_solve(
            m_multi_invgammaN, m_temp0, y[1], m_p.eps_gamma);
#endif //DG_MANUFACTURED
        m_old_gammaN.update( time, m_temp0);
        if(  numberG[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
        dg::blas1::axpby( -1., y[0], 1., m_temp0, m_temp0);
    }
#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SPhie{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
    //----------Invert polarisation----------------------------//
    m_old_phi.extrapolate( time, m_phi[0]);
    std::vector<unsigned> number = m_multigrid.direct_solve(
        m_multi_pol, m_phi[0], m_temp0, m_p.eps_pol);
    m_old_phi.update( time, m_phi[0]);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol[0]);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_psi(
    double time)
{
    //-----------Solve for Gamma Phi---------------------------//
    if (m_p.tau[1] == 0.) {
        dg::blas1::copy( m_phi[0], m_phi[1]);
    } else {
        m_old_psi.extrapolate( time, m_phi[1]);
#ifdef DG_MANUFACTURED
        dg::blas1::copy( m_phi[0], m_temp0);
        dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SGammaPhie{
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
            m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
        std::vector<unsigned> number = m_multigrid.direct_solve(
            m_multi_invgammaP, m_phi[1], m_temp0, m_p.eps_gamma);
#else
        std::vector<unsigned> number = m_multigrid.direct_solve(
            m_multi_invgammaP, m_phi[1], m_phi[0], m_p.eps_gamma);
#endif //DG_MANUFACTURED
        m_old_psi.update( time, m_phi[1]);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
    }
    //-------Compute Psi and derivatives
    dg::blas2::symv( m_dx_P, m_phi[0], m_dP[0][0]);
    dg::blas2::symv( m_dy_P, m_phi[0], m_dP[0][1]);
    if( !m_p.symmetric) dg::blas2::symv( m_dz, m_phi[0], m_dP[0][2]);
    dg::tensor::scalar_product3d( 1., m_binv,
        m_dP[0][0], m_dP[0][1], m_dP[0][2], m_hh, m_binv, //grad_perp
        m_dP[0][0], m_dP[0][1], m_dP[0][2], 0., m_UE2);
    dg::blas1::axpby( -0.5, m_UE2, 1., m_phi[1]);
#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( m_phi[1], dg::plus_equals(), manufactured::SPhii{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
    //m_UE2 now contains u_E^2; also update derivatives
    dg::blas2::symv( m_dx_P, m_phi[1], m_dP[1][0]);
    dg::blas2::symv( m_dy_P, m_phi[1], m_dP[1][1]);
    if( !m_p.symmetric) dg::blas2::symv( m_dz, m_phi[1], m_dP[1][2]);
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_apar(
    double time, std::array<std::array<Container,2>,2>& fields)
{
    //on input
    //fields[0][0] = n_e, fields[1][0]:= w_e
    //fields[0][1] = N_i, fields[1][1]:= W_i
    //----------Compute and set chi----------------------------//
    dg::blas1::axpby(  m_p.beta/m_p.mu[1], fields[0][1],
                      -m_p.beta/m_p.mu[0], fields[0][0], m_temp0);
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_ampere[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    dg::blas1::pointwiseDot(  m_p.beta, fields[0][1], fields[1][1],
                             -m_p.beta, fields[0][0], fields[1][0],
                              0., m_temp0);
    //----------Invert Induction Eq----------------------------//
    m_old_apar.extrapolate( time, m_apar);
    std::vector<unsigned> number = m_multigrid.direct_solve(
        m_multi_ampere, m_apar, m_temp0, m_p.eps_ampere);
    m_old_apar.update( time, m_apar);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol[0]);
#ifdef DG_MANUFACTURED
    //dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SA{
    //    m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
    //    m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
    //here we cheat (a bit)
    dg::blas1::evaluate( m_apar, dg::equals(), manufactured::A{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
    //----------Compute Derivatives----------------------------//
    dg::blas2::symv( m_dx_A, m_apar, m_dA[0]);
    dg::blas2::symv( m_dy_A, m_apar, m_dA[1]);
    if(!m_p.symmetric) dg::blas2::symv( m_dz, m_apar, m_dA[2]);

    //----------Compute Velocities-----------------------------//
    dg::blas1::axpby( 1., fields[1][0], -1./m_p.mu[0], m_apar, fields[1][0]);
    dg::blas1::axpby( 1., fields[1][1], -1./m_p.mu[1], m_apar, fields[1][1]);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_perp(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    const std::array<std::array<Container,2>,2>& fields,
    std::array<std::array<Container,2>,2>& yp)
{
    //MW: we have the possibility to
    // make the implementation conservative since the perp boundaries are
    // penalized away
    //y[0] = N-nbc, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        //First compute forward and backward derivatives for upwind scheme
        dg::blas2::symv( m_dxF_N, y[0][i], m_dFN[i][0]);
        dg::blas2::symv( m_dyF_N, y[0][i], m_dFN[i][1]);
        dg::blas2::symv( m_dxB_N, y[0][i], m_dBN[i][0]);
        dg::blas2::symv( m_dyB_N, y[0][i], m_dBN[i][1]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, y[0][i], m_dFN[i][2]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, y[0][i], m_dBN[i][2]);
        dg::blas2::symv( m_dxF_U, fields[1][i], m_dFU[i][0]);
        dg::blas2::symv( m_dyF_U, fields[1][i], m_dFU[i][1]);
        dg::blas2::symv( m_dxB_U, fields[1][i], m_dBU[i][0]);
        dg::blas2::symv( m_dyB_U, fields[1][i], m_dBU[i][1]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, fields[1][i], m_dFU[i][2]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, fields[1][i], m_dBU[i][2]);
        double mu = m_p.mu[i], tau = m_p.tau[i], beta = m_p.beta;
        dg::blas1::subroutine( [mu, tau, beta] DG_DEVICE (
                double N, double d0FN, double d1FN, double d2FN,
                          double d0BN, double d1BN, double d2BN,
                double U, double d0FU, double d1FU, double d2FU,
                          double d0BU, double d1BU, double d2BU,
                          double d0P, double d1P, double d2P,
                double A, double d0A, double d1A, double d2A,
                double b_0,         double b_1,         double b_2,
                double curv0,       double curv1,       double curv2,
                double curvKappa0,  double curvKappa1,  double curvKappa2,
                double divCurvKappa,
                double& dtN, double& dtU
            )
            {
                dtN = dtU = 0;
                // density - upwind scheme
                double v0 = (b_1*d2P - b_2*d1P) + tau*curv0 + mu*U*U*curvKappa0;
                double v1 = (b_2*d0P - b_0*d2P) + tau*curv1 + mu*U*U*curvKappa1;
                double v2 = (b_0*d1P - b_1*d0P) + tau*curv2 + mu*U*U*curvKappa2;
                double bp0 = 0., bp1 = 0., bp2 = 0.;
                if( beta != 0)
                {
                    bp0 = A * curvKappa0 + ( d1A*b_2 - d2A*b_1);
                    bp1 = A * curvKappa1 + ( d2A*b_0 - d0A*b_2);
                    bp2 = A * curvKappa2 + ( d0A*b_1 - d1A*b_0);

                    v0 += U * bp0;
                    v1 += U * bp1;
                    v2 += U * bp2;
                    //Q: doesn't U in U^2K_kappa and U b_perp create nonlinearity
                    //in velocity equation that may create shocks?
                    //A: we did some studies in the reconnection2d program and
                    //did not find shocks. LeVeque argues that for smooth
                    //solutions the upwind discretization should be fine but is
                    //wrong for shocks
                }
                dtN += ( v0 > 0 ) ? -v0*d0BN : -v0*d0FN;
                dtN += ( v1 > 0 ) ? -v1*d1BN : -v1*d1FN;
                dtN += ( v2 > 0 ) ? -v2*d2BN : -v2*d2FN;
                // velocity - upwind scheme
                v0 += 2.*tau*curvKappa0;
                v1 += 2.*tau*curvKappa1;
                v2 += 2.*tau*curvKappa2;
                dtU += ( v0 > 0 ) ? -v0*d0BU : -v0*d0FU;
                dtU += ( v1 > 0 ) ? -v1*d1BU : -v1*d1FU;
                dtU += ( v2 > 0 ) ? -v2*d2BU : -v2*d2FU;

                // use centered derivatives
                double KappaU = ( curvKappa0*(d0FU+d0BU)+curvKappa1*(d1FU+d1BU)
                        +curvKappa2*(d2FU+d2BU) ) / 2.;
                double KappaN = ( curvKappa0*(d0FN+d0BN)+curvKappa1*(d1FN+d1BN)
                        +curvKappa2*(d2FN+d2BN) ) / 2.;
                double KappaP = curvKappa0*d0P+curvKappa1*d1P+curvKappa2*d2P;
                double KP = curv0*d0P+curv1*d1P+curv2*d2P;

                dtN +=  - N * ( KP + mu * U * U * divCurvKappa
                                + 2. * mu * U * KappaU);
                dtU +=  - U * ( 2. * tau * KappaN / N + tau * divCurvKappa
                                + KappaP);
                if( beta != 0)
                {
                    double divbp = A*divCurvKappa
                                     - (curv0-curvKappa0)*d0A
                                     - (curv1-curvKappa1)*d1A
                                     - (curv2-curvKappa2)*d2A;
                    double bpU = bp0*( d0FU + d0BU) / 2. +
                                 bp1*( d1FU + d1BU) / 2. +
                                 bp2*( d2FU + d2BU) / 2.;
                    double bpN = bp0*( d0FN + d0BN) / 2. +
                                 bp1*( d1FN + d1BN) / 2. +
                                 bp2*( d2FN + d2BN) / 2.;
                    double bpP = bp0 * d0P + bp1 * d1P + bp2 * d2P;
                    dtN +=  -N*( U*divbp + bpU);
                    dtU +=  - bpP/mu - tau/mu * bpN/N;
                }
            },
            //species depdendent
            fields[0][i], m_dFN[i][0], m_dFN[i][1], m_dFN[i][2],
                          m_dBN[i][0], m_dBN[i][1], m_dBN[i][2],
            fields[1][i], m_dFU[i][0], m_dFU[i][1], m_dFU[i][2],
                          m_dBU[i][0], m_dBU[i][1], m_dBU[i][2],
                          m_dP[i][0], m_dP[i][1], m_dP[i][2],
            //aparallel
            m_apar, m_dA[0], m_dA[1], m_dA[2],
            //magnetic parameters
            m_b[0], m_b[1], m_b[2],
            m_curv[0], m_curv[1], m_curv[2],
            m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
            m_divCurvKappa, yp[0][i], yp[1][i]
        );
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_parallel(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    const std::array<std::array<Container,2>,2>& fields,
    std::array<std::array<Container,2>,2>& yp)
{
    //y[0] = N-nbc, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {

        m_fa_N( dg::geo::einsMinus, y[0][i], m_minusN[i]);
        m_fa_N( dg::geo::einsPlus,  y[0][i], m_plusN[i]);
        if( m_p.fci_bc == "along_field")
        {
            m_fa_N( dg::geo::einsMinus, fields[1][i], m_minusU[i]);
            m_fa_N( dg::geo::einsPlus,  fields[1][i], m_plusU[i]);
            m_fa_N( dg::geo::einsMinus, m_phi[i], m_minusP[i]);
            m_fa_N( dg::geo::einsPlus,  m_phi[i], m_plusP[i]);
            dg::geo::ds_centered_bc_along_field( m_fa_N, 1., m_minusN[i],
                    y[0][i], m_plusN[i], 0., m_temp0, m_p.bcxN, {0,0});
            dg::geo::ds_centered_bc_along_field( m_fa_N, 1., m_minusU[i],
                    fields[1][i], m_plusU[i], 0., m_temp1, m_p.bcxU, {0,0});
        }
        else
        {
            m_fa_U( dg::geo::einsMinus, fields[1][i], m_minusU[i]);
            m_fa_U( dg::geo::einsPlus,  fields[1][i], m_plusU[i]);
            m_fa_P( dg::geo::einsMinus, m_phi[i], m_minusP[i]);
            m_fa_P( dg::geo::einsPlus,  m_phi[i], m_plusP[i]);
            dg::geo::ds_centered( m_fa_N, 1., m_minusN[i], y[0][i], m_plusN[i],
                    0., m_temp0);
            dg::geo::ds_centered( m_fa_U, 1., m_minusU[i], fields[1][i], m_plusU[i],
                    0., m_temp1);
        }
        // m_temp0 -> dsN
        // m_temp1 -> dsU
        //---------------------density--------------------------//
        //density: -Div ( NUb)
        //dg::blas1::pointwiseDot(-1., m_temp0, fields[1][i],
        //    -1., fields[0][i], m_temp1, 1., yp[0][i] );
        dg::blas1::pointwiseDot( -1., fields[0][i],fields[1][i],m_divb,
            1.,yp[0][i]);
        dg::blas1::pointwiseDot( -1., fields[0][i], m_temp1, 1., yp[0][i] );
        if( "centered" == m_p.parallel_advection)
            dg::blas1::pointwiseDot(-1., m_temp0, fields[1][i], 1., yp[0][i]);
        else if( "upwind" == m_p.parallel_advection)
        {
            dg::geo::ds_backward( m_fa_N, 1., m_minusN[i], y[0][i], 0., m_temp2);
            dg::geo::ds_forward( m_fa_N, 1., y[0][i], m_plusN[i], 0., m_temp3);
            if( !m_reversed_field)
                dg::blas1::evaluate( yp[0][i], dg::minus_equals(),
                        dg::UpwindProduct(), fields[1][i], m_temp2, m_temp3);
            else
                dg::blas1::evaluate( yp[0][i], dg::minus_equals(),
                        dg::UpwindProduct(), fields[1][i], m_temp3, m_temp2);
        }
        else if( "upwind2" == m_p.parallel_advection)
        {
            m_fa_N( dg::geo::einsMinus, m_minusN[i], m_temp2);
            dg::geo::ds_backward2( m_fa_N, 1., m_temp2, m_minusN[i], y[0][i],
                    0., m_temp2);
            m_fa_N( dg::geo::einsPlus,  m_plusN[i], m_temp3);
            dg::geo::ds_forward2( m_fa_N, 1., y[0][i], m_plusN[i], m_temp3, 0.,
                    m_temp3);
            if( !m_reversed_field)
                dg::blas1::evaluate( yp[0][i], dg::minus_equals(),
                        dg::UpwindProduct(), fields[1][i], m_temp2, m_temp3);
            else
                dg::blas1::evaluate( yp[0][i], dg::minus_equals(),
                        dg::UpwindProduct(), fields[1][i], m_temp3, m_temp2);
        }
        // density regularization added later

        //---------------------velocity-------------------------//
        // Burgers term: -U ds U
        dg::blas1::pointwiseDot(-1., fields[1][i], m_temp1, 1., yp[1][i]);
        // force terms: -tau/mu * ds N/N -1/mu * ds Phi
        dg::blas1::pointwiseDivide( -m_p.tau[i]/m_p.mu[i], m_temp0,
                fields[0][i], 1., yp[1][i]);
        if( m_p.fci_bc == "along_field")
            dg::geo::ds_centered_bc_along_field( m_fa_N, -1./m_p.mu[i],
                    m_minusP[i], m_phi[i], m_plusP[i], 1.0, yp[1][i],
                    m_p.bcxP, {0,0});
        else
            dg::geo::ds_centered( m_fa_P, -1./m_p.mu[i], m_minusP[i],
                m_phi[i], m_plusP[i], 1.0, yp[1][i]);
        // viscosity: + nu_par Delta_par U/N
        compute_lapParU( m_p.nu_parallel_u[i], i, 0., m_temp2);
        dg::blas1::pointwiseDivide( 1., m_temp2, m_fields[0][i], 1., yp[1][i]);
        // Maybe factor this out in an operator splitting method? To get larger timestep
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_source_terms(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    const std::array<std::array<Container,2>,2>& fields,
    std::array<std::array<Container,2>,2>& yp)
{
    if( m_source_rate != 0.0)
    {
        if( m_fixed_profile )
            dg::blas1::subroutine(
                [] DG_DEVICE ( double& result, double ne, double profne,
                    double source, double source_rate){
                    result = source_rate*source*(profne - ne);
                    },
                m_s[0][0], m_fields[0][0], m_profne, m_source, m_source_rate);
        else
            dg::blas1::axpby( m_source_rate, m_source, 0., m_s[0][0]);
    }
    else
        dg::blas1::copy( 0., m_s[0][0]);
    // add prevention to get below lower limit
    if( m_minrate != 0.0)
    {
        // do not make lower forcing a velocity source
        dg::blas1::transform( m_fields[0][0], m_temp0, dg::PolynomialHeaviside(
                    m_minne-m_minalpha/2., m_minalpha/2., -1) );
        dg::blas1::transform( m_fields[0][0], m_temp1, dg::PLUS<double>( -m_minne));
        dg::blas1::pointwiseDot( -m_minrate, m_temp1, m_temp0, 1., yp[0][0]);
        dg::blas1::transform( m_fields[0][1], m_temp0, dg::PolynomialHeaviside(
                    m_minne-m_minalpha/2., m_minalpha/2., -1) );
        dg::blas1::transform( m_fields[0][1], m_temp1, dg::PLUS<double>( -m_minne));
        dg::blas1::pointwiseDot( -m_minrate, m_temp1, m_temp0, 1., yp[0][1]);
    }

    //compute FLR corrections S_N = (1-0.5*mu*tau*Lap)*S_n
    dg::blas2::gemv( m_lapperpN, m_s[0][0], m_temp0);
    dg::blas1::axpby( 1., m_s[0][0], 0.5*m_p.tau[1]*m_p.mu[1], m_temp0, m_s[0][1]);
    // potential part of FLR correction S_N += -div*(mu S_n grad*Phi/B^2)
    dg::blas1::pointwiseDot( m_p.mu[1], m_s[0][0], m_binv, m_binv, 0., m_temp0);
    m_lapperpP.set_chi( m_temp0);
    m_lapperpP.symv( 1., m_phi[0], 1., m_s[0][1]);

    // S_U += - U S_N/N
    dg::blas1::pointwiseDot( -1.,  fields[1][0],  m_s[0][0], 0., m_temp0);
    dg::blas1::pointwiseDot( -1.,  fields[1][1],  m_s[0][1], 0., m_temp1);
    dg::blas1::pointwiseDivide( 1.,  m_temp0,  fields[0][0], 1., m_s[1][0]);
    dg::blas1::pointwiseDivide( 1.,  m_temp1,  fields[0][1], 1., m_s[1][1]);

    //Add all to the right hand side
    dg::blas1::axpby( 1., m_s, 1.0, yp);
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_rhs_penalization(
        std::array<std::array<Container,2>,2>& yp)
{
    //mask right hand side in penalization region
    for( unsigned i=0; i<2; i++)
        for( unsigned j=0; j<2; j++)
        {
            if( m_p.penalize_wall && m_p.penalize_sheath)
            {
                dg::blas1::subroutine( []DG_DEVICE(
                    double& rhs, double wall, double sheath){
                        rhs *= (1.0-wall-sheath);
                    }, yp[i][j], m_wall, m_sheath);
            }
            else if( m_p.penalize_wall)
            {
                dg::blas1::subroutine( []DG_DEVICE( double& rhs, double wall){
                        rhs *= (1.0-wall); }, yp[i][j], m_wall);
            }
            else if( m_p.penalize_sheath)
            {
                dg::blas1::subroutine( []DG_DEVICE( double& rhs, double sheath){
                        rhs *= (1.0-sheath); }, yp[i][j], m_sheath);
            }
        }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_wall_and_sheath_terms(
        double t,
        const std::array<std::array<Container,2>,2>& fields,
        std::array<std::array<Container,2>,2>& yp)
{
    // add sheath boundary conditions
    if( m_sheath_rate != 0)
    {
        //density
        //Here, we need to find out where "downstream" is
        for( unsigned i=0; i<2; i++)
        {
            //The coordinate automatically sees the reversed field
            //but m_plus and m_minus are defined wrt the angle coordinate
            if( m_reversed_field) //bphi negative (exchange + and -)
                dg::blas1::evaluate( m_temp0, dg::equals(), dg::Upwind(),
                     m_sheath_coordinate, m_plusN[i], m_minusN[i]);
            else
                dg::blas1::evaluate( m_temp0, dg::equals(), dg::Upwind(),
                     m_sheath_coordinate, m_minusN[i], m_plusN[i]);
            // plus and minus were generated with y[0] not fields[0]
            dg::blas1::plus( m_temp0, m_p.nbc);
            dg::blas1::pointwiseDot( m_sheath_rate, m_temp0, m_sheath, 1.,
                    yp[0][i]);
        }
        //compute sheath velocity

        //velocity c_s
        if( "insulating" == m_p.sheath_bc)
        {
            // u_e,sh = s*sqrt(1+tau) Ni/ne
            dg::blas1::subroutine( []DG_DEVICE( double& tmp,
                        double sheath_coord, double ne, double ni)
                    { tmp = sheath_coord*ni/ne;},
                    m_temp0, m_sheath_coordinate, fields[0][0], fields[0][1]);
            dg::blas1::pointwiseDot( m_sheath_rate*sqrt( 1. + m_p.tau[1]),
                    m_temp0, m_sheath, 1.,  yp[1][0]);
        }
        else // "bohm" == m_p.sheath_bc
        {
            //exp(-phi)
            dg::blas1::transform( m_phi[0], m_temp0, dg::ExpProfX(1., 0., 1.));
            dg::blas1::pointwiseDot( sqrt( 1 + m_p.tau[1]),
                    m_sheath_coordinate, m_temp0, 0., m_temp0);
            dg::blas1::pointwiseDot( m_sheath_rate, m_temp0, m_sheath, 1.,
                    yp[1][0]);
        }
        // u_i,sh = s*sqrt(1+tau)
        dg::blas1::pointwiseDot( m_sheath_rate*sqrt(1+m_p.tau[1]),
                m_sheath, m_sheath_coordinate, 1.,  yp[1][1]);
    }
    if( m_wall_rate != 0)
    {
        for( unsigned i=0; i<2; i++)
        {
            dg::blas1::axpby( +m_wall_rate*m_nwall, m_wall, 1., yp[0][i] );
            dg::blas1::axpby( +m_wall_rate*m_uwall, m_wall, 1., yp[1][i] );
        }
    }
    // now add wall and sheath penalization "damping" term
    // pointwiseDot catches 0 coefficients
    for( unsigned i=0; i<2; i++)
    {
        dg::blas1::pointwiseDot( -m_wall_rate, m_wall, fields[0][i],
                -m_sheath_rate, m_sheath, fields[0][i], 1., yp[0][i]);
        dg::blas1::pointwiseDot( -m_wall_rate, m_wall, fields[1][i],
                -m_sheath_rate, m_sheath, fields[1][i], 1., yp[1][i]);
    }

}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::operator()(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    std::array<std::array<Container,2>,2>& yp)
{
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    // It would probably be more practical overall not to subtract nbc
    /* y[0][0] := n_e - nbc
       y[0][1] := N_i - nbc
       y[1][0] := w_e
       y[1][1] := W_i
    */

    dg::Timer timer;
    double accu = 0.;//accumulated time
    timer.tic();

#if FELTORPERP == 1

    // set m_phi[0]
    compute_phi( t, y[0]);
    // set m_phi[1], m_dP[0], m_dP[1] and m_UE2 --- needs m_phi[0]
    compute_psi( t);

#else

#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( m_phi[0], dg::equals(), manufactured::Phie{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,t);
    dg::blas1::evaluate( m_phi[1], dg::equals(), manufactured::Phii{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,t);
#endif //DG_MANUFACTURED

#endif

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute phi and psi               took "<<timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic();

    // Transform n-nbc to n (this computes both n_e and N_i)
    dg::blas1::transform( y[0], m_fields[0], dg::PLUS<double>(m_p.nbc));

    // Compute Apar and m_U if necessary --- reads and updates m_fields[1]
    dg::blas1::copy( y[1], m_fields[1]);
    if( m_p.beta != 0)
        compute_apar( t, m_fields);

#if FELTORPERP == 1

    // Set perpendicular dynamics in yp
    compute_perp( t, y, m_fields, yp);

#else

    dg::blas1::copy( 0., yp);

#endif

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute Apar and perp dynamics    took "
              << timer.diff() << "s\t A: "<<accu<<"s\n";
    timer.tic();

    // Add parallel dynamics
#if FELTORPARALLEL == 1

    compute_parallel( t, y, m_fields, yp);

#endif
#if FELTORPERP == 1
    //-------------Add regularization----------------------------//
    for( unsigned i=0; i<2; i++)
    {
        compute_perp_diffusiveN( 1., m_fields[0][i], m_temp0, m_temp1, 1.,
                yp[0][i]);
        compute_perp_diffusiveU( 1., m_fields[1][i], m_temp0, m_temp1, 1.,
                yp[1][i]);
        if( m_p.nu_parallel_n > 0)
            compute_lapParN( m_p.nu_parallel_n, i, 1., yp[0][i], m_temp0);
    }
    //------------------Add Resistivity--------------------------//
    double eta = m_p.eta, mu0 = m_p.mu[0], mu1 = m_p.mu[1];
    dg::blas1::subroutine( [eta,mu0,mu1] DG_DEVICE (
            double ne, double ni,
            double ue, double ui, double& dtUe, double& dtUi){
                double current = (ne)*(ui-ue);
                dtUe += -eta/mu0 * current;
                dtUi += -eta/mu1 * (ne)/(ni) * current;
            },
        m_fields[0][0], m_fields[0][1],
        m_fields[1][0], m_fields[1][1], yp[1][0], yp[1][1]);
#endif

    //Add source terms
    // set m_s
    add_source_terms( t, y, m_fields, yp);

    add_rhs_penalization( yp);

    // use m_plusN and m_minusN
    add_wall_and_sheath_terms( t, m_fields, yp);


#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( yp[0][0], dg::plus_equals(), manufactured::SNe{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[0][1], dg::plus_equals(), manufactured::SNi{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[1][0], dg::plus_equals(), manufactured::SWe{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[1][1], dg::plus_equals(), manufactured::SWi{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,t);
#endif //DG_MANUFACTURED
    timer.toc();
    accu += timer.diff();
    #ifdef MPI_VERSION
        if(rank==0)
    #endif
    std::cout << "## Add parallel dynamics and sources took "<<timer.diff()<<"s\t A: "<<accu<<"\n";
}
} //namespace feltor
