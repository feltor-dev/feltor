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
struct Implicit;
template< class Geometry, class IMatrix, class Matrix, class Container >
struct ImplicitSolver;
template< class Geometry, class IMatrix, class Matrix, class Container >
struct ImplicitDensityMatrix;
template< class Geometry, class IMatrix, class Matrix, class Container >
struct ImplicitVelocityMatrix;

template< class Geometry, class IMatrix, class Matrix, class Container >
struct Explicit
{
    friend class feltor::Implicit<Geometry, IMatrix, Matrix, Container>;
    friend class feltor::ImplicitSolver<Geometry, IMatrix, Matrix, Container>;
    friend class feltor::ImplicitVelocityMatrix<Geometry, IMatrix, Matrix, Container>;
    friend class feltor::ImplicitDensityMatrix<Geometry, IMatrix, Matrix, Container>;
    using vector = std::array<std::array<Container,2>,2>;
    using container = Container;
    Explicit( const Geometry& g, feltor::Parameters p,
        dg::geo::TokamakMagneticField mag, dg::file::WrappedJsonValue js );

    //Given N_i initialize n_e such that phi=0
    void initializene( const Container& ni, Container& ne, std::string initphi);
    //Given n_e initialize N_i such that phi=0
    void initializeni( const Container& ne, Container& ni, std::string initphi);

    void operator()( double t,
        const std::array<std::array<Container,2>,2>& y,
        std::array<std::array<Container,2>,2>& yp);
    void implicit( double t,
        const std::array<std::array<Container,2>,2>& y,
        std::array<std::array<Container,2>,2>& yp);
    void add_implicit_density( double t,
        const std::array<Container,2>& density,
        double beta,
        std::array<Container,2>& yp);
    template<size_t N>
    void add_implicit_velocityST( double t,
        const std::array<Container,2>& densityST,
        const std::array<Container,2>& velocityST,
        double beta,
        std::array<Container,N>& yp);
    /// ///////////////////RESTART    MEMBERS //////////////////////
    const Container& restart_density(int i)const{
        return m_density[i];
    }
    const Container& restart_velocity(int i)const{
        return m_velocityST[i];
    }
    const Container& restart_aparallel() const {
        return m_aparST;
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
        return m_density[i];
    }
    const Container& density_source(int i)const{
        return m_s[0][i];
    }
    const Container& velocity(int i)const{
        return m_velocity[i];
    }
    const Container& velocity_source(int i){
        update_diag();
        return m_s[1][i];
    }
    const Container& potential(int i) const {
        return m_potential[i];
    }
    const Container& aparallel() const {
        return m_apar;
    }
    const std::array<Container, 3> & gradN (int i) {
        update_diag();
        return m_dFN[i];
    }
    const std::array<Container, 3> & gradU (int i) {
        update_diag();
        return m_dFU[i];
    }
    const std::array<Container, 3> & gradP (int i) {
        update_diag();
        return m_dP[i];
    }
    const std::array<Container, 3> & gradA () {
        update_diag();
        return m_dA;
    }
    const Container& divNUb( int i) const{
        return m_divNUb[i];
    }
    const Container& dsN (int i) {
        update_diag();
        return m_dsN[i];
    }
    const Container& dsU (int i) {
        update_diag();
        return m_dsU[i];
    }
    const Container& dsP(int i) {
        update_diag();
        return m_dsP[i];
    }
    const Container& dssU(int i){
        update_diag();
        return m_dssU[i];
    }
    const Container& lapParU( unsigned i) {
        update_diag();
        return m_lapParU[i];
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
    void compute_lapMperpP (int i, Container& result)
    {
        m_lapperpP.set_chi( 1.);
        dg::blas2::gemv( m_lapperpP, m_potential[i], result);
    }
    void compute_lapMperpA ( Container& result)
    {
        dg::blas2::gemv( m_lapperpU, m_apar, result);
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
            const Container& density,
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
            dg::blas1::pointwiseDivide( -alpha*m_p.nu_perp_u, temp0, density, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
    }

    unsigned called() const { return m_called;}

    /// //////////////////////DIAGNOSTICS END////////////////////////////////
    void update_diag(){
        // assume m_density, m_potential, m_velocity, m_velocityST, m_apar
        // compute dsN, dsU, dsP, lapParU, dssU and perp derivatives
        if( !m_upToDate)
        {
            // update m_dN, m_dU, m_dP, m_dA
            update_perp_derivatives( m_density, m_velocity, m_potential, m_apar);
            for( unsigned i=0; i<2; i++)
            {
                // density m_dsN
                m_fa( dg::geo::einsMinus, m_density[i], m_minus);
                m_fa( dg::geo::einsPlus,  m_density[i], m_plus);
                update_parallel_bc_2nd( m_fa, m_minus, m_density[i], m_plus,
                        m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
                dg::geo::ds_centered( m_fa, 1., m_minus, m_plus, 0., m_dsN[i]);
                // potential m_dsP
                m_fa( dg::geo::einsMinus, m_potential[i], m_minus);
                m_fa( dg::geo::einsPlus,  m_potential[i], m_plus);
                update_parallel_bc_2nd( m_fa, m_minus, m_potential[i], m_plus,
                        m_p.bcxP, 0.);
                dg::geo::ds_centered( m_fa, 1., m_minus, m_plus, 0., m_dsP[i]);
                // velocity m_dssU, m_lapParU
                m_fa_diff( dg::geo::einsMinus, m_velocity[i], m_minus);
                m_fa_diff( dg::geo::einsPlus,  m_velocity[i], m_plus);
                update_parallel_bc_2nd( m_fa_diff, m_minus, m_velocity[i], m_plus,
                        m_p.bcxU, 0.);
                dg::geo::dssd_centered( m_fa_diff, 1.,
                        m_minus, m_velocity[i], m_plus, 0., m_lapParU[i]);
                dg::geo::dss_centered( m_fa_diff, 1., m_minus,
                    m_velocity[i], m_plus, 0., m_dssU[i]);
                // velocity m_dsU
                dg::geo::ds_centered( m_fa_diff, 1., m_minus, m_plus, 0.,
                        m_dsU[i]);
                // velocity source
                dg::blas1::evaluate( m_s[1][i], dg::equals(), []DG_DEVICE(
                            double sn, double u, double n){ return -u*sn/n;},
                        m_s[0][i], m_velocity[i], m_density[i]);
            }
            m_upToDate = true;
        }

    }
    void update_parallel_bc_1st( Container& minusST, Container& plusST,
            dg::bc bcx, double value)
    {
        if( m_p.fci_bc == "along_field")
            dg::geo::assign_bc_along_field_1st( m_faST, minusST, plusST,
                    minusST, plusST, bcx, {value,value});
        else
        {
            if( bcx == dg::DIR)
            {
                dg::blas1::plus( minusST, -value);
                dg::geo::swap_bc_perp( m_fa, minusST, plusST,
                        minusST, plusST);
                dg::blas1::plus( minusST, +value);
            }
        }
    }
    void update_parallel_bc_2nd( const dg::geo::Fieldaligned<Geometry, IMatrix,
            Container>& fa, Container& minus, const Container& value0,
            Container& plus, dg::bc bcx, double value)
    {
        if( m_p.fci_bc == "along_field")
        {
            dg::geo::assign_bc_along_field_2nd( fa, minus, value0,
                    plus, minus, plus, bcx, {value,value});
        }
        else
        {
            if( bcx == dg::DIR)
            {
                dg::blas1::plus( minus, -value);
                dg::geo::swap_bc_perp( fa, minus, plus,
                        minus, plus);
                dg::blas1::plus( minus, +value);
            }
        }
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
    void compute_aparST( double t, const std::array<Container,2>&,
            std::array<Container,2>&, Container&, bool);
    void compute_phi( double t, const std::array<Container,2>&, Container&, bool);
    void compute_psi( double t, const Container& phi, Container& psi, bool);
    void update_staggered_density_and_phi( double t,
        const std::array<Container,2>& density,
        const std::array<Container,2>& potential);
    void update_staggered_density_and_ampere( double t,
        const std::array<Container,2>& density);
    void update_velocity_and_apar( double t,
        const std::array<Container,2>& velocityST,
        const Container& aparST);

    void update_perp_derivatives(
        const std::array<Container,2>& density,
        const std::array<Container,2>& velocity,
        const std::array<Container,2>& potential,
        const Container& apar);
    void compute_perp_density( double t,
        const std::array<Container,2>& density,
        const std::array<Container,2>& velocity,
        const std::array<Container,2>& potential,
        const Container& apar,
        std::array<Container,2>& densityDOT);
    void compute_perp_velocity( double t,
        const std::array<Container,2>& density,
        const std::array<Container,2>& velocity,
        const std::array<Container,2>& potential,
        const Container& apar,
        std::array<Container,2>& velocityDOT);
    void compute_parallel_flux(
             const Container& velocityKM,
             const Container& velocityKP,
             const Container& densityM,
             const Container& density,
             const Container& densityP,
             Container& fluxM,
             Container& fluxP,
             std::string slope_limiter);
    void compute_parallel_advection(
             const Container& velocityKM,
             const Container& velocityKP,
             const Container& densityM,
             const Container& density,
             const Container& densityP,
             Container& fluxM,
             Container& fluxP,
             std::string slope_limiter);
    void compute_parallel_flux(
        const Container& velocity,
        const Container& minusST,
        const Container& plusST,
        Container& flux,
        std::string slope_limiter);
    void compute_parallel_advection(
        const Container& velocity,
        const Container& minusST,
        const Container& plusST,
        Container& flux,
        std::string slope_limiter);
    void compute_parallel(          std::array<std::array<Container,2>,2>& yp);
    void multiply_rhs_penalization(      Container& yp);
    void add_wall_and_sheath_terms( std::array<std::array<Container,2>,2>& yp);
    void add_source_terms(          std::array<std::array<Container,2>,2>& yp);
    const dg::geo::Fieldaligned<Geometry, IMatrix, Container>& fieldaligned() const
    {
        return m_fa;
    }
  private:
    void construct_mag( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);
    void construct_bhat( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);
    void construct_invert( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);

#ifdef DG_MANUFACTURED
    Container m_R, m_Z, m_P, m_PST; //coordinates
#endif //DG_MANUFACTURED

    //these should be considered const // m_curv is full curvature
    std::array<Container,3> m_curv, m_curvKappa, m_b; //m_b is bhat/ sqrt(g) / B
    Container m_divCurvKappa;
    Container m_bphi, m_binv, m_divb;
    Container m_source, m_profne, m_sheath_coordinate;
    Container m_wall, m_sheath;

    // Only set once every call to operator()
    std::array<Container,2> m_density, m_densityST;
    std::array<Container,2> m_velocity, m_velocityST;
    std::array<Container,2> m_potential, m_potentialST;
    Container m_apar, m_aparST;

    Container m_UE2;
    std::array<Container,2> m_divNUb;
    std::array<Container,2> m_plusN, m_minusN, m_plusU, m_minusU;
    std::array<Container,2> m_plusSTN, m_minusSTN, m_plusSTU, m_minusSTU;
    std::vector<Container> m_multi_chi;

    // overwritten by diag_update and set once by operator()
    std::array<Container,3> m_dA;
    std::array<std::array<Container,3>,2> m_dP, m_dFN, m_dBN, m_dFU, m_dBU;
    std::array<Container,2> m_dsN, m_dsP, m_dsU;
    std::array<std::array<Container,2>,2> m_s;

    // Set by diag_update
    std::array<Container,2> m_dssU, m_lapParU;

    // Helper variables
    Container m_temp0, m_temp1;
    Container m_minus, m_plus;
    Container m_fluxM, m_fluxP;

    //matrices and solvers
    Matrix m_dxF_N, m_dxB_N, m_dxF_U, m_dxB_U, m_dx_P, m_dx_A;
    Matrix m_dyF_N, m_dyB_N, m_dyF_U, m_dyB_U, m_dy_P, m_dy_A, m_dz;
    dg::geo::Fieldaligned<Geometry, IMatrix, Container> m_fa, m_faST, m_fa_diff;
    dg::Elliptic3d< Geometry, Matrix, Container> m_lapperpN, m_lapperpU, m_lapperpP;
    std::vector<dg::Elliptic3d< Geometry, Matrix, Container> > m_multi_pol;
    std::vector<dg::Helmholtz3d<Geometry, Matrix, Container> > m_multi_invgammaP,
        m_multi_invgammaN, m_multi_ampere;

    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    dg::Extrapolation<Container> m_old_phi, m_old_psi, m_old_gammaN, m_old_apar;
    dg::Extrapolation<Container> m_old_phiST, m_old_psiST, m_old_gammaNST, m_old_aparST;

    dg::SparseTensor<Container> m_hh;

    const feltor::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    double m_source_rate = 0., m_sheath_rate = 0., m_wall_rate = 0.;
    double m_minne = 0., m_minrate  = 0., m_minalpha = 0.;
    double m_nwall = 0., m_uwall = 0.;
    bool m_fixed_profile = true, m_reversed_field = false;
    bool m_upToDate = false;
    unsigned m_called = 0;

};

template<class Grid, class IMatrix, class Matrix, class Container>
void Explicit<Grid, IMatrix, Matrix, Container>::construct_mag(
    const Grid& g, feltor::Parameters p, dg::geo::TokamakMagneticField mag
    )
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
        m_fa.construct( bhat, g, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
            p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz, p.interpolation_method);
        m_faST.construct( bhat, g, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
            p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz/2., p.interpolation_method );
        if( !(p.interpolation_method == "dg"))
            m_fa_diff.construct( bhat, g, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
                p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz, "dg");
        else
            m_fa_diff = m_fa;
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
    feltor::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue js
    ):
#ifdef DG_MANUFACTURED
    m_R( dg::pullback( dg::cooX3d, g)),
    m_Z( dg::pullback( dg::cooY3d, g)),
    m_P( dg::pullback( dg::cooZ3d, g)),
    m_PST( dg::pullback( dg::cooZ3d, g)),
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
    m_old_phiST( 2, dg::evaluate( dg::zero, g)),
    m_old_psiST( m_old_phi), m_old_gammaNST( m_old_phi), m_old_aparST( m_old_phi),
    m_p(p), m_js(js)
{
#ifdef DG_MANUFACTURED
    dg::blas1::plus(m_PST, g.hz()/2.);
#endif //DG_MANUFACTURED
    //--------------------------init vectors to 0-----------------//
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_source = m_sheath_coordinate = m_UE2 = m_temp1 = m_temp0;
    m_apar = m_aparST = m_profne = m_wall = m_sheath = m_temp0;
    m_plus = m_minus = m_temp0;
    m_fluxM = m_fluxP = m_temp0;

    m_potential[0] = m_potential[1] = m_temp0;
    m_plusSTN = m_minusSTN = m_minusSTU = m_plusSTU = m_potential;
    m_plusN = m_minusN = m_minusU = m_plusU = m_potential;
    m_divNUb = m_density = m_densityST = m_velocity = m_potential;
    m_velocityST = m_potentialST = m_potential;
    m_dsN = m_dsP = m_dsU = m_dssU = m_lapParU = m_potential;

    m_dA[0] = m_dA[1] = m_dA[2] = m_temp0;
    m_dP[0] = m_dP[1] = m_dA;
    m_dFN = m_dBN = m_dFU = m_dBU = m_dP;
    m_s[0] = m_s[1] = m_potential ;

    //--------------------------Construct-------------------------//
    construct_mag( g, p, mag);
    construct_bhat( g, p, mag);
    construct_invert( g, p, mag);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::initializene(
    const Container& src, Container& target, std::string initphi)
{
    // ne  = Ni
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        if( initphi == "zero")
        {
            // ne-nbc = Gamma (ni-nbc)
            dg::blas1::transform(src, m_temp0, dg::PLUS<double>(-m_p.nbc));
            dg::blas1::plus(target, -m_p.nbc);
            std::vector<unsigned> number = m_multigrid.direct_solve(
                m_multi_invgammaN, target, m_temp0, m_p.eps_gamma);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma);
            dg::blas1::plus(target, +m_p.nbc);
        }
        else if( initphi == "balance")
        {
            //add FLR correction -0.5*tau*mu*Delta n_e
            dg::blas1::transform(src, m_temp0, dg::PLUS<double>(-m_p.nbc));
            dg::blas2::symv( 0.5*m_p.tau[1]*m_p.mu[1],
                m_lapperpN, m_temp0, 1.0, target);
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
    // Ni = ne
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        if( initphi == "zero")
        {
            //add FLR correction -0.5*tau*mu*Delta n_e
            dg::blas1::transform(src, m_temp0, dg::PLUS<double>(-m_p.nbc));
            dg::blas2::symv( 0.5*m_p.tau[1]*m_p.mu[1],
                m_lapperpN, m_temp0, 1.0, target);
            //wird stark negativ falls alpha klein!!
        }
        else if( initphi == "balance")
        {
            //add FLR correction +0.5*tau*mu*Delta n_e
            dg::blas1::transform(src, m_temp0, dg::PLUS<double>(-m_p.nbc));
            dg::blas2::symv( -0.5*m_p.tau[1]*m_p.mu[1],
                m_lapperpN, m_temp0, 1.0, target);
            //wird stark negativ falls alpha klein!!
        }
        else if( !(initphi == "zero_pol"))
        {
            throw dg::Error(dg::Message(_ping_)<<"Warning! initphi value '"<<initphi<<"' not recognized. I have tau = "<<m_p.tau[1]<<" ! I don't know what to do! I exit!\n");
        }
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_phi(
    double time, const std::array<Container,2>& density,
    Container& phi, bool staggered
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
        if( staggered)
            m_old_gammaNST.extrapolate( time, m_temp0);
        else
            m_old_gammaN.extrapolate( time, m_temp0);
#ifdef DG_MANUFACTURED
        dg::blas1::evaluate( m_temp1, dg::plus_equals(), manufactured::SGammaNi{
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
            m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
            m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
        std::vector<unsigned> numberG = m_multigrid.direct_solve(
            m_multi_invgammaN, m_temp0, m_temp1, m_p.eps_gamma);
        if( staggered)
            m_old_gammaNST.update( time, m_temp0); // store N - nbc
        else
            m_old_gammaN.update( time, m_temp0); // store N - nbc
        if(  numberG[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
        dg::blas1::transform( density[0], m_temp1, dg::PLUS<double>(-m_p.nbc));
        dg::blas1::axpby( -1., m_temp1, 1., m_temp0, m_temp0);
    }
#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SPhie{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
        m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
    //----------Invert polarisation----------------------------//
    if( staggered)
        m_old_phiST.extrapolate( time, phi);
    else
        m_old_phi.extrapolate( time, phi);
    std::vector<unsigned> number = m_multigrid.direct_solve(
        m_multi_pol, phi, m_temp0, m_p.eps_pol);
    if( staggered)
        m_old_phiST.update( time, phi);
    else
        m_old_phi.update( time, phi);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol[0]);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_psi(
    double time, const Container& phi, Container& psi, bool staggered)
{
    //-----------Solve for Gamma Phi---------------------------//
    if (m_p.tau[1] == 0.) {
        dg::blas1::copy( phi, psi);
    } else {
        if( staggered)
            m_old_psiST.extrapolate( time, psi);
        else
            m_old_psi.extrapolate( time, psi);
#ifdef DG_MANUFACTURED
        dg::blas1::copy( phi, m_temp0);
        dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SGammaPhie{
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
            m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
        std::vector<unsigned> number = m_multigrid.direct_solve(
            m_multi_invgammaP, psi, m_temp0, m_p.eps_gamma);
#else
        std::vector<unsigned> number = m_multigrid.direct_solve(
            m_multi_invgammaP, psi, phi, m_p.eps_gamma);
#endif //DG_MANUFACTURED
        if( staggered)
            m_old_psiST.update( time, psi);
        else
            m_old_psi.update( time, psi);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
    }
    //-------Compute Psi and derivatives
    dg::blas2::symv( m_dx_P, phi, m_dP[0][0]);
    dg::blas2::symv( m_dy_P, phi, m_dP[0][1]);
    if( !m_p.symmetric) dg::blas2::symv( m_dz, phi, m_dP[0][2]);
    if( staggered)
        dg::tensor::scalar_product3d( 1., m_binv,
            m_dP[0][0], m_dP[0][1], m_dP[0][2], m_hh, m_binv, //grad_perp
            m_dP[0][0], m_dP[0][1], m_dP[0][2], 1., psi);
    else
    {
        dg::tensor::scalar_product3d( 1., m_binv,
            m_dP[0][0], m_dP[0][1], m_dP[0][2], m_hh, m_binv, //grad_perp
            m_dP[0][0], m_dP[0][1], m_dP[0][2], 0., m_UE2);
        //m_UE2 now contains u_E^2
        dg::blas1::axpby( -0.5, m_UE2, 1., psi);
    }
#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( psi, dg::plus_equals(), manufactured::SPhii{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_aparST(
    double time, const std::array<Container,2>& densityST,
    std::array<Container,2>& velocityST, Container& aparST, bool update)
{
    //on input
    //densityST[0] = n_e, velocityST[0]:= w_e
    //densityST[1] = N_i, velocityST[1]:= W_i

    //----------Compute right hand side------------------------//
    dg::blas1::pointwiseDot(  m_p.beta, densityST[1], velocityST[1],
                             -m_p.beta, densityST[0], velocityST[0],
                              0., m_temp0);
    //----------Invert Induction Eq----------------------------//
    if( update)
        m_old_aparST.extrapolate( time, aparST);
    std::vector<unsigned> number = m_multigrid.direct_solve(
        m_multi_ampere, aparST, m_temp0, m_p.eps_ampere);
    if( update)
        m_old_aparST.update( time, aparST);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_ampere);
#ifdef DG_MANUFACTURED
    //dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SA{
    //    m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
    //    m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},m_R,m_Z,m_P,time);
    //here we cheat (a bit)
    dg::blas1::evaluate( aparST, dg::equals(), manufactured::A{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
        m_R,m_Z,m_PST,time);
#endif //DG_MANUFACTURED
    //----------Compute Velocities-----------------------------//
    dg::blas1::axpby( 1., velocityST[0], -1./m_p.mu[0], aparST, velocityST[0]);
    dg::blas1::axpby( 1., velocityST[1], -1./m_p.mu[1], aparST, velocityST[1]);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::update_perp_derivatives(
    const std::array<Container,2>& density,
    const std::array<Container,2>& velocity,
    const std::array<Container,2>& potential,
    const Container& apar)
{
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        //First compute forward and backward derivatives for upwind scheme
        dg::blas1::transform( density[i], m_temp1, dg::PLUS<double>(-m_p.nbc));
        dg::blas2::symv( m_dxF_N, m_temp1, m_dFN[i][0]);
        dg::blas2::symv( m_dyF_N, m_temp1, m_dFN[i][1]);
        dg::blas2::symv( m_dxB_N, m_temp1, m_dBN[i][0]);
        dg::blas2::symv( m_dyB_N, m_temp1, m_dBN[i][1]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, m_temp1, m_dFN[i][2]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, m_temp1, m_dBN[i][2]);
        dg::blas2::symv( m_dxF_U, velocity[i], m_dFU[i][0]);
        dg::blas2::symv( m_dyF_U, velocity[i], m_dFU[i][1]);
        dg::blas2::symv( m_dxB_U, velocity[i], m_dBU[i][0]);
        dg::blas2::symv( m_dyB_U, velocity[i], m_dBU[i][1]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, velocity[i], m_dFU[i][2]);
        if(!m_p.symmetric) dg::blas2::symv( m_dz, velocity[i], m_dBU[i][2]);
        dg::blas2::symv( m_dx_P, potential[i], m_dP[i][0]);
        dg::blas2::symv( m_dy_P, potential[i], m_dP[i][1]);
        if( !m_p.symmetric) dg::blas2::symv( m_dz, potential[i], m_dP[i][2]);
        dg::blas2::symv( m_dx_A, apar, m_dA[0]);
        dg::blas2::symv( m_dy_A, apar, m_dA[1]);
        if( !m_p.symmetric) dg::blas2::symv( m_dz, apar, m_dA[2]);
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::update_staggered_density_and_phi(
    double t,
    const std::array<Container,2>& density,
    const std::array<Container,2>& potential)
{
    for( unsigned i=0; i<2; i++)
    {

        m_fa( dg::geo::einsMinus, density[i], m_minusN[i]);
        m_fa( dg::geo::einsPlus,  density[i], m_plusN[i]);
        update_parallel_bc_2nd( m_fa, m_minusN[i], density[i], m_plusN[i],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);

        m_faST( dg::geo::zeroMinus, potential[i], m_minus);
        m_faST( dg::geo::einsPlus,  potential[i], m_plus);
        update_parallel_bc_1st( m_minus, m_plus,
                m_p.bcxP, 0.);
        dg::geo::ds_centered( m_faST, 1., m_minus, m_plus, 0., m_dsP[i]);
        dg::blas1::axpby( 0.5, m_minus, 0.5, m_plus, m_potentialST[i]);
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::update_staggered_density_and_ampere(
    double t,
    const std::array<Container,2>& density)
{
    for( unsigned i=0; i<2; i++)
    {
        m_faST( dg::geo::zeroMinus, density[i], m_minusSTN[i]);
        m_faST( dg::geo::einsPlus,  density[i], m_plusSTN[i]);
        update_parallel_bc_1st( m_minusSTN[i], m_plusSTN[i],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
        dg::blas1::axpby( 0.5, m_minusSTN[i], 0.5, m_plusSTN[i], m_densityST[i]);
    }
    //----------Compute and set chi----------------------------//
    if( m_p.beta != 0)
    {
        dg::blas1::axpby(  m_p.beta/m_p.mu[1], m_densityST[1],
                          -m_p.beta/m_p.mu[0], m_densityST[0], m_temp0);
        m_multigrid.project( m_temp0, m_multi_chi);
        for( unsigned u=0; u<m_p.stages; u++)
            m_multi_ampere[u].set_chi( m_multi_chi[u]);
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::update_velocity_and_apar(
    double t,
    const std::array<Container,2>& velocityST,
    const Container& aparST)
{
    for( unsigned i=0; i<2; i++)
    {
        // Compute dsU and velocity
        m_faST( dg::geo::einsMinus, velocityST[i], m_minusSTU[i]);
        m_faST( dg::geo::zeroPlus,  velocityST[i], m_plusSTU[i]);
        update_parallel_bc_1st( m_minusSTU[i], m_plusSTU[i], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, m_minusSTU[i], 0.5, m_plusSTU[i], m_velocity[i]);

        m_fa( dg::geo::einsMinus, velocityST[i], m_minusU[i]);
        m_fa( dg::geo::einsPlus,  velocityST[i], m_plusU[i]);
        update_parallel_bc_2nd( m_fa, m_minusU[i], velocityST[i],
                m_plusU[i], m_p.bcxU, 0.);
    }
    // Compute apar
    m_faST( dg::geo::einsMinus, aparST, m_minus);
    m_faST( dg::geo::zeroPlus,  aparST, m_plus);
    update_parallel_bc_1st( m_minus, m_plus, m_p.bcxA, 0.);
    dg::blas1::axpby( 0.5, m_minus, 0.5, m_plus, m_apar);
    m_old_apar.update( t, m_apar);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_perp_density(
    double t,
    const std::array<Container,2>& density,
    const std::array<Container,2>& velocity,
    const std::array<Container,2>& potential,
    const Container& apar,
    std::array<Container,2>& densityDOT)
{
    update_perp_derivatives( density, velocity, potential, apar);
    //y[0] = N, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
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
                double& dtN
            )
            {
                dtN = 0;
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

                // use centered derivatives
                double KappaU = ( curvKappa0*(d0FU+d0BU)+curvKappa1*(d1FU+d1BU)
                        +curvKappa2*(d2FU+d2BU) ) / 2.;
                double KP = curv0*d0P+curv1*d1P+curv2*d2P;

                dtN +=  - N * ( KP + mu * U * U * divCurvKappa
                                + 2. * mu * U * KappaU);
                if( beta != 0)
                {
                    double divbp = A*divCurvKappa
                                     - (curv0-curvKappa0)*d0A
                                     - (curv1-curvKappa1)*d1A
                                     - (curv2-curvKappa2)*d2A;
                    double bpU = bp0*( d0FU + d0BU) / 2. +
                                 bp1*( d1FU + d1BU) / 2. +
                                 bp2*( d2FU + d2BU) / 2.;
                    dtN +=  -N*( U*divbp + bpU);
                }
                return;
            },
            //species depdendent
            density[i],   m_dFN[i][0], m_dFN[i][1], m_dFN[i][2],
                          m_dBN[i][0], m_dBN[i][1], m_dBN[i][2],
            velocity[i],  m_dFU[i][0], m_dFU[i][1], m_dFU[i][2],
                          m_dBU[i][0], m_dBU[i][1], m_dBU[i][2],
                          m_dP[i][0], m_dP[i][1], m_dP[i][2],
            //aparallel
            apar, m_dA[0], m_dA[1], m_dA[2],
            //magnetic parameters
            m_b[0], m_b[1], m_b[2],
            m_curv[0], m_curv[1], m_curv[2],
            m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
            m_divCurvKappa, densityDOT[i]
        );
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_perp_velocity(
    double t,
    const std::array<Container,2>& density,
    const std::array<Container,2>& velocity,
    const std::array<Container,2>& potential,
    const Container& apar,
    std::array<Container,2>& velocityDOT)
{
    update_perp_derivatives( density, velocity, potential, apar);
    //y[0] = N, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
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
                double& dtU
            )
            {
                dtU = 0;
                // velocity - upwind scheme
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
                // velocity - upwind scheme
                v0 += 2.*tau*curvKappa0;
                v1 += 2.*tau*curvKappa1;
                v2 += 2.*tau*curvKappa2;
                dtU += ( v0 > 0 ) ? -v0*d0BU : -v0*d0FU;
                dtU += ( v1 > 0 ) ? -v1*d1BU : -v1*d1FU;
                dtU += ( v2 > 0 ) ? -v2*d2BU : -v2*d2FU;

                // use centered derivatives
                double KappaN = ( curvKappa0*(d0FN+d0BN)+curvKappa1*(d1FN+d1BN)
                        +curvKappa2*(d2FN+d2BN) ) / 2.;
                double KappaP = curvKappa0*d0P+curvKappa1*d1P+curvKappa2*d2P;

                dtU +=  - U * ( 2. * tau * KappaN / N + tau * divCurvKappa
                                + KappaP);
                if( beta != 0)
                {
                    double bpN = bp0*( d0FN + d0BN) / 2. +
                                 bp1*( d1FN + d1BN) / 2. +
                                 bp2*( d2FN + d2BN) / 2.;
                    double bpP = bp0 * d0P + bp1 * d1P + bp2 * d2P;
                    dtU +=  - bpP/mu - tau/mu * bpN/N;
                }
                return;
            },
            //species depdendent
            density[i],   m_dFN[i][0], m_dFN[i][1], m_dFN[i][2],
                          m_dBN[i][0], m_dBN[i][1], m_dBN[i][2],
            velocity[i],  m_dFU[i][0], m_dFU[i][1], m_dFU[i][2],
                          m_dBU[i][0], m_dBU[i][1], m_dBU[i][2],
                          m_dP[i][0], m_dP[i][1], m_dP[i][2],
            //aparallel
            apar, m_dA[0], m_dA[1], m_dA[2],
            //magnetic parameters
            m_b[0], m_b[1], m_b[2],
            m_curv[0], m_curv[1], m_curv[2],
            m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
            m_divCurvKappa, velocityDOT[i]
        );
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix,
     Container>::compute_parallel_advection(
             const Container& velocityKM,
             const Container& velocityKP,
             const Container& densityM,
             const Container& density,
             const Container& densityP,
             Container& fluxM,
             Container& fluxP,
             std::string slope_limiter
             )
{
    dg::blas1::evaluate( fluxM, dg::equals(), dg::Upwind(),
            velocityKM, densityM, density);
    dg::blas1::evaluate( fluxP, dg::equals(), dg::Upwind(),
            velocityKP, density, densityP);
    if(slope_limiter != "none" )
    {
        m_fa( dg::geo::einsMinus, densityM, m_minus);
        m_fa( dg::geo::einsPlus, densityP, m_plus);
        // Let's keep the default boundaries of NEU
        // boundary values are (probably?) never used in the slope limiter branches
        dg::blas1::copy(density, m_temp0);
        update_parallel_bc_2nd( m_fa, m_temp0, densityP, m_plus, dg::NEU, 0.);
        dg::blas1::copy(density, m_temp0);
        update_parallel_bc_2nd( m_fa, m_minus, densityM, m_temp0, dg::NEU, 0.);
        dg::blas1::axpby( 1., densityP, -1., density, m_temp0);
        dg::blas1::axpby( 1., density, -1., densityM, m_temp1);
        dg::blas1::axpby( 1., densityM, -1., densityM, m_temp1);
        if( slope_limiter == "minmod")
        {
            dg::MinMod minmod;
            dg::blas1::subroutine( [minmod] DG_DEVICE(
                        double& fluxM, double& fluxP, double vKM, double vKP,
                        double dKMM, double dKM, double dK, double dKP, double dKPP)
                    {
                        if( vKP >= 0.)
                            fluxP += 0.5*minmod( dKP-dK, dK-dKM);
                        else
                            fluxP -= 0.5*minmod( dKPP-dKP, dKP-dK);
                        if( vKM >= 0.)
                            fluxM += 0.5*minmod( dK-dKM, dKM-dKMM);
                        else
                            fluxM -= 0.5*minmod( dKP-dK, dK-dKM);

                    }, fluxM, fluxP, velocityKM, velocityKP, m_minus, densityM,
                    density, densityP, m_plus);
        }
        else if( slope_limiter == "vanLeer")
        {
            dg::VanLeer vanLeer;
            dg::blas1::subroutine( [vanLeer] DG_DEVICE(
                        double& fluxM, double& fluxP, double vKM, double vKP,
                        double dKMM, double dKM, double dK, double dKP, double dKPP)
                    {
                        if( vKP >= 0.)
                            fluxP += 0.5*vanLeer( dKP-dK, dK-dKM);
                        else
                            fluxP -= 0.5*vanLeer( dKPP-dKP, dKP-dK);
                        if( vKM >= 0.)
                            fluxM += 0.5*vanLeer( dK-dKM, dKM-dKMM);
                        else
                            fluxM -= 0.5*vanLeer( dKP-dK, dK-dKM);

                    }, fluxM, fluxP, velocityKM, velocityKP, m_minus, densityM,
                    density, densityP, m_plus);
        }
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix,
     Container>::compute_parallel_flux(
             const Container& velocityKM,
             const Container& velocityKP,
             const Container& densityM,
             const Container& density,
             const Container& densityP,
             Container& fluxM,
             Container& fluxP,
             std::string slope_limiter
             )
{
    compute_parallel_advection( velocityKM, velocityKP, densityM, density, densityP,
            fluxM, fluxP, slope_limiter);
    dg::blas1::pointwiseDot( fluxM, velocityKM, fluxM);
    dg::blas1::pointwiseDot( fluxP, velocityKP, fluxP);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix,
     Container>::compute_parallel_advection( const Container& velocity,
             const Container& minusST, const Container& plusST,
             Container& flux,
             std::string slope_limiter
             )
{
    dg::blas1::evaluate( flux, dg::equals(), dg::Upwind(),
            velocity, minusST, plusST);
    if(slope_limiter != "none" )
    {
        dg::blas1::axpby( 1., plusST, -1., minusST, m_temp0);
        m_fa( dg::geo::einsMinus, m_temp0, m_minus);
        m_fa( dg::geo::einsPlus, m_temp0, m_plus);
        // Let's keep the default boundaries of NEU
        // boundary values are (probably?) never used in the slope limiter branches
        update_parallel_bc_2nd( m_fa, m_minus, m_temp0, m_plus, dg::NEU, 0.);
        if( slope_limiter == "minmod")
        {
            dg::blas1::evaluate( flux, dg::plus_equals(),
                dg::SlopeLimiter<dg::MinMod>(), velocity,
                m_minus, m_temp0, m_plus, 0.5, 0.5);
        }
        else if( slope_limiter == "vanLeer")
        {
            dg::blas1::evaluate( flux, dg::plus_equals(),
                dg::SlopeLimiter<dg::VanLeer>(), velocity,
                m_minus, m_temp0, m_plus, 0.5, 0.5);
        }
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix,
     Container>::compute_parallel_flux( const Container& velocity,
             const Container& minusST, const Container& plusST,
             Container& flux,
             std::string slope_limiter
             )
{
    compute_parallel_advection( velocity, minusST, plusST, flux, slope_limiter);
    dg::blas1::pointwiseDot( velocity, flux, flux);
}


template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::compute_parallel(
    std::array<std::array<Container,2>,2>& yp)
{
    for( unsigned i=0; i<2; i++)
    {
        // compute qhat
        compute_parallel_flux( m_minusSTU[i], m_plusSTU[i],
                m_minusN[i], m_density[i], m_plusN[i],
                m_fluxM, m_fluxP, m_p.slope_limiter);
        // Now compute divNUb
        dg::geo::ds_divCentered( m_faST, 1., m_fluxM, m_fluxP, 0.,
                m_divNUb[i]);
        dg::blas1::axpby( -1., m_divNUb[i], 1., yp[0][i]);

        // compute grad U2/2
        dg::blas1::axpby( 0.25, m_minusU[i], 0.25, m_velocityST[i], m_minusSTU[i]);
        dg::blas1::axpby( 0.25, m_velocityST[i], 0.25, m_plusU[i], m_plusSTU[i]);
        compute_parallel_flux( m_minusSTU[i], m_plusSTU[i],
                m_minusU[i], m_velocityST[i], m_plusU[i],
                m_fluxM, m_fluxP,
                m_p.slope_limiter);
        dg::geo::ds_centered( m_faST, -1., m_fluxM, m_fluxP, 1., yp[1][i]);

        // Add density gradient and electric field
        double tau = m_p.tau[i], mu = m_p.mu[i], delta = m_fa.deltaPhi();
        dg::blas1::subroutine( [tau, mu, delta ]DG_DEVICE ( double& WDot,
                    double dsP, double QN, double PN, double bphi)
                {
                    WDot -= 1./mu*dsP;
                    WDot -= tau/mu*bphi*(PN-QN)/delta/2.*(1/PN + 1/QN);
                },
                yp[1][i], m_dsP[i], m_minusSTN[i], m_plusSTN[i], m_fa.bphi()
        );
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_source_terms(
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
                m_s[0][0], m_density[0], m_profne, m_source, m_source_rate);
        else
            dg::blas1::axpby( m_source_rate, m_source, 0., m_s[0][0]);
    }
    else
        dg::blas1::copy( 0., m_s[0][0]);
    // add prevention to get below lower limit
    if( m_minrate != 0.0)
    {
        // do not make lower forcing a velocity source
        // MW it may be that this form does not go well with the potential
        dg::blas1::transform( m_density[0], m_temp0, dg::PolynomialHeaviside(
                    m_minne-m_minalpha/2., m_minalpha/2., -1) );
        dg::blas1::transform( m_density[0], m_temp1, dg::PLUS<double>( -m_minne));
        dg::blas1::pointwiseDot( -m_minrate, m_temp1, m_temp0, 1., yp[0][0]);
        dg::blas1::transform( m_density[1], m_temp0, dg::PolynomialHeaviside(
                    m_minne-m_minalpha/2., m_minalpha/2., -1) );
        dg::blas1::transform( m_density[1], m_temp1, dg::PLUS<double>( -m_minne));
        dg::blas1::pointwiseDot( -m_minrate, m_temp1, m_temp0, 1., yp[0][1]);
    }

    //compute FLR corrections S_N = (1-0.5*mu*tau*Lap)*S_n
    dg::blas2::gemv( m_lapperpN, m_s[0][0], m_temp0);
    dg::blas1::axpby( 1., m_s[0][0], 0.5*m_p.tau[1]*m_p.mu[1], m_temp0, m_s[0][1]);
    // potential part of FLR correction S_N += -div*(mu S_n grad*Phi/B^2)
    dg::blas1::pointwiseDot( m_p.mu[1], m_s[0][0], m_binv, m_binv, 0., m_temp0);
    m_lapperpP.set_chi( m_temp0);
    m_lapperpP.symv( 1., m_potential[0], 1., m_s[0][1]);

    // S_U = - U S_N/N
    for(int i=0; i<2; i++)
    {
        // transform to adjoint plane and add to velocity source
        m_faST( dg::geo::zeroMinus, m_s[0][i], m_minus);
        m_faST( dg::geo::einsPlus,  m_s[0][i], m_plus);
        update_parallel_bc_1st( m_minus, m_plus, m_p.bcxN, 0.);
        dg::geo::ds_average( m_faST, 1., m_minus, m_plus, 0., m_temp0);
        dg::blas1::evaluate( m_s[1][i], dg::equals(), []DG_DEVICE(
                    double sn, double u, double n){ return -u*sn/n;},
                m_temp0, m_velocityST[i], m_densityST[i]);
    }
    //Add all to the right hand side
    dg::blas1::axpby( 1., m_s, 1.0, yp);
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::multiply_rhs_penalization(
        Container& yp)
{
    //mask right hand side in penalization region
    if( m_p.penalize_wall && m_p.penalize_sheath)
    {
        dg::blas1::subroutine( []DG_DEVICE(
            double& rhs, double wall, double sheath){
                rhs *= (1.0-wall-sheath);
            }, yp, m_wall, m_sheath);
    }
    else if( m_p.penalize_wall)
    {
        dg::blas1::subroutine( []DG_DEVICE( double& rhs, double wall){
                rhs *= (1.0-wall); }, yp, m_wall);
    }
    else if( m_p.penalize_sheath)
    {
        dg::blas1::subroutine( []DG_DEVICE( double& rhs, double sheath){
                rhs *= (1.0-sheath); }, yp, m_sheath);
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_wall_and_sheath_terms(
        std::array<std::array<Container,2>,2>& yp)
{
    // add sheath boundary conditions
    if( m_sheath_rate != 0)
    {
        ////density
        ////Here, we need to find out where "downstream" is
        //!! Simulations does not really work without
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
            dg::blas1::pointwiseDot( m_sheath_rate, m_temp0, m_sheath, 1.,
                    yp[0][i]);
        }
        //compute sheath velocity
        if( "wall" == m_p.sheath_bc)
        {
            for( unsigned i=0; i<2; i++)
            {
                //dg::blas1::axpby( +m_sheath_rate*m_nwall, m_sheath, 1., yp[0][i] );
                dg::blas1::axpby( +m_sheath_rate*m_uwall, m_sheath, 1., yp[1][i] );
            }
        }
        else
        {
            //velocity c_s
            double cs = sqrt(1.+m_p.tau[1]), sheath_rate = m_sheath_rate;
            if( "insulating" == m_p.sheath_bc)
            {
                // u_e,sh = s*sqrt(1+tau) Ni/ne
                dg::blas1::evaluate( yp[1][0], dg::plus_equals(),
                        [cs, sheath_rate]DG_DEVICE( double sheath_coord, double
                            sheath, double ne, double ni) {
                            return cs*sheath_rate*sheath_coord*ni/ne*sheath;
                        },
                        m_sheath_coordinate, m_sheath, m_densityST[0],
                        m_densityST[1]);
            }
            else // "bohm" == m_p.sheath_bc
            {
                //u_e,sh = s*1/sqrt(|mu_e|2pi) exp(-phi)
                double mue = fabs(m_p.mu[0]);
                dg::blas1::evaluate( yp[1][0], dg::plus_equals(),
                    [mue, sheath_rate]DG_DEVICE( double sheath_coord, double
                        sheath, double phi) {
                        return sheath_rate*sheath_coord*sheath*exp(-phi)/sqrt( mue*2.*M_PI);
                    },
                    m_sheath_coordinate, m_sheath, m_potentialST[0]);
            }
            // u_i,sh = s*sqrt(1+tau)
            dg::blas1::pointwiseDot( sheath_rate*cs,
                    m_sheath, m_sheath_coordinate, 1.,  yp[1][1]);
        }
    }
    // add wall boundary conditions
    if( m_wall_rate != 0)
    {
        for( unsigned i=0; i<2; i++)
        {
            dg::blas1::axpby( +m_wall_rate*m_nwall, m_wall, 1., yp[0][i] );
            dg::blas1::axpby( +m_wall_rate*m_uwall, m_wall, 1., yp[1][i] );
        }
    }
}

#ifndef WITH_NAVIER_STOKES
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::operator()(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    std::array<std::array<Container,2>,2>& yp)
{
    m_called++;
    m_upToDate = false;
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    /* y[0][0] := n_e
       y[0][1] := N_i
       y[1][0] := w_e^dagger
       y[1][1] := W_i^dagger
    */

    dg::Timer timer;
    double accu = 0.;//accumulated time
    timer.tic();

    dg::blas1::copy( y[0], m_density),

#if FELTORPERP == 1

    // set m_potential[0]
    compute_phi( t, m_density, m_potential[0], false);
    // set m_potential[1] and m_UE2 --- needs m_potential[0]
    compute_psi( t, m_potential[0], m_potential[1], false);

#else

#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( m_potential[0], dg::equals(), manufactured::Phie{
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
            m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
            m_R,m_Z,m_P,t);
    dg::blas1::evaluate( m_potential[1], dg::equals(), manufactured::Phii{
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
            m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
            m_R,m_Z,m_P,t);
#endif //DG_MANUFACTURED

#endif

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute phi and psi               took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );

    //Compute m_densityST and m_potentialST
    update_staggered_density_and_phi( t, m_density, m_potential);
    update_staggered_density_and_ampere( t, m_density);

    //// Now refine potential on staggered grid
    //// set m_potentialST[0]
    //compute_phi( t, m_densityST, m_potentialST[0], true);
    //// set m_potentialST[1]  --- needs m_potentialST[0]
    //compute_psi( t, m_potentialST[0], m_potentialST[1], true);
    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute phi and psi ST            took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );

    // Compute m_aparST and m_velocityST if necessary
    dg::blas1::copy( y[1], m_velocityST);
    if( m_p.beta != 0)
    {
        compute_aparST( t, m_densityST, m_velocityST, m_aparST, true);
    }
    //Compute m_velocity and m_apar
    update_velocity_and_apar( t, m_velocityST, m_aparST);

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute Apar and staggered        took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );

#if FELTORPERP == 1

    // Set perpendicular dynamics in yp
    compute_perp_density(  t, m_density, m_velocity, m_potential, m_apar,
            yp[0]);
    compute_perp_velocity( t, m_densityST, m_velocityST, m_potentialST,
            m_aparST, yp[1]);

#else

    dg::blas1::copy( 0., yp);

#endif

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute perp dynamics             took "
                       << timer.diff() << "s\t A: "<<accu<<"s\n";
    timer.tic();

    // Add parallel dynamics
#if FELTORPARALLEL == 1

    compute_parallel( yp);

#endif
#if FELTORPERP == 1
    //------------------Add Resistivity--------------------------//
    double eta = m_p.eta, mu0 = m_p.mu[0], mu1 = m_p.mu[1];
    dg::blas1::subroutine( [eta,mu0,mu1] DG_DEVICE (
            double ne, double ni,
            double ue, double ui, double& dtUe, double& dtUi){
                double current = ni*ui-ne*ue;
                dtUe += -eta/mu0 * current;
                dtUi += -eta/mu1 * ne/ni * current;
            },
        m_densityST[0], m_densityST[1],
        m_velocityST[0], m_velocityST[1], yp[1][0], yp[1][1]);
#endif

    if( !m_p.partitioned)
    {
        // explicit and implicit timestepper
        add_implicit_density( t, m_density, 1., yp[0]);
        add_implicit_velocityST( t, m_densityST, m_velocityST, 1., yp[1]);
    }
    else
    {
        // partitioned means imex timestepper
        for( unsigned i=0; i<2; i++)
        {
            for( unsigned j=0; j<2; j++)
                multiply_rhs_penalization( yp[i][j]); // F*(1-chi_w-chi_s)
        }
    }

    add_wall_and_sheath_terms( yp);
    //Add source terms
    // set m_s
    add_source_terms( yp );

#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( yp[0][0], dg::plus_equals(), manufactured::SNe{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
        m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[0][1], dg::plus_equals(), manufactured::SNi{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
        m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[1][0], dg::plus_equals(), manufactured::SWe{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
        m_R,m_Z,m_PST,t);
    dg::blas1::evaluate( yp[1][1], dg::plus_equals(), manufactured::SWi{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.eta,
        m_p.beta,m_p.nu_perp_n,m_p.nu_parallel_u[0],m_p.nu_parallel_u[1]},
        m_R,m_Z,m_PST,t);
#endif //DG_MANUFACTURED
    timer.toc();
    accu += timer.diff();
    #ifdef MPI_VERSION
        if(rank==0)
    #endif
    std::cout << "## Add parallel dynamics and sources took "<<timer.diff()
              << "s\t A: "<<accu<<"\n";
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_implicit_density(
    double t,
    const std::array<Container,2>& density,
    double beta,
    std::array<Container,2>& yp)
{
#if FELTORPERP == 1
    for( unsigned i=0; i<2; i++)
        compute_perp_diffusiveN( 1., density[i], m_temp0,
                m_temp1, beta, yp[i]);
#else
    dg::blas1::scal( yp, beta);
#endif
    for( unsigned i=0; i<2; i++)
    {
        multiply_rhs_penalization( yp[i]); // F*(1-chi_w-chi_s)
        dg::blas1::pointwiseDot( -m_wall_rate, m_wall, density[i],
            -m_sheath_rate, m_sheath, density[i], 1., yp[i]); // -r N
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
template<size_t N>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_implicit_velocityST(
    double t,
    const std::array<Container,2>& densityST,
    const std::array<Container,2>& velocityST,
    double beta,
    std::array<Container,N>& yp)
{
    // velocityST[0] := u_e^dagger
    // velocityST[1] := U_i^dagger
#if FELTORPERP == 1
    for( unsigned i=0; i<2; i++)
    {
        compute_perp_diffusiveU( 1., velocityST[i], densityST[i], m_temp0,
                m_temp1, beta, yp[i]);
    }
#else
    dg::blas1::scal( yp, beta);
#endif
#if FELTORPARALLEL == 1
    for( unsigned i=0; i<2; i++)
    {
        // Add parallel viscosity
        if( m_p.nu_parallel_u[i] > 0)
        {
            m_fa_diff( dg::geo::einsMinus, velocityST[i], m_minus);
            m_fa_diff( dg::geo::einsPlus, velocityST[i], m_plus);
            update_parallel_bc_2nd( m_fa_diff, m_minus, velocityST[i],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa_diff, m_p.nu_parallel_u[i],
                    m_minus, velocityST[i], m_plus, 0., m_temp0);
            dg::blas1::pointwiseDivide( 1., m_temp0, densityST[i], 1., yp[i]);
        }
    }
#endif
    for( unsigned i=0; i<2; i++)
    {
        multiply_rhs_penalization( yp[i]); // F*(1-chi_w-chi_s)
        dg::blas1::pointwiseDot( -m_wall_rate, m_wall, velocityST[i],
            -m_sheath_rate, m_sheath, velocityST[i], 1., yp[i]); // -r U
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
struct Implicit
{
    Implicit() {}
    Implicit( Explicit<Geometry,IMatrix,Matrix,Container>& ex) :
        m_ex(&ex) {}

    void operator()( double t, const std::array<std::array<Container,2>,2>& y,
        std::array<std::array<Container,2>,2>& yp)
    {
        // y[0][0] := n_e
        // y[0][1] := N_i
        m_ex->add_implicit_density( t, y[0], 0., yp[0]);
        // y[1][0] := w_e^dagger
        // y[1][1] := W_i^dagger
        m_ex->update_staggered_density_and_ampere( t, y[0]);
        dg::blas1::copy( y[1], m_ex->m_velocityST);
        if( m_ex->m_p.beta != 0)
            m_ex->compute_aparST( t, m_ex->m_densityST, m_ex->m_velocityST, m_ex->m_aparST, true);
        m_ex->add_implicit_velocityST( t, m_ex->m_densityST, m_ex->m_velocityST, 0., yp[1]);
    }
    private:
    Explicit<Geometry,IMatrix,Matrix,Container>* m_ex; // does not own anything
};

}//namespace feltor
namespace dg
{
template< class Geometry, class IMatrix, class Matrix, class Container >
struct TensorTraits<feltor::ImplicitDensityMatrix< Geometry, IMatrix, Matrix, Container >>
{
    using value_type      = get_value_type<Container>;
    using tensor_category = SelfMadeMatrixTag;
};
template< class Geometry, class IMatrix, class Matrix, class Container >
struct TensorTraits<feltor::ImplicitVelocityMatrix< Geometry, IMatrix, Matrix, Container >>
{
    using value_type      = get_value_type<Container>;
    using tensor_category = SelfMadeMatrixTag;
};
}//namespace dg

namespace feltor{

template< class Geometry, class IMatrix, class Matrix, class Container >
struct ImplicitDensityMatrix
{
    ImplicitDensityMatrix() {}
    ImplicitDensityMatrix( Explicit<Geometry,IMatrix,Matrix,Container>& ex) : m_ex(&ex){}
    void symv ( const std::array<Container,2> & density,
            std::array<Container,2>& yp)
    {
        if( m_alpha != 0)
            m_ex->add_implicit_density( m_time, density, 0., yp);
        dg::blas1::axpby( 1., density, m_alpha, yp);
        dg::blas2::symv( m_ex->m_lapperpN.weights(), yp, yp);
    }
    void set_params( double alpha, double time){
        m_alpha = alpha;
        m_time = time;
    }
    private:
    Explicit<Geometry,IMatrix,Matrix,Container>* m_ex; // does not own anything
    double m_alpha=0., m_time=0.;
};
template< class Geometry, class IMatrix, class Matrix, class Container >
struct ImplicitVelocityMatrix
{
    ImplicitVelocityMatrix() {}
    ImplicitVelocityMatrix( Explicit<Geometry,IMatrix,Matrix,Container>& ex) : m_ex(&ex){}
    void symv( const std::array<Container,3>& w, std::array<Container,3>& wp)
    {
        // w[0] := w_e^dagger
        // w[1] := W_i^dagger
        if( m_alpha!=0)
        {
            if( m_ex->m_p.beta !=0)
            {
                dg::blas1::axpby( 1., w[0], -1./m_ex->m_p.mu[0], w[2], m_ex->m_velocityST[0]);
                dg::blas1::axpby( 1., w[1], -1./m_ex->m_p.mu[1], w[2], m_ex->m_velocityST[1]);
            }
            else
            {
                dg::blas1::copy( w[0], m_ex->m_velocityST[0]);
                dg::blas1::copy( w[1], m_ex->m_velocityST[1]);
            }
            m_ex->add_implicit_velocityST( m_time, m_ex->m_densityST, m_ex->m_velocityST, 0., wp);
        }
        dg::blas1::axpby( 1., w[0], m_alpha, wp[0]);
        dg::blas1::axpby( 1., w[1], m_alpha, wp[1]);

        if( m_ex->m_p.beta != 0)
        {
            dg::blas2::symv( m_ex->m_multi_ampere[0], w[2], wp[2]);
            dg::blas1::pointwiseDot( m_ex->m_multi_ampere[0].inv_weights(), wp[2], wp[2]);
            dg::blas1::pointwiseDot( -m_ex->m_p.beta, m_ex->m_densityST[1], w[1],
                                      m_ex->m_p.beta, m_ex->m_densityST[0], w[0],
                                      1., wp[2]);
        }
    }
    void set_params( double alpha, double time){
        m_alpha = alpha;
        m_time = time;
        m_counter = 0;
    }
    private:
    Explicit<Geometry,IMatrix,Matrix,Container>* m_ex; // does not own anything
    double m_alpha = 0., m_time = 0.;
    unsigned m_counter = 0;
};

template< class Geometry, class IMatrix, class Matrix, class Container >
struct ImplicitSolver
{
    ImplicitSolver() {}
    ImplicitSolver( Explicit<Geometry,IMatrix,Matrix,Container>& ex,
        double eps_time) :
        m_ex(&ex), m_imdens(ex), m_imvelo(ex),
        m_y({ m_ex->m_density[0], m_ex->m_density[0], m_ex->m_density[0]}),
        m_rhs(m_y), m_eps_time(eps_time)
    {
        m_pcg.construct( m_ex->m_density, 3000);
        // these should be input parameters
        if( m_ex->m_p.solver_type == "lgmres")
            m_lgmres.construct( m_rhs, 30, 3, 100);
        else if( m_ex->m_p.solver_type == "bicgstabl")
            m_bicgstabl.construct( m_rhs, 3000, 2);
        else
            throw std::runtime_error( "Implicit solver type "+
                m_ex->m_p.solver_type+" not recognized!\n");
    }
    const std::array<std::array<Container,2>,2>& copyable() const{
        return m_ex->m_s;
    }
    // solve (y + alpha I(t,y) = rhs
    void solve( double alpha,
            Implicit<Geometry,IMatrix,Matrix,Container>& im,
            double t,
            std::array<std::array<Container,2>,2>& y,
            const std::array<std::array<Container,2>,2>& rhs)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif//MPI
        dg::Timer ti;
        ti.tic();
        m_imdens.set_params( alpha, t);
        m_imvelo.set_params( alpha, t);

        dg::blas2::symv( m_ex->m_lapperpN.weights(), rhs[0], m_ex->m_dssU);
        unsigned number = m_pcg( m_imdens, y[0], m_ex->m_dssU,
            m_ex->m_lapperpN.precond(), m_ex->m_lapperpN.inv_weights(), m_eps_time);
        ti.toc();
        DG_RANK0 std::cout << "# of PCG iterations time solver: "<<number<<"/"
                  <<m_pcg.get_max()<<" took "<<ti.diff()<<"s\n";
        ti.tic();
        m_ex->update_staggered_density_and_ampere( t, y[0]);
        double precond=1;
        dg::blas1::copy( y[1][0], m_y[0]);
        dg::blas1::copy( y[1][1], m_y[1]);
        m_ex->m_old_aparST.extrapolate( t, m_y[2]);
        dg::blas1::copy( rhs[1][0], m_rhs[0]);
        dg::blas1::copy( rhs[1][1], m_rhs[1]);
        dg::blas1::copy( 0., m_rhs[2]);
        if( m_ex->m_p.solver_type == "lgmres")
            number = m_lgmres.solve( m_imvelo, m_y, m_rhs,
                precond, m_ex->m_lapperpN.weights(), m_eps_time);
        else if( m_ex->m_p.solver_type == "bicgstabl")
            number = m_bicgstabl.solve( m_imvelo, m_y, m_rhs,
                precond, m_ex->m_lapperpN.weights(), m_eps_time);


        m_ex->m_old_aparST.update( t, m_y[2]);
        dg::blas1::copy( m_y[0], y[1][0]);
        dg::blas1::copy( m_y[1], y[1][1]);

        ti.toc();
        DG_RANK0 std::cout << "# of "<<m_ex->m_p.solver_type<<" iterations time solver: "<<number
                  <<" took "<<ti.diff()<<"s\n";

    }
    private:
    Explicit<Geometry,IMatrix,Matrix,Container>* m_ex; // does not own anything
    dg::CG< std::array<Container,2>> m_pcg;
    dg::LGMRES<std::array<Container,3>> m_lgmres;
    dg::BICGSTABl<std::array<Container,3>> m_bicgstabl;
    ImplicitDensityMatrix<Geometry,IMatrix, Matrix,Container> m_imdens;
    ImplicitVelocityMatrix<Geometry,IMatrix, Matrix,Container> m_imvelo;
    std::array<Container,3> m_y, m_rhs;
    double m_eps_time;
};

#else // WITH_NAVIER_STOKES
#include "../navier_stokes/navier_stokes.h"
#endif // WITH_NAVIER_STOKES

} //namespace feltor
