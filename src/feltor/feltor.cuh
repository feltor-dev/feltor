#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/geometries/geometries.h"
#ifndef DG_MANUFACTURED
#define FELTORPARALLEL 1
#define FELTORPERP 1
#endif

namespace feltor
{

namespace routines{
//Resistivity (consistent density dependency,
//parallel momentum conserving, quadratic current energy conservation dependency)
struct AddResistivity{
    AddResistivity( double C, std::array<double,2> mu): m_C(C){
        m_mu[0] = mu[0], m_mu[1] = mu[1];
    }
    DG_DEVICE
    void operator()( double ne, double ni, double ue,
        double ui, double& dtUe, double& dtUi) const{
        double current = ne*(ui-ue);
        dtUe += -m_C/m_mu[0] * current;
        dtUi += -m_C/m_mu[1] * ne/ni * current;
    }
    private:
    double m_C;
    double m_mu[2];
};
struct ComputePerpDrifts{
    ComputePerpDrifts( double mu, double tau, double beta):
        m_mu(mu), m_tau(tau), m_beta(beta){}
    DG_DEVICE
    void operator()(
            double N, double d0N, double d1N, double d2N,
            double U, double d0U, double d1U, double d2U,
            double d0P, double d1P, double d2P,
            double b_0,         double b_1,         double b_2,
            double curv0,       double curv1,       double curv2,
            double curvKappa0,  double curvKappa1,  double curvKappa2,
            double divCurvKappa,
            double& dtN, double& dtU
        )
    {
        double KappaU = curvKappa0*d0U+curvKappa1*d1U+curvKappa2*d2U;
        double KappaN = curvKappa0*d0N+curvKappa1*d1N+curvKappa2*d2N;
        double KappaP = curvKappa0*d0P+curvKappa1*d1P+curvKappa2*d2P;
        double KU = curv0*d0U+curv1*d1U+curv2*d2U;
        double KN = curv0*d0N+curv1*d1N+curv2*d2N;
        double KP = curv0*d0P+curv1*d1P+curv2*d2P;
        double PN = b_0*( d1P*d2N-d2P*d1N)+
                    b_1*( d2P*d0N-d0P*d2N)+
                    b_2*( d0P*d1N-d1P*d0N);//ExB drift
        double PU = b_0*( d1P*d2U-d2P*d1U)+
                    b_1*( d2P*d0U-d0P*d2U)+
                    b_2*( d0P*d1U-d1P*d0U);//ExB drift
        dtN =   -PN
                -N * KP
                -m_tau * KN
                -m_mu * U * U * KappaN
                -2. * m_mu * N * U * KappaU
                -m_mu * N * U * U * divCurvKappa;
        dtU =   -PU
                -U*KappaP
                -m_tau * KU
                -m_tau * U * divCurvKappa
                -(2. * m_tau + m_mu * U * U)*KappaU
                - 2. * m_tau * U * KappaN / N;
    }
    DG_DEVICE
    void operator()(
            double N, double d0N, double d1N, double d2N,
            double U, double d0U, double d1U, double d2U,
            double d0P, double d1P, double d2P,
            double A,       double d0A, double d1A, double d2A,
            double b_0,         double b_1,         double b_2,
            double curv0,       double curv1,       double curv2,
            double curvKappa0,  double curvKappa1,  double curvKappa2,
            double divCurvKappa,
            double& dtN, double& dtU
        )
    {
        //first compute the regular dynamics
        this->operator()( N,  d0N,  d1N,  d2N,
             U,        d0U,  d1U,  d2U,
             d0P,  d1P,  d2P,
             b_0,          b_1,          b_2,
             curv0,        curv1,        curv2,
             curvKappa0,   curvKappa1,   curvKappa2,
             divCurvKappa,
             dtN, dtU);
        //now add the additional terms from modified parallel derivative
        double KappaU = curvKappa0*d0U+curvKappa1*d1U+curvKappa2*d2U;
        double KappaN = curvKappa0*d0N+curvKappa1*d1N+curvKappa2*d2N;
        double KappaP = curvKappa0*d0P+curvKappa1*d1P+curvKappa2*d2P;

        double KnablaBA = (curv0-curvKappa0)*d0A
                         +(curv1-curvKappa1)*d1A
                         +(curv2-curvKappa2)*d2A;
        double UA = b_0*( d1U*d2A-d2U*d1A)+
                    b_1*( d2U*d0A-d0U*d2A)+
                    b_2*( d0U*d1A-d1U*d0A);
        double NA = b_0*( d1N*d2A-d2N*d1A)+
                    b_1*( d2N*d0A-d0N*d2A)+
                    b_2*( d0N*d1A-d1N*d0A);
        double PA = b_0*( d1P*d2A-d2P*d1A)+
                    b_1*( d2P*d0A-d0P*d2A)+
                    b_2*( d0P*d1A-d1P*d0A);
        dtN +=  -m_beta*( A*N*KappaU + A*U*KappaN + N*UA + U*NA)
                -m_beta*N*U*( A*divCurvKappa - KnablaBA);
        dtU +=  -m_beta/m_mu*( A*KappaP + PA)
                -m_beta*U*( A*KappaU + UA)
                -m_beta*m_tau/m_mu/N*(A*KappaN + NA);


    }
    private:
    double m_mu, m_tau, m_beta;
};
struct ComputeChi{
    DG_DEVICE
    void operator() ( double& chi, double tilde_Ni, double binv,
    double mu_i) const{
        chi = mu_i*(tilde_Ni+1.)*binv*binv;
    }
};
struct ComputePsi{
    DG_DEVICE
    void operator()( double& gradPhi2, double dxPhi, double dyPhi,
        double dzPhi, double& HdxPhi, double HdyPhi, double HdzPhi
        ) const{
        gradPhi2 = (dxPhi*HdxPhi + dyPhi*HdyPhi + dzPhi*HdzPhi);
    }
    DG_DEVICE
    void operator()( double& GammaPhi, double dxPhi, double dyPhi,
        double dzPhi, double& HdxPhi, double HdyPhi, double HdzPhi,
        double binv) const{
        //u_E^2
        this->operator()(
            HdxPhi, dxPhi, dyPhi, dzPhi, HdxPhi, HdyPhi , HdzPhi);
        HdxPhi   = binv*binv*HdxPhi;
        //Psi
        GammaPhi = GammaPhi - 0.5*HdxPhi;
    }
};
struct ComputeDiss{
    ComputeDiss( double mu, double tau):m_mu(mu), m_tau(tau){}
    DG_DEVICE
    void operator()( double& energy, double logN, double phi, double U) const{
        energy = m_tau*(1.+logN) + phi + 0.5*m_mu*U*U;
    }
    private:
    double m_mu, m_tau;
};
struct ComputeLogN{
    DG_DEVICE
    void operator()( double tilde_n, double& npe, double& logn) const{
        npe =  tilde_n + 1.;
        logn =  log(npe);
    }
};
struct ComputeSource{
    DG_DEVICE
    void operator()( double& result, double tilde_n, double profne,
        double source, double omega_source) const{
        double temp = omega_source*source*(profne - tilde_n);
        result = temp;
    }
};
}//namespace routines


template<class Geometry, class IMatrix, class Matrix, class container>
struct Implicit
{

    Implicit( const Geometry& g, feltor::Parameters p,
            dg::geo::TokamakMagneticField mag):
        m_p(p),
        m_lapM_perpN( g, p.bcxN,p.bcyN,dg::PER, dg::normed, dg::centered),
        m_lapM_perpU( g, p.bcxU,p.bcyU,dg::PER, dg::normed, dg::centered)
    {
        dg::assign( dg::evaluate( dg::zero, g), m_temp);
        auto bhat = dg::geo::createEPhi(); //bhat = ephi except when "true"
        if( p.curvmode == "true")
            bhat = dg::geo::createBHat(mag);
        dg::SparseTensor<dg::DVec> hh
            = dg::geo::createProjectionTensor( bhat, g);
        //set perpendicular projection tensor h
        m_lapM_perpN.set_chi( hh);
        m_lapM_perpU.set_chi( hh);
    }

    void operator()( double t, const std::array<std::array<container,2>,2>& y, std::array<std::array<container,2>,2>& yp)
    {
#if FELTORPERP == 1
        /* y[0][0] := N_e - 1
           y[0][1] := N_i - 1
           y[1][0] := w_e
           y[1][1] := W_i
        */
        for( unsigned i=0; i<2; i++)
        {
            //dissipation acts on w!
            if( m_p.perp_diff == "hyperviscous")
            {
                dg::blas2::symv( m_lapM_perpN, y[0][i],      m_temp);
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpN, m_temp, 0., yp[0][i]);
                dg::blas2::symv( m_lapM_perpU, y[1][i],      m_temp);
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpU, m_temp, 0., yp[1][i]);
            }
            else // m_p.perp_diff == "viscous"
            {
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpN, y[0][i],  0., yp[0][i]);
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpU, y[1][i],  0., yp[1][i]);
            }
        }
#else
        dg::blas1::copy( 0, yp);
#endif
    }

    const container& weights() const{
        return m_lapM_perpU.weights();
    }
    const container& inv_weights() const {
        return m_lapM_perpU.inv_weights();
    }
    const container& precond() const {
        return m_lapM_perpU.precond();
    }

  private:
    const feltor::Parameters m_p;
    container m_temp;
    dg::Elliptic3d<Geometry, Matrix, container> m_lapM_perpN, m_lapM_perpU;
};

struct Quantities
{
    double mass = 0, diff = 0; //mass and mass diffusion
    double energy = 0, ediff = 0; //total energy and energy diffusion
    //entropy parallel and perp energies
    double S[2] = {0,0}, Tpar[2] = {0,0}, Tperp = 0, Apar = 0;
    //resisitive and diffusive terms
    double Dres = 0, Dpar[4] = {0,0,0,0}, Dperp[4] = {0,0,0,0};
    double aligned = 0; //alignment parameter
    void display( std::ostream& os = std::cout ) const
    {
        os << "Quantities: \n"
           << "    Mass: "<<std::setw(11)<< mass  <<" Mass diffusion   "<<diff<<"\n"
           << "  Energy: "<<std::setw(11)<<energy <<" Energy diffusion "<<ediff<<"\n"
           << "       S: ["<<S[0]<<", "<<S[1]<<"]\n"
           << "   Tperp: "<<Tperp<<"  Apar: "<<Apar<<"\n"
           << "    Tpar: ["<<Tpar[0]<<", "<<Tpar[1]<<"]\n"
           << "    Dres: "<<Dres<<"\n"
           << "    Dpar: ["<<Dpar[0]<<", "<<Dpar[1]<<", "<<Dpar[2]<<", "<<Dpar[3]<<"]\n"
           << "   Dperp: ["<<Dperp[0]<<", "<<Dperp[1]<<", "<<Dperp[2]<<", "<<Dperp[3]<<"]\n"
           << " aligned: "<<aligned;
    }
};

template< class Geometry, class IMatrix, class Matrix, class container >
struct Explicit
{
    Explicit( const Geometry& g, feltor::Parameters p,
        dg::geo::TokamakMagneticField mag);

    //potential[0]: electron potential, potential[1]: ion potential
    const std::array<container,2>& potential( ) const {
        return m_phi;
    }
    const container& induction() const {
        return m_apar;
    }
    //Given N_i-1 initialize n_e-1 such that phi=0
    void initializene( const container& ni, container& ne);
    //Given n_e-1 initialize N_i-1 such that phi=0
    void initializeni( const container& ne, container& ni);

    void operator()( double t,
        const std::array<std::array<container,2>,2>& y,
        std::array<std::array<container,2>,2>& yp);

    // update quantities to the state of the last call to operator()
    // This is to possibly save some computation time in the timestepper
    void update_quantities() {
        // set energy quantities in m_q, --- needs m_apar, m_logn and m_UE2
        compute_energies( m_fields);

        // remaining of m_q --- needs Delta_par U, N, m_logn
        compute_dissipation( m_fields);
    }

    // get a link to the internal storage for quantities
    const Quantities& quantities( ) const{
        return m_q;
    }
    const std::array<std::array<container,2>,2>& fields() const{
        return m_fields;
    }
    const dg::geo::DS<Geometry, IMatrix, Matrix, container>& ds() {
        return m_ds_N;
    }

    //source strength, profile - 1 and damping
    void set_source( double omega_source, container profile, container damping)
    {
        m_omega_source = omega_source;
        m_profne = profile;
        m_source = damping;

    }
  private:
    void compute_phi( double t, const std::array<container,2>& y);
    void compute_psi( double t, const std::array<container,2>& y);
    void compute_apar( double t, std::array<std::array<container,2>,2>& fields);
    void compute_energies(
        const std::array<std::array<container,2>,2>& fields);
    void compute_dissipation(
        const std::array<std::array<container,2>,2>& fields);
    void compute_perp( double t,
        const std::array<std::array<container,2>,2>& y,
        const std::array<std::array<container,2>,2>& fields,
        std::array<std::array<container,2>,2>& yp);
    void compute_parallel( double t,
        const std::array<std::array<container,2>,2>& y,
        const std::array<std::array<container,2>,2>& fields,
        std::array<std::array<container,2>,2>& yp);
    void construct_mag( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);
    void construct_bhat( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);
    void construct_invert( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField);

    container m_UE2;
    container m_temp0, m_temp1, m_temp2;//helper variables
#ifdef DG_MANUFACTURED
    container m_R, m_Z, m_P; //coordinates
#endif //DG_MANUFACTURED

    //these should be considered const
    std::array<container,3> m_curv, m_curvKappa, m_b;
    container m_divCurvKappa;
    container m_binv, m_divb;
    container m_source, m_profne;
    container m_vol3d;

    container m_apar, m_dxA, m_dyA, m_dzA;
    std::array<container,2> m_phi, m_dxPhi, m_dyPhi, m_dzPhi;
    std::array<container,2> m_logn, m_dxN, m_dyN, m_dzN, m_dsN;
    std::array<container,2> m_dxU, m_dyU, m_dzU, m_dsU;
    std::array<std::array<container,2>,2> m_fields;

    std::vector<container> m_multi_chi;

    //matrices and solvers
    Matrix m_dx_N, m_dx_U, m_dx_P, m_dx_A, m_dy_N, m_dy_U, m_dy_P, m_dy_A, m_dz;
    dg::geo::DS<Geometry, IMatrix, Matrix, container> m_ds_P, m_ds_N, m_ds_U;
    dg::Elliptic3d< Geometry, Matrix, container> m_lapperpN, m_lapperpU;
    std::vector<dg::Elliptic3d< Geometry, Matrix, container> > m_multi_pol;
    std::vector<dg::Helmholtz3d<Geometry, Matrix, container> > m_multi_invgammaP,
        m_multi_invgammaN, m_multi_induction;

    dg::MultigridCG2d<Geometry, Matrix, container> m_multigrid;
    dg::Extrapolation<container> m_old_phi, m_old_psi, m_old_gammaN, m_old_apar;

    dg::SparseTensor<container> m_hh;

    const feltor::Parameters m_p;
    Quantities m_q;
    double m_omega_source =0.;

};

template<class Grid, class IMatrix, class Matrix, class container>
void Explicit<Grid, IMatrix, Matrix, container>::construct_mag(
    const Grid& g, feltor::Parameters p, dg::geo::TokamakMagneticField mag)
{
    //due to the various approximations bhat and mag not always correspond
    dg::geo::CylindricalVectorLvl0 curvNabla, curvKappa;
    if( p.curvmode == "true" )
    {
        curvNabla = dg::geo::createTrueCurvatureNablaB(mag);
        curvKappa = dg::geo::createTrueCurvatureKappa(mag);
        dg::assign(  dg::pullback(dg::geo::TrueDivCurvatureKappa(mag), g),
            m_divCurvKappa);
    }
    else if( p.curvmode == "low beta")
    {
        curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag);
        dg::assign( dg::evaluate(dg::zero, g), m_divCurvKappa);
    }
    else if( p.curvmode == "toroidal")
    {
        curvNabla = dg::geo::createCurvatureNablaB(mag);
        curvKappa = dg::geo::createCurvatureKappa(mag);
        dg::assign(  dg::pullback(dg::geo::DivCurvatureKappa(mag), g),
            m_divCurvKappa);
    }
    dg::pushForward(curvNabla.x(), curvNabla.y(), curvNabla.z(),
        m_curv[0], m_curv[1], m_curv[2], g);
    dg::pushForward(curvKappa.x(), curvKappa.y(), curvKappa.z(),
        m_curvKappa[0], m_curvKappa[1], m_curvKappa[2], g);
    dg::blas1::axpby( 1., m_curvKappa, 1., m_curv);
    dg::assign(  dg::pullback(dg::geo::InvB(mag), g), m_binv);
    dg::assign(  dg::pullback(dg::geo::Divb(mag), g), m_divb);

}
template<class Grid, class IMatrix, class Matrix, class container>
void Explicit<Grid, IMatrix, Matrix, container>::construct_bhat(
    const Grid& g, feltor::Parameters p, dg::geo::TokamakMagneticField mag)
{
    //in DS we take the true bhat
    auto bhat = dg::geo::createBHat( mag);
    m_ds_N.construct( bhat, g, p.bcxN, p.bcyN, dg::geo::NoLimiter(),
        dg::forward, p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz );
    if( p.bcxU == p.bcxN && p.bcyU == p.bcyN)
        m_ds_U.construct( m_ds_N);
    else
        m_ds_U.construct( bhat, g, p.bcxU, p.bcyU, dg::geo::NoLimiter(),
            dg::forward, p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz);
    if( p.bcxP == p.bcxN && p.bcyP == p.bcyN)
        m_ds_P.construct( m_ds_N);
    else if( p.bcxP == p.bcxU && p.bcyP == p.bcyU)
        m_ds_P.construct( m_ds_U);
    else
        m_ds_P.construct( bhat, g, p.bcxP, p.bcyP, dg::geo::NoLimiter(),
            dg::forward, p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz);

    // in Poisson we take EPhi except for the true curvmode
    bhat = dg::geo::createEPhi();
    if( p.curvmode == "true")
        bhat = dg::geo::createBHat(mag);
    dg::pushForward(bhat.x(), bhat.y(), bhat.z(), m_b[0], m_b[1], m_b[2], g);
    dg::SparseTensor<dg::DVec> metric = g.metric();
    dg::tensor::inv_multiply3d( metric, m_b[0], m_b[1], m_b[2],
                                        m_b[0], m_b[1], m_b[2]);
    container vol = dg::tensor::volume( metric);
    dg::blas1::pointwiseDivide( m_binv, vol, vol); //1/vol/B
    for( int i=0; i<3; i++)
        dg::blas1::pointwiseDot( vol, m_b[i], m_b[i]); //b_i/vol/B
    m_hh = dg::geo::createProjectionTensor( bhat, g);
    m_lapperpN.construct ( g, p.bcxN, p.bcyN, dg::PER, dg::normed, dg::centered),
    m_lapperpU.construct ( g, p.bcxU, p.bcyU, dg::PER, dg::normed, dg::centered),
    m_lapperpN.set_chi( m_hh);
    m_lapperpU.set_chi( m_hh);
}
template<class Grid, class IMatrix, class Matrix, class container>
void Explicit<Grid, IMatrix, Matrix, container>::construct_invert(
    const Grid& g, feltor::Parameters p, dg::geo::TokamakMagneticField mag)
{
    /////////////////////////init elliptic and helmholtz operators/////////
    auto bhat = dg::geo::createEPhi(); //bhat = ephi except when "true"
    if( p.curvmode == "true")
        bhat = dg::geo::createBHat( mag);
    m_multi_chi = m_multigrid.project( m_temp0);
    m_multi_pol.resize(p.stages);
    m_multi_invgammaP.resize(p.stages);
    m_multi_invgammaN.resize(p.stages);
    m_multi_induction.resize(p.stages);
    for( unsigned u=0; u<p.stages; u++)
    {
        dg::SparseTensor<dg::DVec> hh = dg::geo::createProjectionTensor(
            bhat, m_multigrid.grid(u));
        m_multi_pol[u].construct( m_multigrid.grid(u),
            p.bcxP, p.bcyP, dg::PER, dg::not_normed,
            dg::centered, p.jfactor);
        m_multi_pol[u].set_chi( hh);
        m_multi_invgammaP[u].construct(  m_multigrid.grid(u),
            p.bcxP, p.bcyP, dg::PER, -0.5*p.tau[1]*p.mu[1], dg::centered);
        m_multi_invgammaP[u].elliptic().set_chi( hh);
        m_multi_invgammaN[u].construct(  m_multigrid.grid(u),
            p.bcxN, p.bcyN, dg::PER, -0.5*p.tau[1]*p.mu[1], dg::centered);
        m_multi_invgammaN[u].elliptic().set_chi( hh);
        m_multi_induction[u].construct(  m_multigrid.grid(u),
            p.bcxA, p.bcyA, dg::PER, -1., dg::centered);
        m_multi_induction[u].elliptic().set_chi( hh);
    }
}
template<class Grid, class IMatrix, class Matrix, class container>
Explicit<Grid, IMatrix, Matrix, container>::Explicit( const Grid& g,
    feltor::Parameters p, dg::geo::TokamakMagneticField mag):
#ifdef DG_MANUFACTURED
    m_R( dg::pullback( dg::cooX3d, g)),
    m_Z( dg::pullback( dg::cooY3d, g)),
    m_P( dg::pullback( dg::cooZ3d, g)),
#endif //DG_MANUFACTURED
    /////////the poisson operators ////////////////////////////////////////
    m_dx_N( dg::create::dx( g, p.bcxN) ),
    m_dx_U( dg::create::dx( g, p.bcxU) ),
    m_dx_P( dg::create::dx( g, p.bcxP) ),
    m_dx_A( dg::create::dx( g, p.bcxA) ),
    m_dy_N( dg::create::dy( g, p.bcyN) ),
    m_dy_U( dg::create::dy( g, p.bcyU) ),
    m_dy_P( dg::create::dy( g, p.bcyP) ),
    m_dy_A( dg::create::dy( g, p.bcyA) ),
    m_dz( dg::create::dz( g, dg::PER) ),
    /////////the elliptic and Helmholtz operators//////////////////////////
    m_multigrid( g, p.stages),
    m_old_phi( 2, dg::evaluate( dg::zero, g)),
    m_old_psi( m_old_phi), m_old_gammaN( m_old_phi), m_old_apar( m_old_phi),
    m_p(p)
{
    ////////////////////////////init temporaries///////////////////
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_UE2 = m_temp2 = m_temp1 = m_temp0;
    m_phi[0] = m_phi[1] = m_temp0;
    m_dxPhi = m_dyPhi = m_dzPhi = m_fields[0] = m_fields[1] = m_logn = m_phi;
    m_dxN = m_dyN = m_dzN = m_dsN = m_dxU = m_dyU = m_dzU = m_dsU = m_phi;
    m_apar = m_dxA = m_dyA = m_dzA = m_phi[0];
    construct_mag( g, p, mag);
    construct_bhat( g, p, mag);
    construct_invert( g, p, mag);
    //////////////////////////////Metric///////////////////////////////
    dg::assign( dg::create::volume(g), m_vol3d);
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::initializene(
    const container& src, container& target)
{
    // ne = Ni
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        // ne-1 = Gamma (ni-1)
        std::vector<unsigned> number = m_multigrid.direct_solve(
            m_multi_invgammaN, target, src, m_p.eps_gamma);
        if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_gamma);
    }
}
template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::initializeni(
    const container& src, container& target)
{
    // Ni = ne
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        //add FLR correction -0.5*tau*mu*Delta n_e
        dg::blas2::symv( 0.5*m_p.tau[1]*m_p.mu[1],
            m_lapperpN, src, 1.0, target);
    }
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_phi(
    double time, const std::array<container,2>& y)
{
    //y[0]:= n_e - 1
    //y[1]:= N_i - 1
    //----------Compute and set chi----------------------------//
    dg::blas1::subroutine( routines::ComputeChi(),
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
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
            m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,time);
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
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
        m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
    //----------Invert polarisation----------------------------//
    m_old_phi.extrapolate( time, m_phi[0]);
    std::vector<unsigned> number = m_multigrid.direct_solve(
        m_multi_pol, m_phi[0], m_temp0, m_p.eps_pol);
    m_old_phi.update( time, m_phi[0]);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol);
    //-----------Solve for Gamma Phi---------------------------//
    if (m_p.tau[1] == 0.) {
        dg::blas1::copy( m_phi[0], m_phi[1]);
    } else {
        m_old_psi.extrapolate( time, m_phi[1]);
#ifdef DG_MANUFACTURED
        dg::blas1::copy( m_phi[0], m_temp0);
        dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SGammaPhie{
            m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
            m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,time);
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
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_psi(
    double time, const std::array<container,2>& y)
{
    //-------Compute Psi and derivatives
    dg::blas2::symv( m_dx_P, m_phi[0], m_dxPhi[0]);
    dg::blas2::symv( m_dy_P, m_phi[0], m_dyPhi[0]);
    dg::blas2::symv( m_dz  , m_phi[0], m_dzPhi[0]);
    dg::tensor::multiply3d( m_hh, //grad_perp
        m_dxPhi[0], m_dyPhi[0], m_dzPhi[0], m_UE2, m_temp0, m_temp1);
    dg::blas1::subroutine( routines::ComputePsi(),
        m_phi[1], m_dxPhi[0], m_dyPhi[0], m_dzPhi[0],
        m_UE2, m_temp0, m_temp1, m_binv);
#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( m_phi[1], dg::plus_equals(), manufactured::SPhii{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
        m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
    //m_UE2 now contains u_E^2; also update derivatives
    dg::blas2::symv( m_dx_P, m_phi[1], m_dxPhi[1]);
    dg::blas2::symv( m_dy_P, m_phi[1], m_dyPhi[1]);
    dg::blas2::symv( m_dz  , m_phi[1], m_dzPhi[1]);
}
template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_apar(
    double time, std::array<std::array<container,2>,2>& fields)
{
    //on input
    //fields[0][0] = n_e, fields[1][0]:= w_e
    //fields[0][1] = N_i, fields[1][1]:= W_i
    //----------Compute and set chi----------------------------//
    dg::blas1::axpby(  m_p.beta/m_p.mu[1], fields[0][1],
                      -m_p.beta/m_p.mu[0], fields[0][0], m_temp0);
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_induction[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    dg::blas1::pointwiseDot(  1., fields[0][1], fields[1][1],
                             -1., fields[0][0], fields[1][0], 0., m_temp0);
    //----------Invert Induction Eq----------------------------//
    m_old_apar.extrapolate( time, m_apar);
#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( m_temp0, dg::plus_equals(), manufactured::SA{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
        m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,time);
#endif //DG_MANUFACTURED
    std::vector<unsigned> number = m_multigrid.direct_solve(
        m_multi_induction, m_apar, m_temp0, m_p.eps_pol);
    m_old_apar.update( time, m_apar);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol);
    //----------Compute Derivatives----------------------------//
    dg::blas2::symv( m_dx_A, m_apar, m_dxA);
    dg::blas2::symv( m_dy_A, m_apar, m_dyA);
    dg::blas2::symv( m_dz,   m_apar, m_dzA);

    //----------Compute Velocities-----------------------------//
    dg::blas1::axpby( 1., fields[1][0], -m_p.beta/m_p.mu[0], m_apar, fields[1][0]);
    dg::blas1::axpby( 1., fields[1][1], -m_p.beta/m_p.mu[1], m_apar, fields[1][1]);
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_perp(
    double t,
    const std::array<std::array<container,2>,2>& y,
    const std::array<std::array<container,2>,2>& fields,
    std::array<std::array<container,2>,2>& yp)
{
    //y[0] = N-1, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        dg::blas2::symv( m_dx_N, y[0][i], m_dxN[i]);
        dg::blas2::symv( m_dy_N, y[0][i], m_dyN[i]);
        dg::blas2::symv( m_dz,   y[0][i], m_dzN[i]);
        dg::blas2::symv( m_dx_U, fields[1][i], m_dxU[i]);
        dg::blas2::symv( m_dy_U, fields[1][i], m_dyU[i]);
        dg::blas2::symv( m_dz,   fields[1][i], m_dzU[i]);
        if( m_p.beta == 0){
            dg::blas1::subroutine( routines::ComputePerpDrifts(
                m_p.mu[i], m_p.tau[i], m_p.beta),
                //species depdendent
                fields[0][i], m_dxN[i], m_dyN[i], m_dzN[i],
                fields[1][i], m_dxU[i], m_dyU[i], m_dzU[i],
                m_dxPhi[i], m_dyPhi[i], m_dzPhi[i],
                //magnetic parameters
                m_b[0], m_b[1], m_b[2],
                m_curv[0], m_curv[1], m_curv[2],
                m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
                m_divCurvKappa, yp[0][i], yp[1][i]
            );
        }
        if( m_p.beta != 0){
            dg::blas1::subroutine( routines::ComputePerpDrifts(
                m_p.mu[i], m_p.tau[i], m_p.beta),
                //species depdendent
                fields[0][i], m_dxN[i], m_dyN[i], m_dzN[i],
                fields[1][i], m_dxU[i], m_dyU[i], m_dzU[i],
                m_dxPhi[i], m_dyPhi[i], m_dzPhi[i],
                //induction
                m_apar, m_dxA, m_dyA, m_dzA,
                //magnetic parameters
                m_b[0], m_b[1], m_b[2],
                m_curv[0], m_curv[1], m_curv[2],
                m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
                m_divCurvKappa, yp[0][i], yp[1][i]
            );
        }
    }
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_parallel(
    double t,
    const std::array<std::array<container,2>,2>& y,
    const std::array<std::array<container,2>,2>& fields,
    std::array<std::array<container,2>,2>& yp)
{
    //y[0] = N-1, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {
        //---------------------density--------------------------//
        //density: -Div ( NUb)
        m_ds_N.centered( y[0][i], m_dsN[i]);
        m_ds_U.centered( fields[1][i], m_dsU[i]);
        dg::blas1::pointwiseDot(-1., m_dsN[i], fields[1][i],
            -1., fields[0][i], m_dsU[i], 1., yp[0][i] );
        dg::blas1::pointwiseDot( -1., fields[0][i],fields[1][i],m_divb,
            1.,yp[0][i]);
        //density: + nu_par Delta_par N
        dg::blas1::pointwiseDot( m_p.nu_parallel, m_divb, m_dsN[i],
                                 1., yp[0][i]);
        m_ds_N.dss( y[0][i], m_dsN[i]);
        dg::blas1::axpby( m_p.nu_parallel, m_dsN[i], 1., yp[0][i]);
        //---------------------velocity-------------------------//
        // Burgers term: -0.5 ds U^2
        dg::blas1::pointwiseDot(fields[1][i], fields[1][i], m_temp1); //U^2
        m_ds_U.centered(-0.5, m_temp1, 1., yp[1][i]);
        // force terms: -tau/mu * ds lnN -1/mu * ds Phi
        // (These two terms converge slowly and require high z resolution)
        m_ds_N.centered(-m_p.tau[i]/m_p.mu[i], m_logn[i], 1.0, yp[1][i]);
        m_ds_P.centered(-1./m_p.mu[i], m_phi[i], 1.0, yp[1][i]);
        // diffusion: + nu_par Delta_par U
        dg::blas1::pointwiseDot(m_p.nu_parallel, m_divb, m_dsU[i],
                                1., yp[1][i]);
        m_ds_U.dss( fields[1][i], m_dsU[i]);
        dg::blas1::axpby( m_p.nu_parallel, m_dsU[i], 1., yp[1][i]);
    }
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_energies(
    const std::array<std::array<container,2>,2>& fields)
{
    double z[2]    = {-1.0,1.0};
    m_q.mass = dg::blas1::dot( m_vol3d, fields[0][0]);
    for(unsigned i=0; i<2; i++)
    {
        m_q.S[i] = z[i]*m_p.tau[i]*dg::blas2::dot(
            m_logn[i], m_vol3d, fields[0][i]);
        dg::blas1::pointwiseDot( fields[1][i], fields[1][i], m_temp0); //U^2
        m_q.Tpar[i] = z[i]*0.5*m_p.mu[i]*dg::blas2::dot(
            fields[0][i], m_vol3d, m_temp0);
    }
    //= 0.5 beta (grad_perp Apar)^2
    if( m_p.beta != 0)
    {
        dg::tensor::multiply3d( m_hh, m_dxA, m_dyA, m_dzA,
            m_temp0, m_temp1, m_temp2);
        dg::blas1::subroutine( routines::ComputePsi(),
            m_temp0, m_dxA, m_dyA, m_dzA,
            m_temp0, m_temp1, m_temp2);
        m_q.Apar = 0.5*m_p.beta*dg::blas1::dot( m_vol3d, m_temp0);
    }
    //= 0.5 mu_i N_i u_E^2
    m_q.Tperp = 0.5*m_p.mu[1]*dg::blas2::dot( fields[0][1], m_vol3d, m_UE2);
    m_q.energy = m_q.S[0] + m_q.S[1] + m_q.Tperp + m_q.Apar
                 + m_q.Tpar[0] + m_q.Tpar[1];
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_dissipation(
    const std::array<std::array<container,2>,2>& fields)
{
    //alignement: lnN * Delta_s N
    m_q.aligned = dg::blas2::dot( m_logn[0], m_vol3d, m_dsN[0]);
    /////////////////DISSIPATION TERMS//////////////////////////////
    m_q.diff = m_p.nu_parallel*dg::blas1::dot( m_vol3d, m_dsN[0]);
    // energy dissipation through diffusion
    double z[2] = {-1.0,1.0};
    for( unsigned i=0; i<2;i++)
    {
        //Compute dissipation for N
        // Z*(tau (1+lnN )+psi + 0.5 mu U^2)
        dg::blas1::subroutine( routines::ComputeDiss(m_p.mu[i], m_p.tau[i]),
                m_temp2, m_logn[i], m_phi[i], fields[1][i]);
        // perp dissipation for N: nu_perp Delta_p N or -nu_perp Delta_p**2 N
        if( m_p.perp_diff == "viscous")
        {
            dg::blas1::transform( fields[0][i], m_temp1, dg::PLUS<double>(-1));
            //minus in Laplacian!
            dg::blas2::gemv( m_lapperpN, m_temp1, m_temp0);
        }
        else
        {
            dg::blas1::transform( fields[0][i], m_temp0, dg::PLUS<double>(-1));
            dg::blas2::gemv( m_lapperpN, m_temp0, m_temp1);
            dg::blas2::gemv( m_lapperpN, m_temp1, m_temp0);
        }
        if( i==0)
            m_q.diff += -m_p.nu_perp*dg::blas1::dot( m_vol3d, m_temp0);
        m_q.Dperp[i] = -z[i]*m_p.nu_perp*dg::blas2::dot(
                        m_temp2, m_vol3d, m_temp0);
        // parallel dissipation for N: nu_parallel *(Delta_s N)
        m_q.Dpar[i] = z[i]*m_p.nu_parallel*dg::blas2::dot(
                        m_temp2, m_vol3d, m_dsN[i]);
        //Compute parallel dissipation for U
        //Z*mu*N*U nu_parallel *( Delta_s U)
        dg::blas1::pointwiseDot( z[i]*m_p.mu[i], fields[0][i], fields[1][i],
                0, m_temp2);
        // perp dissipation for U: nu_perp Delta_p W or -nu_perp Delta_p**2 W
        if( m_p.perp_diff == "viscous")
        {
            dg::blas1::axpby( m_p.beta/m_p.mu[i], m_apar, 1., fields[1][i], m_temp1);
            dg::blas2::gemv( m_lapperpU, m_temp1, m_temp0);
        }
        else
        {
            dg::blas1::axpby( m_p.beta/m_p.mu[i], m_apar, 1., fields[1][i], m_temp0);
            dg::blas2::gemv( m_lapperpU, m_temp0, m_temp1);
            dg::blas2::gemv( m_lapperpU, m_temp1, m_temp0);
        }
        m_q.Dperp[i+2] = -m_p.nu_perp *dg::blas2::dot(
            m_temp2, m_vol3d, m_temp0);
        // parallel dissipation for U: nu_parallel *(Delta_s U)
        m_q.Dpar[i+2] = m_p.nu_parallel*dg::blas2::dot(
            m_temp2, m_vol3d, m_dsU[i]);
    }
    // resistive energy (quadratic current): -C (n_e (U_i-u_e))**2
    dg::blas1::pointwiseDot(1., fields[0][0], fields[1][1],
        -1., fields[0][0], fields[1][0], 0., m_temp0);
    m_q.Dres = -m_p.c*dg::blas2::dot(m_temp0, m_vol3d, m_temp0);
    m_q.ediff = m_q.Dres
        + m_q.Dpar[0]+m_q.Dperp[0]+m_q.Dpar[1]+m_q.Dperp[1]
        + m_q.Dpar[2]+m_q.Dperp[2]+m_q.Dpar[3]+m_q.Dperp[3];
}

/* y[0][0] := n_e - 1
   y[0][1] := N_i - 1
   y[1][0] := w_e
   y[1][1] := W_i
*/
template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::operator()(
    double t,
    const std::array<std::array<container,2>,2>& y,
    std::array<std::array<container,2>,2>& yp)
{
    dg::Timer timer;
    timer.tic();

    // set m_phi[0]
    compute_phi( t, y[0]);

    // set m_psi[0], m_d*Phi[0], m_d*Phi[1] and m_UE2 --- needs m_phi[0]
    compute_psi( t, y[0]);

    // Transform n-1 to n and n to logn
    dg::blas1::subroutine( routines::ComputeLogN(), y[0], m_fields[0], m_logn);

    // Compute Apar and m_U if necessary --- reads and updates m_fields
    dg::blas1::copy( y[1], m_fields[1]);
    if( m_p.beta != 0)
        compute_apar( t, m_fields);

#if FELTORPERP == 1

    // Set perpendicular dynamics in yp
    compute_perp( t, y, m_fields, yp);
    //------------------Add Resistivity--------------------------//
    dg::blas1::subroutine( routines::AddResistivity( m_p.c, m_p.mu),
        m_fields[0][0], m_fields[0][1], m_fields[1][0], m_fields[1][1],
        yp[1][0], yp[1][1]);

#else

    dg::blas1::copy( 0., yp);

#endif

    // Add parallel dynamics --- needs m_logn
#if FELTORPARALLEL == 1
    compute_parallel( t, y, m_fields, yp);
#endif

    //Add particle source to dtNe
    if( m_omega_source != 0)
    {
        dg::blas1::subroutine( routines::ComputeSource(),
            m_temp1, y[0][0], m_profne, m_source, m_omega_source);
        dg::blas1::axpby( 1., m_temp1, 1.0, yp[0][0]);
        //add FLR correction to dtNi
        dg::blas1::axpby( 1., m_temp1, 1.0, yp[1][1]);
        dg::blas2::gemv( 0.5*m_p.tau[1]*m_p.mu[1],
            m_lapperpN, m_temp1, 1.0, yp[1][1]);
    }
#ifdef DG_MANUFACTURED
    dg::blas1::evaluate( yp[0][0], dg::plus_equals(), manufactured::SNe{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
        m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[0][1], dg::plus_equals(), manufactured::SNi{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
        m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[1][0], dg::plus_equals(), manufactured::SUe{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
        m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,t);
    dg::blas1::evaluate( yp[1][1], dg::plus_equals(), manufactured::SUi{
        m_p.mu[0],m_p.mu[1],m_p.tau[0],m_p.tau[1],m_p.c,
        m_p.beta,m_p.nu_perp,m_p.nu_parallel},m_R,m_Z,m_P,t);
#endif //DG_MANUFACTURED
    timer.toc();
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif
    //std::cout << "#One rhs took "<<timer.diff()<<"s\n";
}

} //namespace feltor
