#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "geometries/geometries.h"

namespace feltor
{
//Resistivity (consistent density dependency, parallel momentum conserving, quadratic current energy conservation dependency)
struct AddResistivity{
    ddResistivity( double C, std::array<double,2> mu): m_C(C), m_mu(mu){
    }
    DG_DEVICE
    void operator()( double tilde_ne, double tilde_ni, double ue, double ui, double& dtUe, double& dtUi) const{
        double current = (tilde_ne+1)*(ui-ue);
        dtUe += -m_C/m_mu[0] * current;
        dtUi += -m_C/m_mu[1] * (tilde_ne+1)/(tilde_ni+1) * current;
    }
    private:
    double m_C;
    std::array<double,2> m_mu;
};

template<class Geometry, class IMatrix, class Matrix, class container>
struct Implicit
{

    Implicit( const Geometry& g, feltor::Parameters p, dg::geo::solovev::Parameters gp, dg::geo::DS<Geometry, IMatrix, Matrix, container> dsN, dg::geo::DS<Geometry, IMatrix, Matrix,  container> dsDIR):
        m_p(p),
        m_gp(gp),
        m_lapM_perpN  ( g, p.bc, p.bc, dg::normed, dg::centered),
        m_lapM_perpD( g, dg::DIR, dg::DIR, dg::normed, dg::centered)
    {
        dg::assign( dg::evaluate( dg::zero, g), m_temp);
        //set perpendicular projection tensor h
        const dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
        const dg::geo::BinaryVectorLvl0 bhat = dg::geo::createBHat(mag);
        dg::SparseTensor<dg::DVec> hh = dg::geo::createProjectionTensor( bhat, g3d);
        m_lapM_perpN.set_chi( hh);
        m_lapM_perpD.set_chi( hh);
    }

    void operator()( double t, const std::array<std::array<container,2>,2>& y, std::array<std::array<container,2>,2>& yp)
    {
        /* y[0][0] := N_e - 1
           y[0][1] := N_i - 1
           y[1][0] := U_e
           y[1][1] := U_i
        */
        for( unsigned i=0; i<2; i++)
        {
            //perpendicular hyperdiffusion for N and U
            dg::blas2::gemv( m_lapM_perpN, y[0][i],      m_temp);
            dg::blas2::gemv( -m_p.nu_perp, m_lapM_perpN, m_temp, 0., yp[0][i]);
            dg::blas2::gemv( m_lapM_perpD, y[1][i],      m_temp);
            dg::blas2::gemv( -m_p.nu_perp, m_lapM_perpD, m_temp, 0., yp[1][i]);
        }
    }

    dg::Elliptic3d<Geometry, Matrix, container>& laplacianM() {
        return m_lapM_perpD;
    }

    const container& weights() const{
        return m_lapM_perpD.weights();
    }
    const container& inv_weights() const {
        return m_lapM_perpD.inv_weights();
    }
    const container& precond() const {
        return m_lapM_perpD.precond();
    }

  private:
    const feltor::Parameters m_p;
    const dg::geo::solovev::Parameters m_gp;
    container m_temp;
    dg::Elliptic3d<Geometry, Matrix, container> m_lapM_perpN, m_lapM_perpD;
};

template< class Geometry, class IMatrix, class Matrix, class container >
struct Explicit
{
    Explicit( const Geometry& g, feltor::Parameters p, dg::geo::solovev::Parameters gp);


    dg::geo::DS<Geometry, IMatrix, Matrix, container>& ds(){
        return m_dsN;
    }
    dg::geo::DS<Geometry, IMatrix, Matrix, container>& dsDIR(){
        return m_dsDIR;
    }

    //potential[0]: electron potential, potential[1]: ion potential
    const std::array<container,2>& potential( ) const {
        return m_phi;
    }
    /**
     * @brief Given N_i-1 initialize N_e -1 such that phi=0
     *
     * @param y N_i -1
     * @param target N_e -1
     */
    void initializene( const container& y, container& target);

    ///@param y y[0] := N_e - 1, y[1] := N_i - 1, y[2] := U_e, y[3] := U_i
    void operator()( double t, const std::array<std::array<container,2>,2>& y, std::array<std::array<container,2>,2>& yp);

    /**
     * @brief \f[ M := \int_V (n_e-1) dV \f]
     *
     * @return mass of plasma contained in volume
     * @note call energies() before use
     */
    double mass( ) const {
        return m_mass;
    }
    /**
     * @brief
     \f[
\begin{align}
 E = \partial_t \sum_z \int_V d^3x \left[\frac{1}{2}m NU^2 + \frac{(\nabla_\perp A_\parallel)^2}{2\mu_0} + \frac{1}{2}mN\left(\frac{\nabla_\perp\phi}{B}\right)^2 + T N\ln(N)\right]
\end{align}
\f]

     * @return Total energy contained in volume
     */
    double energy( ) const {
        return m_energy;
    }

    //E[0]=S_e, E[1] = S_i, E[2] = U_E, E[3] = T_pare, E[4] = T_pari
    std::vector<double> energy_vector( ) const{
        return m_evec;
    }
    /**
     * @brief
     \f[
     \begin{align}
\sum_z \int_V d^3 x \left[ T(1+\ln N)\Lambda_N + q\psi\Lambda_N + N U\Lambda_U + \frac{1}{2}mU^2\Lambda_N \right] ,
\end{align}
\f]
     * @return Total energy diffusion
     */
    double energy_diffusion( )const{
        return m_ediff;
    }

    /**
     * @brief
     \f[
     \begin{align}
\int_V d^3 x \left[ (1+\ln N)\Delta_\parallel N \right] = - \int_V d^3x \frac{(\nabla_\parallel N)^2}{N} ,
\end{align}
\f]
     * @return energy loss by parallel electron diffusion
     */
    double fieldalignment() const {
        return m_aligned;
    }

  private:
    //extrapolates and solves for phi[1],
    //then adds square velocity ( omega)
    void compute_phi( const std::array<container,2>& densities);
    struct ComputePerpDrifts{
        ComputePerpDrifts( double mu, double tau):m_mu(mu), m_tau(tau){}
        DG_DEVICE
        void operator()(
                double tilde_N, double d0N, double d1N, double d2N,
                double U,       double d0U, double d1U, double d2U,
                double d0P, double d1P, double d2P,
                double b_0,         double b_1,         double b_2,
                double curv0,       double curv1,       double curv2,
                double curvKappa0,  double curvKappa1,  double curvKappa2,
                double divCurvKappa,
                double& dtN, double& dtU
            )
        {
            double N = tilde_N + 1.;
            double KappaU = curvKappa0*d0U+curvKappa1*d1U+curvKappa2*d2U;
            double KappaN = curvKappa0*d0N+curvKappa1*d1N+curvKappa2*d2N;
            double KappaP = curvKappa0*d0P+curvKappa1*d1P+curvKappa2*d2P;
            double KU = curv0*d0U+curv1*d1U+curv2*d2U;
            double KN = curv0*d0N+curv1*d1N+curv2*d2N;
            double KP = curv0*d0P+curv1*d1P+curv2*d2P;
            dtN =
                -b_0*( d1P*d2N-d2P*d1N)
                -b_1*( d2P*d0N-d0P*d2N)
                -b_2*( d0P*d1N-d1P*d0N) //ExB drift
                -m_tau*( KN)
                -N*(     KP)
                -m_mu*U*U* (   KappaN )
                -2.*m_mu*N*U*( KappaU )
                -m_mu*N*U*U*divCurvKappa;
            dtU =
                -b_0*( d1P*d2U-d2P*d1U)
                -b_1*( d2P*d0U-d0P*d2U)
                -b_2*( d0P*d1U-d1P*d0U)
                -U*KappaP
                -m_tau* KU
                -m_tau*U*divCurvKappa
                -(2.*m_tau + m_mu*U*U)*( KappaU )
                - 2.*m_tau*U*( KappaN )/N;
        }
        private:
        double m_mu, m_tau;
    };
    struct ComputeChi{
        DG_DEVICE
        void operator() ( double& chi, double tilde_Ni, double binv, double mu_i) const{
            chi = mu_i*(tilde_Ni+1.)*binv*binv;
        }
    };
    struct ComputePsi{
        DG_DEVICE
        void operator()( double& GammaPhi, double dxPhi, double dyPhi, double dzPhi, double& GdxPhi, double GdyPhi, double GdzPhi, double binv) const{
            //u_E^2
            GdxPhi   = (dxPhi*GdxPhi + dyPhi*GdyPhi + dzPhi*GdzPhi)*binv*binv;
            //Psi
            GammaPhi = GammaPhi - 0.5*GdxPhi;
        }
    };
    struct ComputeDiss{
        ComputeDiss( double mu, double tau):m_mu(mu), m_tau(tau){}
        DG_DEVICE
        void operator()( double& energy, double logN, double phi, double U, double mu, double tau) const{
            energy = tau*(1.+logN) + phi + 0.5*mu*U*U;
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
        void operator()( double& result, double tilde_n, double profne, double source, double omega_source) const{
            double temp = omega_source*source*(profne - (tilde_n+1.));
            if ( temp > 0 )
                result = temp;
            else
                result = 0.;

        }
    };

    container m_chi, m_omega, m_lambda;//helper variables

    //these should be considered const
    std::array<container,3> m_curv, m_curvKappa, m_b;
    container m_divCurvKappa;
    container m_binv, m_gradlnB;
    container m_source, m_profne;

    std::array<container,2> m_phi, m_dxPhi, m_dyPhi, m_dzPhi;
    std::array<container,2> m_npe, m_logn, m_dxN, m_dyN, m_dzN,
    std::array<container,2> m_dxU, m_dyU, m_dzU;

    //matrices and solvers
    dg::geo::DS<Geometry, IMatrix, Matrix, container> m_dsDIR, m_dsN;
    Matrix m_dx, m_dxDIR, m_dy, m_dyDIR, m_dz;
    dg::Elliptic3d<   Geometry, Matrix, container> m_lapperpN, m_lapperpDIR;
    std::vector<container> m_multi_chi;
    std::vector<dg::Elliptic3d< Geometry, Matrix, container> > m_multi_pol;
    std::vector<dg::Helmholtz3d<Geometry, Matrix, container> > m_multi_invgammaDIR,
        m_multi_invgammaN;

    dg::MultigridCG2d<Geometry, Matrix, container> m_multigrid;
    dg::Extrapolation<container> m_old_phi, m_old_psi, m_old_gammaN;

    //metric and volume elements
    container m_vol3d;
    dg::SparseTensor<container> m_metric;

    const feltor::Parameters m_p;
    const dg::geo::solovev::Parameters m_gp;

    double m_mass, m_energy, m_ediff, m_aligned;
    std::vector<double> m_evec;
};
///@}

///@cond
template<class Grid, class IMatrix, class Matrix, class container>
Explicit<Grid, IMatrix, Matrix, container>::Explicit( const Grid& g, feltor::Parameters p, dg::geo::solovev::Parameters gp):
    m_dsDIR( dg::geo::createSolovevField(gp), g, dg::DIR, dg::DIR, dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), dg::forward, gp.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz ),
    m_dsN( dg::geo::createSolovevField(gp), g, g.bcx(), g.bcy(), dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), dg::forward, gp.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz),
    /////////the poisson operators ////////////////////////////////////////
    m_dx( dg::create::dx( g, p.bc) ), m_dxDIR( dg::create::dx( g, dg::DIR) ),
    m_dy( dg::create::dy( g, p.bc) ), m_dyDIR( dg::create::dy( g, dg::DIR) ),
    m_dz( dg::create::dz( g, dg::PER) ),
    /////////the elliptic and Helmholtz operators//////////////////////////
    m_lapperpN (     g, g.bcx(), g.bcy(),   dg::normed,        dg::centered),
    m_lapperpDIR (   g, dg::DIR, dg::DIR,   dg::normed,        dg::centered),
    m_multigrid( g, m_p.stages),
    m_old_phi( 2, dg::evaluate( dg::zero, g)),
    m_old_psi( 2, dg::evaluate( dg::zero, g)),
    m_old_gammaN( 2, dg::evaluate( dg::zero, g)),
    m_p(p), m_gp(gp), m_evec(5)
{
    ////////////////////////////init temporaries///////////////////
    dg::assign( dg::evaluate( dg::zero, g), m_chi );
    m_omega = m_lambda = m_chi;
    m_phi[0] = m_phi[1] = m_chi;
    m_dxPhi = m_dyPhi = m_dzPhi = m_npe = m_logn = m_dxN = m_dyN = m_dzN = m_phi;
    m_dxU = m_dyU = m_dzU = m_phi;
    //////////////////////////////init fields /////////////////////
    dg::assign(  dg::pullback(dg::geo::InvB(mag),      g), m_binv);
    dg::assign(  dg::pullback(dg::geo::GradLnB(mag),   g), m_gradlnB);
    dg::assign(  dg::pullback(dg::geo::TanhSource(mag.psip(), gp.psipmin, gp.alpha),         g), m_source);
    ////////////////////////////transform curvature components////////
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    const dg::geo::BinaryVectorLvl0 bhat, curvNabla, curvKappa;
    if( p.curvmode == "true")
    {
        bhat = dg::geo::createBHat(mag);
        curvNabla = dg::geo::createTrueCurvatureNablaB(mag);
        curvKappa = dg::geo::createTrueCurvatureKappa(mag);
        dg::assign(  dg::pullback(dg::geo::TrueDivCurvatureKappa(mag), g),
            m_divCurvKappa);
    }
    else if( p.curvmode == "low beta")
    {
        bhat = dg:geo::createEPhi();
        curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag);
        dg::assign(  dg::pullback(dg::zero, g), m_divCurvKappa);
    }
    else if( p.curvmode == "toroidal")
    {
        bhat = dg:geo::createEPhi();
        curvNabla = dg::geo::createCurvatureNablaB(mag);
        curvKappa = dg::geo::createCurvatureKappa(mag);
        dg::assign(  dg::pullback(dg::geo::DivCurvatureKappa(mag), g),
            m_divCurvKappa);
    }
    dg::pushForwardPerp(bhat.x(), bhat.y(), bhat.z(), m_b[0], m_b[1], m_b[2], g);
    m_metric = g.metric();
    dg::tensor::inv_multiply3d( m_metric, m_b[0], m_b[1], m_b[2],
                                          m_b[0], m_b[1], m_b[2]);
    container vol = dg::tensor::volume(m_metric);
    dg::blas1::pointwiseDivide( m_binv, vol, vol); //1/vol/B
    for( int i=0; i<3; i++)
        dg::blas1::pointwiseDot( vol, m_b[i], m_b[i]); //b_i/vol/B
    dg::pushForwardPerp(curvNabla.x(), curvNabla.y(), curvNabla.z(),
        m_curv[0], m_curv[1], m_curv[2], g);
    dg::pushForwardPerp(curvKappa.x(), curvKappa.y(), curvKappa.z(),
        m_curvKappa[0], m_curvKappa[1], m_curvKappa[2], g);
    if (p.curvmode=="low beta")
    {
        dg::blas1::copy( m_curvX, m_curvKappaX);
        dg::blas1::copy( m_curvY, m_curvKappaY);
        dg::blas1::scal( m_divCurvKappa, 0.);
    }
    dg::blas1::axpby( 1., m_curvKappa, 1., m_curv);
    //////////////////////////////init elliptic and helmholtz operators////////////
    m_multi_chi = m_multigrid.project( m_chi);
    m_multi_pol.resize(m_p.stages);
    m_multi_invgammaDIR.resize(m_p.stages);
    m_multi_invgammaN.resize(m_p.stages);
    for( unsigned u=0; u<m_p.stages; u++)
    {
        dg::SparseTensor<dg::DVec> hh = dg::geo::createProjectionTensor(
            bhat, m_multigrid.grids()[u].get());
        m_multi_pol[u].construct(           m_multigrid.grids()[u].get(),
            dg::DIR, dg::DIR, dg::not_normed, dg::centered, m_p.jfactor);
        m_multi_pol[u].set_chi( hh);
        m_multi_invgammaDIR[u].construct(   m_multigrid.grids()[u].get(),
            dg::DIR, dg::DIR, -0.5*m_p.tau[1]*m_p.mu[1], dg::centered);
        m_multi_invgammaDIR[u].elliptic().set_chi( hh);
        m_multi_invgammaN[u].construct(     m_multigrid.grids()[u].get(),
            g.bcx(), g.bcy(), -0.5*m_p.tau[1]*m_p.mu[1], dg::centered);
        m_multi_invgammaN[u].elliptic().set_chi( hh);
    }
    ///////////////////init densities//////////////////////////////
    dg::assign( dg::pullback(dg::geo::Nprofile(
        p.bgprofamp, p.nprofileamp, gp, mag.psip()),g), m_profne);
    /////////////////////init limiter in parallel derivatives/////////
    if (p.pollim==true){
        m_dsN.set_boundaries( p.bc, 0, 0);  //ds N  on limiter
        m_dsDIR.set_boundaries( dg::DIR, 0, 0); //ds psi on limiter
    }
    //////////////////////////////Metric///////////////////////////////
    dg::assign( dg::create::volume(g), m_vol3d);
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::initializene( const container& src, container& target)
{
    if (m_p.tau[1] == 0.) {
        dg::blas1::copy( src, target); //  ne-1 = N_i -1
    }
    else {  //ne-1 = Gamma (ni-1)
        std::vector<unsigned> number = m_multigrid.direct_solve(
            m_multi_invgammaN, target, src, m_p.eps_gamma);
        if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_gamma);
    }
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::compute_phi( const std::array<container,2>& y)
{
    //y[0]:= n_e - 1
    //y[1]:= N_i - 1

    //First, compute and set chi
    dg::blas1::subroutine( ComputeChi(), m_chi, y[1], m_binv, m_p.mu[1]);
    m_multigrid.project( m_chi, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_pol[u].set_chi( m_multi_chi[u]);

    //Now, compute right hand side
    if (m_p.tau[1] == 0.) {
        //compute N_i - n_e
        dg::blas1::axpby( 1., y[1], -1., y[0], m_chi);
    }
    else
    {
        //compute Gamma N_i - n_e
        m_old_gammaN.extrapolate( m_chi);
        std::vector<unsigned> numberG = m_multigrid.direct_solve(
            m_multi_invgammaN, m_chi, y[1], m_p.eps_gamma);
        m_old_gammaN.update( m_chi);
        if(  numberG[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
        dg::blas1::axpby( -1., y[0], 1., m_chi, m_chi);
    }
    //----------Invert polarisation----------------------------//
    m_old_phi.extrapolate( m_phi[0]);
    std::vector<unsigned> number = m_multigrid.direct_solve(
        m_multi_pol, m_phi[0], m_chi, m_p.eps_pol);
    m_old_phi.update( m_phi[0]);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol);
    //---------------------------------------------------------//
    //Solve for Gamma Phi
    if (m_p.tau[1] == 0.) {
        dg::blas1::copy( m_phi[0], m_phi[1]);
    } else {
        m_old_psi.extrapolate( m_phi[1]);
        std::vector<unsigned> number = m_multigrid.direct_solve(
            m_multi_invgammaDIR, m_phi[1], m_phi[0], m_p.eps_gamma);
        m_old_psi.update( m_phi[1]);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
    }
    //-------Compute Psi and derivatives
    dg::blas2::symv( m_dxDIR, m_phi[0], m_dxPhi[0]);
    dg::blas2::symv( m_dyDIR, m_phi[0], m_dyPhi[0]);
    dg::blas2::symv( m_dz,    m_phi[0], m_dzPhi[0]);
    dg::tensor::multiply3d( m_metric,
        m_dxPhi[0], m_dyPhi[0], m_dzPhi[0], m_omega, m_chi, m_lambda);
    dg::blas1::subroutine( ComputePsi(),
        m_phi[1], m_dxPhi[0], m_dyPhi[0], m_dzPhi[0],
        m_omega, m_chi, m_lambda, m_binv);
    //m_omega now contains u_E^2; also update derivatives
    dg::blas2::symv( m_dxDIR, m_phi[1], m_dxPhi[1]);
    dg::blas2::symv( m_dyDIR, m_phi[1], m_dyPhi[1]);
    dg::blas2::symv( m_dz   , m_phi[1], m_dzPhi[1]);
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::operator()( double t, const std::array<std::array<container,2>,2>& y, std::array<std::array<container,2>,2>& yp)
{
    /* y[0][0] := N_e - 1
       y[0][1] := N_i - 1
       y[1][0] := U_e
       y[1][1] := U_i
    */
    dg::Timer timer;
    timer.tic();

    //Set phi[0], phi[1], m_dxPhi, m_dyPhi, and m_omega (u_E^2)
    compute_phi( y[0]);
    //Transform n-1 to n and n to logn
    dg::blas1::subroutine( ComputeLogN(), y[0], m_npe, m_logn);

    ////////////////////ENERGETICS///////////////////////////////////////
    double z[2]    = {-1.0,1.0};
    double S[2]    = {0.0, 0.0};
    double Tpar[2] = {0.0, 0.0};
    double Dpar[4]  = {0.0, 0.0,0.0,0.0};
    double Dperp[4] = {0.0, 0.0,0.0,0.0};

    m_mass = dg::blas1::dot( m_vol3d, y[0][0]);
    //compute energies
    for(unsigned i=0; i<2; i++)
    {
        S[i]    = z[i]*m_p.tau[i]*dg::blas2::dot( m_logn[i], m_vol3d, m_npe[i]);
        dg::blas1::pointwiseDot( y[1][i], y[1][i], m_chi); //U^2
        Tpar[i] = z[i]*0.5*m_p.mu[i]*dg::blas2::dot( m_npe[i], m_vol3d, m_chi);
    }
    double Tperp = 0.5*m_p.mu[1]*dg::blas2::dot( m_npe[1], m_vol3d, m_omega);   //= 0.5 mu_i N_i u_E^2
    m_energy = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1];
    m_evec[0] = S[0], m_evec[1] = S[1], m_evec[2] = Tperp, m_evec[3] = Tpar[0], m_evec[4] = Tpar[1];
    /////////////////ENERGY DISSIPATION TERMS//////////////////////////////
    // resistive energy (quadratic current)
    dg::blas1::pointwiseDot(1., m_npe[0], y[1][1], -1., m_npe[0], y[1][0], 0., m_omega); // omega = n_e (U_i - u_e)
    double Dres = -m_p.c*dg::blas2::dot(m_omega, m_vol3d, m_omega); //- C*(N_e (U_i - U_e))^2
    // energy dissipation through diffusion
    for( unsigned i=0; i<2;i++)
    {

        dg::blas1::subroutine( ComputeDiss(m_p.mu[i], m_p.tau[i]), m_chi, m_logn[i], m_phi[i], y[1][i]); //chi = tau(1+lnN) + phi + 0.5 mu U^2
        //Compute parallel dissipation for N
        dg::blas2::symv(m_p.nu_parallel, m_dsN, y[0][i], 0.,  m_lambda); //lambda = nu_parallel Delta_s N

        Dpar[i] = z[i]*dg::blas2::dot(m_chi, m_vol3d, m_lambda); //Z*(tau (1+lnN )+psi + 0.5 mu U^2) nu_parallel *(Delta_s N)
        if( i==0) //only electrons
        {
            dg::blas1::transform( m_logn[i],m_omega, dg::PLUS<>(+1)); //omega = (1+lnN)
            m_aligned = dg::blas2::dot( m_omega, m_vol3d, m_lambda); //(1+lnN)*Delta_s N
        }
        //Compute perp dissipation for N
        dg::blas2::gemv( m_lapperpN, y[0][i], m_lambda);
        dg::blas2::gemv( m_lapperpN, m_lambda, m_omega);//Delta^2 N
        Dperp[i] = -z[i]* m_p.nu_perp*dg::blas2::dot(m_chi, m_vol3d, m_omega);

        dg::blas1::pointwiseDot( m_npe[i], y[1][i], m_omega); // omega = N U
        //Compute parallel dissipation for U
        dg::blas2::symv( m_p.nu_parallel, m_dsDIR, y[1][i], 0., m_lambda);//lambda = nu_parallel Delta_s U
        Dpar[i+2] = z[i]*m_p.mu[i]*dg::blas2::dot(m_omega, m_vol3d, m_lambda);      //Z*mu*N*U nu_parallel *( Delta_s U)
        //Compute perp dissipation  for U
        dg::blas2::gemv( m_lapperpDIR, y[1][i], m_lambda);
        dg::blas2::gemv( m_lapperpDIR, m_lambda,m_chi);//Delta^2 U
        Dperp[i+2] = -z[i]*m_p.mu[i]*m_p.nu_perp* dg::blas2::dot(m_omega, m_vol3d, m_chi);
    }
    m_ediff = Dres + Dpar[0]+Dperp[0]+Dpar[1]+Dperp[1]+Dpar[2]+Dperp[2]+Dpar[3]+Dperp[3];
    ///////////////////////////////////EQUATIONS///////////////////////////////
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        dg::blas2::gemv( m_dx, y[0][i], m_dxN[i]);
        dg::blas2::gemv( m_dy, y[0][i], m_dyN[i]);
        dg::blas2::gemv( m_dxDIR, y[1][i], m_dxU[i]);
        dg::blas2::gemv( m_dyDIR, y[1][i], m_dyU[i]);
        dg::blas2::gemv( m_dz, y[0][i], m_dzN[i]);
        dg::blas2::gemv( m_dz, y[1][i], m_dzU[i]);
        dg::blas1::subroutine( ComputePerpDrifts(m_p.mu[i], m_p.tau[i]),
            //species depdendent
            y[0][i], m_dxN[i], m_dyN[i], m_dzN[i],
            y[1][i], m_dxU[i], m_dyU[i], m_dzU[i],
            m_dxPhi[i], m_dyPhi[i], m_dzPhi[i],
            //magnetic parameters
            m_binv, m_b[0], m_b[1], m_b[2],
            m_curv[0], m_curv[1], m_curv[2],
            m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
            m_divCurvKappa, yp[0][i], yp[1][i]
        );

        ///////////parallel dynamics///////////////////////////////
        //velocity
        //Burgers term
        dg::blas1::pointwiseDot(y[1][i], y[1][i], m_omega); //U^2
        m_dsDIR.centered(-0.5, m_omega, 1., yp[1][i]); //dtU += -0.5 ds U^2
        //parallel force terms
        m_dsN.centered(-m_p.tau[i]/m_p.mu[i], m_logn[i], 1.0, yp[1][i]);
        m_dsDIR.centered(-1./m_p.mu[i], m_phi[i], 1.0, yp[1][i]);        //dtU += - tau/(hat(mu))*ds lnN - 1/(hat(mu))*ds psi

        //density
        dg::blas1::pointwiseDot(m_npe[i], y[1][i], m_chi);   // NU
        m_dsDIR.centered(-1., m_chi, 1., yp[0][i]);          // dtU += - ds NU
        dg::blas1::pointwiseDot(+1., m_chi, m_gradlnB, 1., yp[0][i]);// dtU += U N ds ln B
        //direct parallel diffusion for N and U
        dsN.ds( dg::centered, y[0][i], m_chi);
        dg::blas1::pointwiseDot( -m_p.nu_parallel, gradlnB, m_chi, 1., yp[0][i]);
        dsN.dss( m_p.nu_parallel, y[0][i], 1., yp[0][i]);

        dsDIR.ds( dg::centered, y[1][i], m_chi);
        dg::blas1::pointwiseDot( -m_p.nu_parallel, gradlnB, m_chi, 1., yp[1][i]);
        dsDIR.dss( m_p.nu_parallel, y[1][i], 1., yp[1][i]);

    }
    //Add Resistivity
    dg::blas1::subroutine( AddResistivity( m_p.c, m_p.mu),
        y[0][0], y[0][1], y[1][0], y[1][1], yp[1][0], yp[1][1]);
    //Add particle source to dtN
    dg::blas1::subroutine( ComputeSource(),
        m_lambda, y[0][0], m_profne, m_source, m_p.omega_source);
    dg::blas1::axpby( 1., m_lambda, 1.0, yp[0][0]);
    dg::blas1::axpby( 1., m_lambda, 1.0, yp[1][1]);
    //add FLR correction to dtNi
    dg::blas2::gemv( -0.5*m_p.tau[1]*m_p.mu[1], m_lapperpN, m_lambda, 1.0, yp[1][1]);

    timer.toc();
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif
    std::cout << "One rhs took "<<timer.diff()<<"s\n";
}

} //namespace feltor
