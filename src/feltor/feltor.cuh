#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "geometries/geometries.h"

namespace feltor
{
///@addtogroup solver
///@{
//Resistivity (consistent density dependency, parallel momentum conserving, quadratic current energy conservation dependency)
struct AddResistivity{
    AddResistivity( double C, std::array<double,2> mu): m_C(C), m_mu(mu){
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

/**
 * @brief Implicit (perpendicular diffusive) terms for Explicit solver
 *
 \f[
    \begin{align}
     -\nu_\perp\Delta_\perp^2 N \\
    \frac{C}{\mu} (U_e - U_i) - \nu_\perp\Delta_\perp^2 U
    \end{align}
\f]
 */
template<class Geometry, class IMatrix, class Matrix, class container>
struct Implicit
{

    Implicit( const Geometry& g, feltor::Parameters p, dg::geo::solovev::Parameters gp, dg::geo::DS<Geometry, IMatrix, Matrix, container>& dsN, dg::geo::DS<Geometry, IMatrix, Matrix,  container>& dsDIR):
        p(p),
        gp(gp),
        LaplacianM_perpN  ( g, g.bcx(), g.bcy(), dg::normed, dg::centered),
        LaplacianM_perpDIR( g, dg::DIR, dg::DIR, dg::normed, dg::centered),
        m_dsN(dsN),
        m_dsDIR(dsDIR)
    {
        dg::blas1::transfer( dg::evaluate( dg::zero, g), m_temp);
    }

    void operator()( double t, const std::array<std::array<container,2>,2>& x, std::array<std::array<container,2>,2>& y)
    {
        /* x[0][0] := N_e - 1
           x[0][1] := N_i - 1
           x[1][0] := U_e
           x[1][1] := U_i
        */
        for( unsigned i=0; i<2; i++)
        {
            //perpendicular hyperdiffusion for N and U
            dg::blas2::gemv( LaplacianM_perpN,   x[0][i],    m_temp);
            dg::blas2::gemv( -p.nu_perp, LaplacianM_perpN,   m_temp, 0., y[0][i]);
            dg::blas2::gemv( LaplacianM_perpDIR, x[1][i],    m_temp);
            dg::blas2::gemv( -p.nu_perp, LaplacianM_perpDIR, m_temp, 0., y[1][i]);
            //parallel diffusion for N and U
            dg::blas2::symv( p.nu_parallel, m_dsN,   x[0][i], 1., y[0][i]);
            dg::blas2::symv( p.nu_parallel, m_dsDIR, x[1][i], 1., y[1][i]);
        }
        //Resistivity is not linear!
        //dg::blas1::subroutine( AddResistivity( p.c, p.mu), x[0][0], x[0][1], x[1][0], x[1][1], y[1][0], y[1][1]);
    }

    dg::Elliptic<Geometry, Matrix, container>& laplacianM() {
        return LaplacianM_perpDIR;
    }

    const container& weights() const{
        return LaplacianM_perpDIR.weights();
    }
    const container& inv_weights() const {
        return LaplacianM_perpDIR.inv_weights();
    }
    const container& precond() const {
        return LaplacianM_perpDIR.precond();
    }

  private:
    const feltor::Parameters p;
    const dg::geo::solovev::Parameters gp;
    container m_temp;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perpN, LaplacianM_perpDIR;
    dg::geo::DS<Geometry, IMatrix, Matrix, container> m_dsN, m_dsDIR;
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

    /**
     * @brief Returns phi and psi that belong to the last solve of the polarization equation
     *
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const {
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
     * @note call energies() before use
     */
    double energy( ) const {
        return m_energy;
    }

    /**
     * @brief Individual energies
     *
     * @return individual energy terms in total energy
     E[0]=S_e, E[1] = S_i, E[2] = U_E, E[3] = T_pare, E[4] = T_pari
     * @note call energies() before use
     */
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
     * @note call energies() before use
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
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    void compute_phi( const std::array<container,2>& densities);
    double add_parallel_dynamics( const std::vector<container>& y, std::vector<container>& yp);
    struct Equations{
        DG_DEVICE
        void operator()(
                double tilde_N, double dxN, double dyN, double dsN,
                double U,       double dxU, double dyU, double dsU,
                double binv,        double gradlnB,
                double curvX,       double curvY,
                double curvKappaX,  double curvKappaY, double divCurvKappa,
                double vol_perp,
                double& dtne, double& dtni, double& dtue, double& dtui
                double tau, double mu)
        {
            double N = tilde_N + 1.;
            dtN =
                -binv/vol_perp*( dxPhi*dyN-dyPhi*dxN)
                - dsN*U - dsU*N + N*U*gradlnB
                - tau*(curvX*dxN+curvY*dyN)
                -N*(curvX*dxPhi + curvY*dyPhi)
                -mu*U*U* ( curvKappaX*dxN + curvKappaY*dyN)
                -2.*mu*N*U*( curvKappaX*dxU + curvKappaY*dyU)
                -mu*N*U*U*divCurvKappa;
            dtU =
                -binv/vol_perp*( dxPhi*dyU-dyPhi*dxU)
                - dsPhi/mu - U*dsU - tau/mu*dsN/N
                -U*(curvKappaX*dxPhi + curvKappaY*dyPhi)
                -tau*( curvX*dxU + curvY*dyU)
                -tau*U*divCurvKappa
                -(2.*tau + mu*U*U)*( curvKappaX*dxU + curvKappaY*dyU)
                - 2.*tau*U*( curvKappaX*dxN + curvKappaY*dyN)/N;
        }
    };
    struct ComputeChi{
        DG_DEVICE
        void operator() ( double tilde_Ni, double& chi, double binv, double mu_i) const{
            chi = mu_i*(tilde_Ni+1.)*binv*binv;
        }
    };
    struct ComputePsi{
        DG_DEVICE
        void operator()( double& GammaPhi, double dxPhi, double dyPhi, double& GdxPhi, double GdyPhi, double binv) const{
            //u_E^2
            GdxPhi   = (dxPhi*GdxPhi + dyPhi*GdyPhi)*binv*binv
            //Psi
            GammaPhi = GammaPhi - 0.5*GdxPhi;
        }

    };

    container m_chi, m_omega, m_lambda;//helper variables

    //these should be considered const
    container m_binv, m_curvX, m_curvY, m_curvKappaX, m_curvKappaY,m_divCurvKappa;
    container m_gradlnB;
    container m_source, m_damping, m_one;
    container m_profne, m_profNi;
    container m_w3d, m_v3d;

    std::array<container,2> m_phi, m_dxPhi, m_dyPhi;
    std::array<container,2> m_npe, m_logn;
    std::array<container,2> m_dsy, m_curvy,m_curvkappay;

    //matrices and solvers
    dg::geo::DS<Geometry, IMatrix, Matrix, container> m_dsDIR, m_dsN;
    Matrix m_dxN, m_dxDIR, m_dyN, m_dyDIR;
    dg::Poisson<    Geometry, Matrix, container> m_poissonN, m_poissonDIR;
    dg::Elliptic<   Geometry, Matrix, container> m_lapperpN, m_lapperpDIR;
    std::vector<container> m_multi_chi;
    std::vector<dg::Elliptic<Geometry, Matrix, container> > m_multi_pol;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > m_multi_invgammaDIR, m_multi_invgammaN;

    dg::Invert<container> m_invert_pol, m_invert_invgamma;
    dg::MultigridCG2d<Geometry, Matrix, container> m_multigrid;
    dg::Extrapolation<container> m_old_phi, m_old_psi, m_old_gammaN;
    dg::SparseTensor<container> m_metric;

    const feltor::Parameters m_p;
    const dg::geo::solovev::Parameters m_gp;

    double m_mass, m_energy, m_diff, m_ediff, m_aligned;
    std::vector<double> m_evec;
};
///@}

///@cond
template<class Grid, class IMatrix, class Matrix, class container>
Explicit<Grid, IMatrix, Matrix, container>::Explicit( const Grid& g, feltor::Parameters p, dg::geo::solovev::Parameters gp):
    m_dsDIR( dg::geo::createSolovevField(gp), g, dg::DIR, dg::DIR, dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), dg::normed, dg::forward, gp.rk4eps, 10, 10, true, true,  true, 2.*M_PI/(double)p.Nz ),
    m_dsN( dg::geo::createSolovevField(gp), g, g.bcx(), g.bcy(), dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), dg::normed, dg::forward, gp.rk4eps, 10, 10, true, true,  true, 2.*M_PI/(double)p.Nz),
    //////////the poisson operators ////////////////////////////////////////
    poissonN(  g, g.bcx(), g.bcy(), dg::DIR, dg::DIR), //first N/U then phi BCC
    poissonDIR(g, dg::DIR, dg::DIR, dg::DIR, dg::DIR), //first N/U then phi BCC
    //////////the elliptic and Helmholtz operators//////////////////////////
    m_lapperpN (     g, g.bcx(), g.bcy(),   dg::normed,        dg::centered),
    m_lapperpDIR (   g, dg::DIR, dg::DIR,   dg::normed,        dg::centered),
    m_multigrid( g, m_p.stages),
    m_old_phi( 2, dg::evaluate( dg::zero, g)),
    m_old_psi( 2, dg::evaluate( dg::zero, g)),
    m_old_gammaN( 2, dg::evaluate( dg::zero, g)),
    m_p(p), m_gp(gp), m_evec(5)
{
    ////////////////////////////init temporaries///////////////////
    dg::blas1::transfer( dg::evaluate( dg::zero, g), m_chi );
    dg::blas1::transfer( dg::evaluate( dg::zero, g), m_omega );
    dg::blas1::transfer( dg::evaluate( dg::zero, g), m_lambda );
    dg::blas1::transfer( dg::evaluate( dg::one,  g), m_one);
    phi.resize(2); phi[0] = phi[1] = m_chi;
    curvphi = curvkappaphi = npe = logn =  phi;
    dsy.resize(4); dsy[0] = dsy[1] = dsy[2] = dsy[3] = chi;
    curvy = curvkappay =dsy;
    //////////////////////////init invert objects///////////////////
    invert_pol.construct(        omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_pol  );
    invert_invgamma.construct(   omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_gamma);
    //////////////////////////////init elliptic and helmholtz operators////////////
    multi_chi = multigrid.project( chi);
    multi_pol.resize(m_p.stages);
    multi_invgammaDIR.resize(m_p.stages);
    multi_invgammaN.resize(m_p.stages);
    for( unsigned u=0; u<m_p.stages; u++)
    {
        multi_pol[u].construct(           multigrid.grids()[u].get(), dg::DIR, dg::DIR, dg::not_normed, dg::centered, p.jfactor);
        multi_invgammaDIR[u].construct(   multigrid.grids()[u].get(), dg::DIR, dg::DIR, -0.5*p.tau[1]*p.mu[1], dg::centered);
        multi_invgammaN[u].construct(     multigrid.grids()[u].get(), g.bcx(), g.bcy(), -0.5*p.tau[1]*p.mu[1], dg::centered);
    }
    //////////////////////////////init fields /////////////////////
    dg::geo::TokamakMagneticField mf = dg::geo::createSolovevField(gp);
    dg::blas1::transfer(  dg::pullback(dg::geo::InvB(mf),      g), binv);
    dg::blas1::transfer(  dg::pullback(dg::geo::GradLnB(mf),   g), gradlnB);
    dg::blas1::transfer(  dg::pullback(dg::geo::TanhSource(mf.psip(), gp.psipmin, gp.alpha),         g), source);
    dg::blas1::transfer(  dg::pullback(dg::geo::GaussianDamping(mf.psip(), gp.psipmax, gp.alpha), g), damping);
    ////////////////////////////transform curvature components////////
    dg::pushForwardPerp(dg::geo::CurvatureNablaBR(mf), dg::geo::CurvatureNablaBZ(mf), curvX, curvY, g);
    dg::blas1::transfer(  dg::pullback(dg::geo::DivCurvatureKappa(mf), g), divCurvKappa);
    dg::pushForwardPerp(dg::geo::CurvatureKappaR(), dg::geo::CurvatureKappaZ(mf), curvKappaX, curvKappaY, g);
    if (p.curvmode==0)
    {
        dg::blas1::transfer(  curvX, curvKappaX);
        dg::blas1::transfer(  curvY, curvKappaY);
        dg::blas1::scal(divCurvKappa,0.);
    }
    dg::blas1::axpby( 1.,curvX,1.,curvKappaX,curvX);
    dg::blas1::axpby( 1.,curvY,1.,curvKappaY,curvY);
    ///////////////////init densities//////////////////////////////
    dg::blas1::transfer( dg::pullback(dg::geo::Nprofile(p.bgprofamp, p.nprofileamp, gp, mf.psip()),g), profne);
    dg::blas1::transfer(  profne ,profNi);
    dg::blas1::plus( profNi, -1);
    initializene(profNi, profne); //ne = Gamma N_i (needs Invert object)
    dg::blas1::plus( profne, +1);
    dg::blas1::plus( profNi, +1);
    //////////////////////////init weights////////////////////////////
    dg::blas1::transfer( dg::create::volume(g),     w3d);
    dg::blas1::transfer( dg::create::inv_volume(g), v3d);
}

template<class Geometry, class IMatrix, class Matrix, class container>
double Explicit<Geometry, IMatrix, Matrix, container>::compute_phi( const std::array<container,2>& y)
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
        dg::blas1::axpby( 1., y[1], -1., y[0], m_chi); //chi = N_i - n_e
    } else { //solve for Gamma N_i
        m_old_gammaN.extrapolate( m_chi);
        std::vector<unsigned> numberG = m_multigrid.direct_solve( m_multi_invgammaN, m_chi, y[1], m_p.eps_gamma);
        m_old_gammaN.update( m_chi);
        if(  numberG[0] == m_invert_invgamma.get_max())
            throw dg::Fail( m_p.eps_gamma);
        dg::blas1::axpby( -1., y[0], 1., m_chi, m_chi); //chi= Gamma N_i - n_e
    }
    //Invert polarisation
    m_old_phi.extrapolate( m_phi[0]);
    std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_pol, m_phi[0], m_chi, m_p.eps_pol);
    m_old_phi.update( m_phi[0]);
    if(  number[0] == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    //Solve for Gamma Phi
    if (p.tau[1] == 0.) {
        dg::blas1::copy( m_phi[0], m_phi[1]);
    } else {
        m_old_psi.extrapolate( m_phi[1]);
        std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_invgammaDIR, m_phi[1], m_phi[0], m_p.eps_gamma);
        m_old_psi.update( m_phi[1]);
        if(  number[0] == m_invert_invgamma.get_max())
            throw dg::Fail( m_p.eps_gamma);
    }
    //Compute Psi
    blas2::symv( m_dxDIR, m_phi[0], m_dxPhi[0]);
    blas2::symv( m_dyDIR, m_phi[0], m_dyPhi[0]);
    tensor::multiply2d( m_metric, m_dxPhi[0], m_dyPhi[0], m_omega, m_chi);
    dg::blas1::subroutine( ComputePsi(), m_phi[1], m_dxPhi[0], m_dyPhi[0], m_omega, m_chi, m_binv);
    //m_omega now contains u_E^2; also update derivatives
    blas2::symv( m_dxDIR, m_phi[1], m_dxPhi[1]);
    blas2::symv( m_dyDIR, m_phi[1], m_dyPhi[1]);
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Explicit<Geometry, IMatrix, Matrix, container>::initializene( const container& src, container& target)
{
    if (p.tau[1] == 0.) {
        dg::blas1::copy( src, target); //  ne-1 = N_i -1
    }
    else {
        std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_invgammaN, target, src, m_p.eps_gamma);  //=ne-1 = Gamma (ni-1)
        if(  number[0] == m_invert_invgamma.get_max())
        throw dg::Fail( m_p.eps_gamma);
    }
}


template<class G, class IMatrix, class M, class V>
double Explicit<G, IMatrix, M, V>::add_parallel_dynamics( const std::vector<V>& y, std::vector<V>& yp)
{
    double z[2]     = {-1.0,1.0};
    double Dpar[4]  = {0.0, 0.0,0.0,0.0};
    double Dperp[4] = {0.0, 0.0,0.0,0.0};
    if (p.pollim==1) m_dsN.set_boundaries( p.bc, 0, 0);  //ds N  on limiter
    if (p.pollim==1) m_dsDIR.set_boundaries( dg::DIR, 0, 0); //ds psi on limiter
    //Parallel dynamics
    for(unsigned i=0; i<2; i++)
    {
        if (p.bc==dg::DIR)
        {
            dg::blas1::pointwiseDot(npe[i], y[i+2],chi);      // NU
            //with analytic expression
            m_dsDIR.centered(chi, omega);                      //ds NU
            dg::blas1::pointwiseDot(chi, gradlnB, chi);     // U N ds ln B
            dg::blas1::axpbypgz(-1., omega, 1., chi, 1., yp[i]);  // dtN += - ds U N +  U N ds ln B
            //Alternative: direct with adjoint derivative
//             m_dsDIR.centeredDiv(-1, chi, 1., yp[i]);     // dtN+= - ds^dagger U N
        }
        if (p.bc==dg::NEU)
        {
            m_dsN.centered(y[i], chi);
            dg::blas1::pointwiseDot(y[i+2], chi, omega);        // U ds N
            m_dsDIR.centered(y[i+2], chi);
            dg::blas1::pointwiseDot(npe[i], chi,chi);           // N ds U
            dg::blas1::axpbypgz(-1.0, chi, -1.0, omega, 1.0, yp[i]); // dtN += - ds U N
            dg::blas1::pointwiseDot(1.0, npe[i], y[i+2], gradlnB, 1.0, yp[i]); // dtN += + U N ds ln B
        }
        //Burgers term
        dg::blas1::pointwiseDot(y[i+2], y[i+2], omega);      //U^2
        m_dsDIR.centered(-0.5, omega, 1., yp[2+i]);          //dtU += - 0.5 ds U^2
        //parallel force terms
        m_dsN.centered(-p.tau[i]/p.mu[i], logn[i], 1.0, yp[2+i]);   //dtU += - tau/(hat(mu))*ds lnN
        m_dsDIR.centered(-1./p.mu[i], phi[i], 1.0, yp[2+i]);      //dtU +=  - 1/(hat(mu))  *ds psi
    }
    //Parallel dissipation
//     double nu_parallel[] = {-p.mu[0]/p.c, -p.mu[0]/p.c, p.nu_parallel, p.nu_parallel};
    double nu_parallel[] = {p.nu_parallel, p.nu_parallel, p.nu_parallel, p.nu_parallel};
    for( unsigned i=0; i<2;i++)
    {
        //Compute parallel dissipation and dissipative energy for N///////////////

        dg::blas2::symv(m_dsN,y[i],lambda); // lambda= ds^2 N
        dg::blas1::axpby( nu_parallel[i], lambda,  0., lambda,lambda);  //lambda = nu_parallel ds^2 N

        //compute chi = (tau_e(1+lnN_e)+phi + 0.5 mu U^2)
        dg::blas1::axpby(1.,one,1., logn[i] ,chi); //chi = (1+lnN_e)
        dg::blas1::pointwiseDot(y[i+2],y[i+2], omega);  //U^2
        dg::blas1::axpbypgz(0.5*p.mu[i], omega, 1.0, phi[i], p.tau[i], chi); //chi = (tau (1+lnN_e) + psi + 0.5 mu U^2)
        Dpar[i] = z[i]*dg::blas2::dot(chi, w3d, lambda); //Z*(tau (1+lnN )+psi + 0.5 mu U^2) nu_para *(ds^2 N -ds lnB ds N)
        if( i==0) //only electrons
        {
            //do not write into chi
            dg::blas1::axpby(1.,one,1., logn[i] ,omega); //omega = (1+lnN)
            aligned_ = dg::blas2::dot( omega, w3d, lambda); //(1+lnN)*Delta_s N
        }

        //Compute perp dissipation for N
        dg::blas2::gemv( lapperpN, y[i], lambda);
        dg::blas2::gemv( lapperpN, lambda, omega);//nabla_RZ^4 N_e
        Dperp[i] = -z[i]* p.nu_perp*dg::blas2::dot(chi, w3d, omega);

        dg::blas2::symv(m_dsDIR, y[i+2],lambda);
        dg::blas1::axpby( nu_parallel[i+2], lambda,  0., lambda,lambda);

        //compute omega = NU
        dg::blas1::pointwiseDot( npe[i], y[i+2], omega); //N U
        Dpar[i+2] = z[i]*p.mu[i]*dg::blas2::dot(omega, w3d, lambda);      //Z*N*U nu_para *(ds^2 U -ds lnB ds U)

        //Compute perp dissipation  for U
        dg::blas2::gemv( lapperpDIR, y[i+2], lambda);
        dg::blas2::gemv( lapperpDIR, lambda,chi);//nabla_RZ^4 U
        Dperp[i+2] = -z[i]*p.mu[i]*p.nu_perp* dg::blas2::dot(omega, w3d, chi);

    }
    return Dpar[0]+Dperp[0]+Dpar[1]+Dperp[1]+Dpar[2]+Dperp[2]+Dpar[3]+Dperp[3];
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
    //compute phi via polarisation
    compute_phi( y[0]); //set phi[0], phi[1] and m_omega (u_E^2)

    //transform n-1 to n and n to logn
    dg::blas1::transform( y[0], m_npe, dg::PLUS<>(+1));
    dg::blas1::transform( m_npe, m_logn, dg::LN<double>());

    double z[2]    = {-1.0,1.0};
    double S[2]    = {0.0, 0.0};
    double Tpar[2] = {0.0, 0.0};
    //compute energies
    for(unsigned i=0; i<2; i++)
    {
        S[i]    = z[i]*m_p.tau[i]*dg::blas2::dot( m_logn[i], m_w3d, m_npe[i]);
        dg::blas1::pointwiseDot( y[1][i], y[1][i], m_chi); //U^2
        Tpar[i] = z[i]*0.5*m_p.mu[i]*dg::blas2::dot( m_npe[i], m_w3d, m_chi);
    }
    m_mass = dg::blas1::dot( m_w3d, y[0][0]);
    double Tperp = 0.5*m_p.mu[1]*dg::blas2::dot( m_npe[1], m_w3d, m_omega);   //= 0.5 mu_i N_i u_E^2
    m_energy = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1];
    m_evec[0] = S[0], m_evec[1] = S[1], m_evec[2] = Tperp, m_evec[3] = Tpar[0], m_evec[4] = Tpar[1];
    // resistive energy (quadratic current)
    dg::blas1::pointwiseDot(1., m_npe[0], y[1][1], -1., m_npe[0], y[1][0], m_omega); // omega = n_e (U_i - u_e)
    double Dres = -m_p.c*dg::blas2::dot(m_omega, m_w3d, m_omega); //- C*(N_e (U_i - U_e))^2
    for( unsigned i=0; i<2; i++)
    {
        dg::blas2::gemv( m_dxN, y[0][i], dxN[i]);
        dg::blas2::gemv( m_dyN, y[0][i], dyN[i]);
        dg::blas2::gemv( m_dxDIR, y[1][i], dxU[i]);
        dg::blas2::gemv( m_dyDIR, y[1][i], dyU[i]);
        dg::blas1::subroutine( Equations(),                       

        poissonN( y[i], phi[i], yp[i]);  //[N-1,phi]_RZ
        poissonDIR( y[i+2], phi[i], yp[i+2]);//[U,phi]_RZ
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);                        // dtN =1/B [N,phi]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);                    // dtU =1/B [U,phi]_RZ

        //Curvature dynamics:
        if (p.curvmode==1)
        {
            vecdotnablaN(curvX, curvY, y[i], curvy[i]);            //K(N) = K(N-1)
            vecdotnablaDIR(curvX, curvY,  y[i+2], curvy[2+i]);     //K(U) = K(U)
            vecdotnablaDIR(curvX, curvY, phi[i], curvphi[i]);      //K(phi)
            vecdotnablaN(curvKappaX, curvKappaY, y[i], curvkappay[i]);            //K_kappa(N) = K_kappa(N-1)
            vecdotnablaDIR(curvKappaX, curvKappaY,  y[i+2], curvkappay[2+i]);     //K_kappa(U)
            vecdotnablaDIR(curvKappaX, curvKappaY, phi[i], curvkappaphi[i]);      //K_kappa(phi)

            dg::blas1::pointwiseDot( y[i+2], curvkappay[2+i], omega);       //omega = U K_kappa(U)
            dg::blas1::pointwiseDot( y[i+2], omega, chi);                   //chi = U^2 K_kappa(U)
            dg::blas1::pointwiseDot( npe[i], omega, omega);                 // omega= N U K_kappa(U)
            if (p.bc==dg::DIR)
            {
                dg::blas1::pointwiseDot(y[i+2],y[i+2],omega);           //omega  = U^2
                dg::blas1::pointwiseDot(omega,npe[i],chi);              //chi   = N U^2
                vecdotnablaN(curvKappaX, curvKappaY, chi, lambda);      //lambda = K_kappa( N U^2)
                dg::blas1::pointwiseDot(omega,y[i+2],chi);              // chi = U^3
                vecdotnablaN(curvKappaX, curvKappaY, chi, omega);       //omega = K_kappa( U^3)

                dg::blas1::axpby( -p.mu[i],   lambda, 1., yp[i]);       //dtN = dtN - (hat(mu))  K_kappa(N U^2)
                dg::blas1::axpby( -p.mu[i]/3., omega, 1., yp[2+i]);     //dtU = dtU -  (hat(mu))/3 K_kappa(U^3)
            }
            if (p.bc==dg::NEU)
            {
                dg::blas1::axpby( -2.*p.mu[i], omega, 1., yp[i]);              // dtN = dtN - 2 (hat(mu)) N U K_kappa(U)
                dg::blas1::axpby( -p.mu[i], chi, 1., yp[2+i]);                 // dtU = dtU -  (hat(mu)) U^2 K_kappa(U)

                dg::blas1::pointwiseDot( y[i+2], curvkappay[i], omega);        // U K_kappa( N)
                dg::blas1::pointwiseDot( y[i+2], omega, chi);                  // U^2 K_kappa( N)
                dg::blas1::axpby( -p.mu[i], chi, 1., yp[i]);                   // dtN = dtN - (hat(mu)) U^2 K_kappa(N)
            }
            vecdotnablaN(curvKappaX, curvKappaY, logn[i], omega);         //K_kappa(ln N)
            dg::blas1::pointwiseDot(y[i+2], omega, omega);                //U K_kappa(ln N)
            dg::blas1::axpbypgz( -2.*p.tau[i], omega, -p.tau[i], curvy[2+i], 1., yp[2+i]);    //dtU += - 2.*tau U K_kappa(lnN)- tau K(U)

            dg::blas1::pointwiseDot(npe[i],curvphi[i], omega);         //N K(psi)
            dg::blas1::axpbypgz(-p.tau[i], curvy[i], -1., omega, 1., yp[i]);                  //dtN+= - tau K(N) - N K(psi)

            dg::blas1::pointwiseDot( y[i+2], curvkappaphi[i], omega);  //U K_kappa(psi)
            dg::blas1::axpbypgz(-1., omega, -2.*p.tau[i], curvkappay[2+i], 1., yp[2+i]); // dtU += - U K_kappa(psi) -2.*tau K_kappa(U)
            //         div(K_kappa) terms
            dg::blas1::pointwiseDot(y[i+2],divCurvKappa,omega);              // U div(K_kappa)
            dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);                //dtU += -tau U div(K_kappa)
            dg::blas1::pointwiseDot(-p.mu[i], y[i+2], npe[i], omega, 1., yp[i]); //dtN += -hat(mu) N U^2 div(K_kappa)
        }
        if (p.curvmode==0)
        {
            vecdotnablaN(curvX, curvY, y[i], curvy[i]);                   //K(N) = K(N-1)
            vecdotnablaDIR(curvX, curvY,  y[i+2], curvy[2+i]);            //K(U) = K(U)
            vecdotnablaDIR(curvX, curvY, phi[i], curvphi[i]);             //K(phi)

            dg::blas1::pointwiseDot(-0.5*p.mu[i], y[i+2], y[i+2], curvy[2+i],1., yp[2+i]); //dtU +=- 0.5 (hat(mu)) U^2 K(U)
            dg::blas1::pointwiseDot(-p.mu[i], npe[i], y[i+2], curvy[2+i], 1., yp[i]); //dtN += - (hat(mu)) N U K(U)

            vecdotnablaN(curvX, curvY, logn[i], omega);                   //K(ln N)
            dg::blas1::pointwiseDot(-p.tau[i], y[i+2], omega, 1., yp[2+i]); //dtU += - tau U K(lnN)

            dg::blas1::pointwiseDot(-0.5*p.mu[i],  y[i+2], y[i+2], curvy[i], 1., yp[i]);  //dtN += - 0.5 mu U^2 K(N)

            dg::blas1::pointwiseDot(npe[i],curvphi[i], omega);            //N K(psi)
            dg::blas1::axpbypgz( -p.tau[i], curvy[i], -1., omega, 1., yp[i]); //dtN+= - tau K(N) - N K(psi)

            dg::blas1::pointwiseDot( y[i+2], curvphi[i], omega);          //U K(phi)
            dg::blas1::axpbypgz(-2.*p.tau[i], curvy[2+i], -0.5, omega, 1., yp[2+i]);   //dtU += - 2 tau K(U) -0.5 U K(psi)
        }
    }
    //parallel dynamics
    double Dpar_plus_perp = add_parallel_dynamics( y, yp);
    m_ediff= Dpar_plus_perp + Dres;
    for( unsigned i=0; i<2; i++)
    {
        //damping
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]);
        dg::blas1::pointwiseDot( damping, yp[i+2], yp[i+2]);

    }
    //add particle source to dtN
    //dtN_e
    dg::blas1::axpby(+1.0,profne,-1.0,npe[0],lambda);//lambda = ne0 - ne
    dg::blas1::pointwiseDot(source,lambda,omega);//tanhSource on profNe
    dg::blas1::transform(omega, omega, dg::POSVALUE<double>());
    dg::blas1::axpby(p.omega_source,omega,1.0,yp[0]);
    //dtN_i
    dg::blas1::axpby(p.omega_source,omega,1.0,yp[1]);
    //add FLR correction
    dg::blas1::pointwiseDot(source,lambda,lambda);//tanhSource on profNe
    dg::blas1::transform(lambda, lambda, dg::POSVALUE<double>()); 
    dg::blas2::gemv( lapperpN, lambda, omega); 
    dg::blas1::axpby(-p.omega_source*0.5*p.tau[1]*p.mu[1],omega,1.0,yp[1]);   

    timer.toc();
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif 
    std::cout << "One rhs took "<<timer.diff()<<"s\n";
}



///@endcond
} //namespace feltor

