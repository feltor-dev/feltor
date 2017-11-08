#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "parameters.h"

#include "geometries/geometries.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif //DG_BENCHMARK

// #define APAR
namespace eule
{
///@addtogroup solver
///@{
/**
 * @brief Implicit (perpendicular diffusive) terms for Feltor solver
 *
 \f[
    \begin{align}
     -\nu_\perp\Delta_\perp^2 N \\
     -\nu_\perp\Delta_\perp^2 w 
     -\nu_\parallel \Delta_\parallel N
     -\nu_\parallel \Delta_\parallel w
    \end{align}
\f]
 * @tparam Matrix The Matrix class
 * @tparam container The Vector class 
 * @tparam container The container class
 */
template<class Geometry, class DS, class Matrix, class container>
struct Rolkar
{
        /**
     * @brief Construct from parameters
     *
     * @tparam Grid3d three-dimensional grid class 
     * @param g The grid
     * @param p the physics parameters
     * @param gp the geometry parameters
     */
    Rolkar( const Geometry& g, eule::Parameters p, dg::geo::solovev::GeomParameters gp, DS& dsN, DS& dsDIR):
        p(p),
        gp(gp),
        LaplacianM_perpN  ( g, g.bcx(), g.bcy(), dg::normed, dg::centered),
        LaplacianM_perpDIR( g, dg::DIR, dg::DIR, dg::normed, dg::centered),
        dsN_(dsN),
        dsDIR_(dsDIR)
    {
        using dg::geo::solovev::Psip;
        dg::blas1::transfer( dg::evaluate( dg::zero, g), temp);
        dg::blas1::transfer( dg::pullback( dg::geo::GaussianDamping<Psip>(Psip(gp), gp.psipmaxcut, gp.alpha), g), dampgauss_);
    }
        /**
     * @brief Return implicit terms
     *
     * @param x input vector (x[0] := N_e -1, x[1] := N_i-1, x[2] := w_e, x[3] = w_i)
     * @param y output vector
     */
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
           x[1] := N_i - 1
           x[2] := w_e
           x[3] := w_i
        */
        dg::blas1::axpby( 0., x, 0, y);
        double nu_parallel[] = {p.nu_parallel, p.nu_parallel, p.nu_parallel, p.nu_parallel};
        for( unsigned i=0; i<2; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perpN, x[i], temp);
            dg::blas2::gemv( LaplacianM_perpN, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
            //dissipation acts not on u!
            dg::blas2::gemv( LaplacianM_perpDIR, x[i+2], temp);
            dg::blas2::gemv( LaplacianM_perpDIR, temp, y[i+2]);
            dg::blas1::scal( y[i+2], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
            //dissipation acts not on u!
            if (p.pardiss==0) 
            {
                dg::blas2::symv(dsN_, x[i],temp);
                dg::blas1::axpby( nu_parallel[i], temp, 1., y[i]); 
                dg::blas2::symv(dsDIR_, x[i+2],temp);
                dg::blas1::axpby( nu_parallel[i+2], temp, 1., y[i+2]); 
            }
        }
        //cut contributions to boundary now with damping on all 4 quantities
        for( unsigned i=0; i<y.size(); i++)
        {
            dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
        }
    }
    /**
     * @brief Return the laplacian with dirichlet BC
     *
     * @return 
     */
    dg::Elliptic<Geometry, Matrix, container>& laplacianM() {return LaplacianM_perpDIR;}

    /**
     * @brief Model function for Inversion
     *
     * @return weights for the inversion function in
     */
    const container& weights(){return LaplacianM_perpDIR.weights();}
    /**
     * @brief Model function for Inversion
     *
     * @return preconditioner for the inversion function in
     */
    const container& precond(){return LaplacianM_perpDIR.precond();}
    /**
     * @brief Damping used in the diffusion equations
     *
     * @return Vector containing damping 
     */
  private:
    const eule::Parameters p;
    const dg::geo::solovev::GeomParameters gp;
    container temp;
    container dampgauss_;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perpN,LaplacianM_perpDIR;
    DS& dsN_,dsDIR_;

};

/**
 * @brief compute explicit terms
 *
 * @tparam Matrix matrix class to use
 * @tparam container main container to hold the vectors
 * @tparam container class of the weights
 */
template< class Geometry, class DS, class Matrix, class container >
struct Asela
{
    /**
     * @brief Construct from parameters
     *
     * @tparam Grid3d three-dimensional grid class 
     * @param g The grid
     * @param p the physics parameters
     * @param gp the geometry parameters
     */
    Asela( const Geometry& g, eule::Parameters p, dg::geo::solovev::GeomParameters gp);
    /**
     * @brief Return a ds class for evaluation purposes
     *
     * @return 
     */
    DS& ds(){return dsN_;}
    DS& dsDIR(){return dsDIR_;}
        /**
     * @brief Returns phi and psi that belong to the last solve of the polarization equation
     *
     * In a multistep scheme this corresponds to the point HEAD-1
     * unless energies() is called beforehand, then they always belong to HEAD
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
    /**
     * @brief Returns Aparallel that belongs to the last solve of the induction equation
     *
     * In a multistep scheme this corresponds to the point HEAD-1
     * unless energies() is called beforehand, then they always belong to HEAD
     * @return Aparallel is parallel vector Potential
     */
    const container& aparallel( ) const { return apar[0];}
    /**
     * @brief Returns U_e and U_i
     * @return u[0] is the electron and u[1] the ion gyro-center velocity
     */
    const std::vector<container>& uparallel( ) const { return u;}
    /**
     * @brief Given N_i-1 initialize N_e -1 such that phi=0
     *
     * @param y N_i -1 
     * @param target N_e -1
     */
    void initializene( const container& y, container& target);
    /**
     * @brief Compute explicit rhs of Feltor equations
     *
     * @param y y[0] := N_e - 1, y[1] := N_i - 1, y[2] := w_e, y[3] := w_i
     * @param yp Result
     */
    void operator()( std::vector<container>& y, std::vector<container>& yp);
    
    /**
     * @brief \f[ M := \int_V (n_e-1) dV \f]
     *
     * @return mass of plasma contained in volume
     * @note call energies() before use
     */
    double mass( ) {return mass_;}
        /**
     * @brief Do not use! Not implemented yet!
     *
     * @return 0
     */
    double mass_diffusion( ) {return diff_;}
        /**
     * @brief 
     \f[
\begin{align}
 E = \partial_t  \int_V d^3x \left[\frac{1}{2}m_e n_e u_e^2 +\frac{1}{2}m_i N_i U_i^2 + \frac{(\nabla_\perp A_\parallel)^2}{2\mu_0} + \frac{1}{2} m_i N_i\left(\frac{\nabla_\perp\phi}{B}\right)^2 + t_e n_e\ln(n_e)+T_i N_i\ln(N_i)\right] 
\end{align}
\f]

     * @return Total energy contained in volume
     * @note call energies() before use
     */
    double energy( ) {return energy_;}
     /**
     * @brief Individual energies
     *
     * @return individual energy terms in total energy
     E[0]=S_e, E[1] = S_i, E[2] = U_E, E[3] = T_pare, E[4] = T_pari
     * @note call energies() before use
     */
    std::vector<double> energy_vector( ) {return evec;}
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
    double energy_diffusion( ){ return ediff_;}
    /**
     * @brief 
     \f[
     \begin{align}
\int_V d^3 x \left[ (1+\ln N)\Delta_\parallel N \right] = - \int_V d^3x \frac{(\nabla_\parallel N)^2}{N} , 
\end{align}
\f]
     * @return energy loss by parallel electron diffusion
     */
    double fieldalignment() { return aligned_;}
    
  private:
    void vecdotnablaN(const container& x, const container& y, container& z, container& target);
    void vecdotnablaDIR(const container& x, const container& y, container& z, container& target);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation
    container& induct(const std::vector<container>& y);//solves induction equation
    double add_parallel_dynamics( std::vector<container>& y, std::vector<container>& yp);

    container chi, omega,lambda;//1d container

    container binv,  curvX, curvY, curvKappaX, curvKappaY,divCurvKappa, dslnB;
    container source, damping, one;
    container profne, profNi;
    container w3d, v3d;
    
    std::vector<container> phi,apar,dsblnB, u, u2, npe, logn,un,curvphi,curvkappaphi, dsphi, dslogn,dsun,dsu2; //2d container
    std::vector<container> poissonn,poissonu,poissonw,poissonun,poissonlogn,poissonphi,poissonu2; //2d container
    std::vector<container> dsy, curvy,curvkappay;  //4d container

    //matrices and solvers
    DS dsDIR_,dsN_;
    dg::Poisson< Geometry, Matrix, container > poissonN,poissonDIR; 
    dg::Elliptic<  Geometry, Matrix, container  > pol,lapperpN,lapperpDIR; //note the host vector
    dg::Helmholtz< Geometry, Matrix, container  > maxwell, invgammaDIR, invgammaN;
    dg::Invert<container> invert_maxwell, invert_pol, invert_invgammaN,invert_invgammaNW,invert_invgammaA, invert_invgammaPhi;

    const eule::Parameters p;
    const dg::geo::solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_, aligned_;
    std::vector<double> evec;

};
///@}

template<class Grid, class DS, class Matrix, class container>
Asela<Grid, DS, Matrix, container>::Asela( const Grid& g, Parameters p, dg::geo::solovev::GeomParameters gp): 
    dsDIR_( typename DS::FieldAligned( 
                dg::geo::Field<dg::geo::solovev::MagneticField>(
                    dg::geo::solovev::MagneticField(gp), gp.R_0
                    ), 
                g, gp.rk4eps, 
                dg::geo::PsiLimiter<dg::geo::solovev::Psip>(
                    dg::geo::solovev::Psip(gp), gp.psipmaxlim
                    ), 
                dg::DIR, (2*M_PI)/((double)p.Nz)
                ), 
            dg::geo::Field<dg::geo::solovev::MagneticField>(
                dg::geo::solovev::MagneticField(gp), gp.R_0
                ), 
            dg::normed, dg::forward ),
    dsN_( typename DS::FieldAligned(
                dg::geo::Field<dg::geo::solovev::MagneticField>(
                    dg::geo::solovev::MagneticField(gp), gp.R_0), 
                g, gp.rk4eps, 
                dg::geo::PsiLimiter<dg::geo::solovev::Psip>(
                    dg::geo::solovev::Psip(gp), gp.psipmaxlim
                    ), 
                g.bcx(), (2*M_PI)/((double)p.Nz)
                ), 
          dg::geo::Field<dg::geo::solovev::MagneticField>(
              dg::geo::solovev::MagneticField(gp), gp.R_0
              ), 
          dg::normed, dg::forward ),
    //////////the poisson operators ////////////////////////////////////////
    poissonN(g, g.bcx(), g.bcy(), dg::DIR, dg::DIR), //first N/U then phi BCC
    poissonDIR(g, dg::DIR, dg::DIR, dg::DIR, dg::DIR), //first N/U then phi BCC
    //////////the elliptic and Helmholtz operators//////////////////////////
    pol(           g, dg::DIR, dg::DIR,   dg::not_normed,    dg::centered, p.jfactor), 
    lapperpN (     g, g.bcx(), g.bcy(),   dg::normed,        dg::centered),
    lapperpDIR (   g, dg::DIR, dg::DIR,   dg::normed,        dg::centered),
    maxwell(       g, dg::DIR, dg::DIR, 1., dg::centered), //sign is already correct!
    invgammaDIR(   g, dg::DIR, dg::DIR, -0.5*p.tau[1]*p.mu[1], dg::centered),
    invgammaN(     g, g.bcx(), g.bcy(), -0.5*p.tau[1]*p.mu[1], dg::centered),
    p(p), gp(gp), evec(6)
{ 
    ////////////////////////////init temporaries///////////////////
    dg::blas1::transfer( dg::evaluate( dg::zero, g), chi ); 
    dg::blas1::transfer( dg::evaluate( dg::zero, g), omega ); 
    dg::blas1::transfer( dg::evaluate( dg::zero, g), lambda ); 
    dg::blas1::transfer( dg::evaluate( dg::one,  g), one); 
    phi.resize(2);apar.resize(2); phi[0] = phi[1] = apar[0]=apar[1] =  chi;
    dsblnB.resize(2); dsblnB[0] = dsblnB[1] = chi;
    curvphi = curvkappaphi = npe = logn = u = u2 = un =  phi;
    dsphi = dslogn = dsun = dsu2 = poissonn = poissonu =  phi;
    poissonu2 =poissonw = poissonun = poissonlogn =poissonphi  = phi;
    dsy.resize(4); dsy[0] = dsy[1] = dsy[2] = dsy[3] = chi;
    curvy = curvkappay =dsy;
    //////////////////////////init invert objects///////////////////
    invert_pol.construct(         omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_pol  ); 
    invert_maxwell.construct(     omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_maxwell ); 
    invert_invgammaN.construct(   omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_gamma); 
    invert_invgammaNW.construct(  omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_gamma); 
    invert_invgammaA.construct(   omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_gamma); 
    invert_invgammaPhi.construct( omega, p.Nx*p.Ny*p.Nz*p.n*p.n, p.eps_gamma); 
    //////////////////////////////init fields /////////////////////
    using namespace dg::geo::solovev;
    MagneticField mf(gp);
    dg::blas1::transfer(  dg::pullback(dg::geo::Field<MagneticField>(mf, gp.R_0),                     g), binv);
    dg::blas1::transfer(  dg::pullback(dg::geo::GradLnB<MagneticField>(mf, gp.R_0),                   g), dslnB);
    dg::blas1::transfer(  dg::pullback(dg::geo::TanhSource<Psip>(mf.psip, gp.psipmin, gp.alpha),      g), source);
    dg::blas1::transfer(  dg::pullback(dg::geo::GaussianDamping<Psip>(mf.psip, gp.psipmax, gp.alpha), g), damping);
    ////////////////////////////transform curvature components////////
    dg::geo::pushForwardPerp(dg::geo::CurvatureNablaBR<MagneticField>(mf, gp.R_0), dg::geo::CurvatureNablaBZ<MagneticField>(mf, gp.R_0), curvX, curvY, g);
    dg::blas1::transfer(  dg::pullback(dg::geo::DivCurvatureKappa<MagneticField>(mf, gp.R_0), g), divCurvKappa);
    if (p.curvmode==1) 
    {
        dg::geo::pushForwardPerp(dg::geo::CurvatureKappaR(), dg::geo::CurvatureKappaZ<MagneticField>(mf, gp.R_0), curvKappaX, curvKappaY, g);
        dg::blas1::axpby( 1.,curvX,1.,curvKappaX,curvX);
        dg::blas1::axpby( 1.,curvY,1.,curvKappaY,curvY);
    }
    if (p.curvmode==0) 
    {
        dg::blas1::transfer(  tempX, curvKappaX);
        dg::blas1::transfer(  tempY, curvKappaY);
        dg::blas1::axpby( 1.,curvX,1.,curvKappaX,curvX);
        dg::blas1::axpby( 1.,curvY,1.,curvKappaY,curvY);
        dg::blas1::scal(divCurvKappa,0.);
    }
    ///////////////////init densities//////////////////////////////
    dg::blas1::transfer( dg::pullback(dg::geo::Nprofile<Psip>(p.bgprofamp, p.nprofileamp, gp, mf.psip),g), profne);
    dg::blas1::transfer(  profne ,profNi);
    dg::blas1::plus( profNi, -1); 
    initializene(profNi, profne); //ne = Gamma N_i (needs Invert object)
    dg::blas1::plus( profne, +1); 
    dg::blas1::plus( profNi, +1); 
    //////////////////////////init weights////////////////////////////
    dg::blas1::transfer( dg::create::volume(g),     w3d);
    dg::blas1::transfer( dg::create::inv_volume(g), v3d);
}



//computes and modifies expy!!
template<class Geometry, class DS, class Matrix, class container>
container& Asela<Geometry, DS, Matrix, container>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);        //chi =  \mu_i (n_i-1) 
    dg::blas1::plus( chi, p.mu[1]);
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);        //chi = (\mu_i n_i ) /B^2
    pol.set_chi( chi);
    dg::blas1::pointwiseDivide(v3d,chi,omega);

    invert_invgammaN(invgammaN,chi,y[1]);             //chi= Gamma (Ni-1)
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);        //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi, w3d, omega, v3d); //Gamma n_i -ne = -nabla (chi nabla phi)
    if(  number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template<class Geometry, class DS, class Matrix, class container>
container& Asela<Geometry, DS, Matrix, container>::induct(const std::vector<container>& y)
{
    if (p.flrmode == 0)
    {
        dg::blas1::axpby( p.beta/p.mu[0], npe[0], 0., chi); //chi = beta/mu_e N_e
        dg::blas1::axpby(- p.beta/p.mu[1],  npe[1], 1., chi); //chi =beta/mu_e N_e-beta/mu_i  N_i
        maxwell.set_chi(chi);

        dg::blas1::pointwiseDot( npe[0], y[2], chi);                 //chi     = n_e w_e
        dg::blas1::pointwiseDot( npe[1], y[3], lambda);               //lambda = n_i w_i
        dg::blas1::axpby( -1.,lambda , 1., chi);  //chi = -n_i w_i + n_e w_e
        //maxwell = (lap_per - beta*(N_i/mu_i - n_e/mu_e)) A_parallel 
        //chi=n_e w_e -N_i w_i
        unsigned number = invert_maxwell( maxwell, apar[0], chi); //omega equals a_parallel
        if( number == invert_maxwell.get_max())
            throw dg::Fail( p.eps_maxwell);
    }
    if (p.flrmode == 1)
    {
        dg::blas1::axpby( p.beta/p.mu[0], npe[0], 0., chi); //chi = beta/mu_e N_e
        maxwell.set_chi(chi);

        dg::blas1::pointwiseDot( npe[1], y[3], chi);               //lambda = N_i w_i
        invert_invgammaNW(invgammaDIR,lambda,chi);             //chi= Gamma (Ni wi)
        dg::blas1::pointwiseDot( npe[0], y[2], chi);                 //chi     = n_e w_e
        dg::blas1::axpby( -1.,lambda , 1., chi);  //chi = - Gamma (n_i w_i) + n_e w_e
        //maxwell = (lap_per + beta*( n_e/mu_e)) A_parallel 
        //chi=n_e w_e -Gamma (N_i w_i )
        unsigned number = invert_maxwell( maxwell, apar[0], chi); //omega equals a_parallel
        if( number == invert_maxwell.get_max())
            throw dg::Fail( p.eps_maxwell);
    }
    return apar[0];
}
template<class Geometry, class DS, class Matrix, class container>
container& Asela<Geometry, DS, Matrix,container>::compute_psi( container& potential)
{
    invert_invgammaPhi(invgammaDIR,chi,potential);                    //chi  Gamma phi
    poissonN.variationRHS(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}

template<class Geometry, class DS, class Matrix, class container>
void Asela<Geometry, DS, Matrix, container>::initializene( const container& src, container& target)
{ 
    invert_invgammaN(invgammaN,target,src); //=ne-1 = Gamma (ni-1)    
}

template<class G, class DS, class M, class V>
double Asela<G, DS, M, V>::add_parallel_dynamics( std::vector<V>& y, std::vector<V>& yp)
{
    double z[2]     = {-1.0,1.0};
    double Dpar[4]  = {0.0, 0.0,0.0,0.0};
    double Dperp[4] = {0.0, 0.0,0.0,0.0};
    if (p.pollim==1) dsN_.set_boundaries( p.bc, 0, 0);  //ds N  on limiter
    if (p.pollim==1) dsDIR_.set_boundaries( dg::DIR, 0, 0); //ds psi on limiter    
    //Parallel dynamics
    for(unsigned i=0; i<2; i++)
    {
        //compute em Poisson bracket of ds_b
        poissonN( y[i],    apar[i],poissonn[i]);                            //- [Apar,N]_RZ (for dissi)
        poissonDIR( y[i+2],  apar[i],poissonw[i]);                          //- [Apar,w]_RZ (for dissi)
        poissonN( logn[i], apar[i],poissonlogn[i]);                         // -[Apar,logN]_RZ
        poissonDIR( u[i],   apar[i],poissonu[i]);                           // -[Apar,U]_RZ
        poissonDIR( u2[i],   apar[i],poissonu2[i]);                         // -[Apar,U^2]_RZ 
	if (p.bc==dg::NEU)
	{
	  dg::blas1::pointwiseDot(poissonu[i],npe[i],poissonu[i]);         // -N[Apar,U]_RZ 
	  dg::blas1::pointwiseDot(poissonn[i],  u[i],poissonun[i]);        //- U[Apar,N]_RZ 
	  dg::blas1::axpby(1.0,poissonu[i],1.0,poissonun[i],poissonun[i]); //- N[Apar,U]_RZ  - U[Apar,N]_RZ
	}
	if (p.bc==dg::DIR)
	{
	  poissonDIR( un[i],   apar[i],poissonun[i]);                         // -[Apar,U N]_RZ 
	}
	//multiply em Poisson bracket by 1/B
        dg::blas1::pointwiseDot(poissonlogn[i],binv, poissonlogn[i]);  //-1/B [Apar,logN]_RZ
        dg::blas1::pointwiseDot(poissonun[i],binv,poissonun[i]);       //-1/B [Apar,UN]_RZ
        dg::blas1::pointwiseDot(poissonu2[i],binv,poissonu2[i]);       //-1/B [Apar,U^2]_RZ

        //Parallel dynamics (compute ds UN, ds U^2, ds psi, ds ln N) 
        if (p.bc==dg::NEU)
	{
	  dsN_.centered(y[i], dsun[i]);                                                 // ds N
	  dg::blas1::pointwiseDot(dsun[i],u[i], dsun[i]);                               // U ds N
	  dsN_.centered(logn[i], dslogn[i]);                                            // ds log N
	  dsDIR_.centered(u[i], poissonu[i]);                                           // ds u
	  dg::blas1::pointwiseDot(poissonu[i],npe[i],poissonu[i]);                      // N ds U
	  dg::blas1::axpby(1.0, poissonu[i], 1.0, dsun[i],dsun[i]);                     // ds UN = N ds U +   U ds N 
	}
	if (p.bc==dg::DIR)
	{
	    dsN_.centered(logn[i], dslogn[i]);                                           // ds log N
	    dsN_.centered(un[i], dsun[i]);                                               // ds U N
	}
        dsDIR_.centered(u2[i],dsu2[i]);                                                  // ds U^2
        dsDIR_.centered(phi[i], dsphi[i]);                                               // ds psi
        
        //add em Poisson bracket terms to the parallel derivatives to obtain ds^b   
        dg::blas1::axpby(p.beta,poissonlogn[i],1.,dslogn[i]);// ds^b logN = ds logN -beta/B [Apar,logN ]
        dg::blas1::axpby(p.beta,poissonun[i]  ,1.,dsun[i]);  // ds^b UN   = ds UN   -beta/B [Apar,UN ]
        dg::blas1::axpby(p.beta,poissonu2[i]  ,1.,dsu2[i]);  // ds^b U^2  = ds U^2  -beta/B [Apar,U^2 ] 

        //Add terms  ds^b terms       
        dg::blas1::pointwiseDot(un[i], dsblnB[i], omega);                   // U N ds^b ln B
        dg::blas1::axpby(  1. , omega, 1., yp[i]);                          // dtN = dtN + U N ds^b ln B
        dg::blas1::axpby( -1., dsun[i], 1., yp[i]);                         // dtN = dtN - ds^b U N
        dg::blas1::axpby( -p.tau[i]/p.mu[i], dslogn[i], 1., yp[2+i]);       // dtw = dtw - tau/(hat(mu))*ds^b lnN
        dg::blas1::axpby( -1./p.mu[i],dsphi[i], 1., yp[2+i]);               // dtw = dtw - 1/(hat(mu))  *ds^b psi  
        dg::blas1::axpby( -0.5, dsu2[i], 1., yp[2+i]);                      // dtw = dtw - 0.5 ds U^2

        //Parallel dissipation
        double nu_parallel[] = {p.nu_parallel, p.nu_parallel, p.nu_parallel, p.nu_parallel};
        //Compute parallel dissipation and dissipative energy for N///////////////
        if (p.pardiss==0)
        {
            dg::blas2::symv(dsN_,y[i],lambda); // lambda= ds^2 N
            dg::blas1::axpby( nu_parallel[i], lambda,  0., lambda,lambda);  //lambda = nu_parallel ds^2 N
        }
        if (p.pardiss==1)
        {
            dsN_.forward( y[i], omega); 
            dsN_.forwardTD(omega,lambda);
            dg::blas1::axpby( 0.5*nu_parallel[i], lambda, 0., lambda,lambda);  //lambda = 0.5 nu_parallel ds^2_f N
            dsN_.backward( y[i], omega); 
            dsN_.backwardTD(omega,chi);
            dg::blas1::axpby( 0.5*nu_parallel[i],chi, 1., lambda,lambda);    //lambda = 0.5 nu_parallel ds^2_f N + 0.5 nu_parallel ds^2_b N
            dg::blas1::axpby( 1., lambda, 1., yp[i]);  //add to yp //dtN += 0.5 nu_parallel ds^2_f N + 0.5 nu_parallel ds^2_b N
        }           
        //compute chi = (tau_e(1+lnN_e)+phi + 0.5 mu U^2)
        dg::blas1::axpby(1.,one,1., logn[i] ,chi); //chi = (1+lnN_e)
        dg::blas1::axpby(1.,phi[i],p.tau[i], chi); //chi = (tau_e(1+lnN_e)+phi)
        dg::blas1::pointwiseDot(y[i+2],y[i+2], omega);  
        dg::blas1::axpby(0.5*p.mu[i], omega,1., chi); //chi = (tau_e(1+lnN_e)+phi + 0.5 mu U^2)
        Dpar[i] = z[i]*dg::blas2::dot(chi, w3d, lambda); //Z*(tau (1+lnN )+psi) nu_para *(ds^2 N -ds lnB ds N)
        
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

        if (p.pardiss==0)
        {
            dg::blas2::symv(dsDIR_, y[i+2],lambda);
            dg::blas1::axpby( nu_parallel[i+2], lambda,  0., lambda,lambda); 
        }
        if (p.pardiss==1)
        {
            dsDIR_.forward( u[i], omega); 
            dsDIR_.forwardTD(omega,lambda);
            dg::blas1::axpby( 0.5*nu_parallel[i+2], lambda, 0., lambda,lambda); //lambda = 0.5 nu_parallel ds^2_f U
            dsDIR_.backward( u[i], omega); 
            dsDIR_.backwardTD(omega,chi);
            dg::blas1::axpby( 0.5*nu_parallel[i+2], chi, 1., lambda,lambda);  //lambda = 0.5 nu_parallel ds^2_f U + 0.5 nu_parallel ds^2_b U
            dg::blas1::axpby( 1., lambda, 1., yp[i+2]); //0.5 nu_parallel ds^2_f U + 0.5 nu_parallel ds^2_b U
        }   

        //compute omega = NU
        Dpar[i+2] = z[i]*p.mu[i]*dg::blas2::dot(un[i], w3d, lambda);      //Z*N*U nu_para *(ds^2 U ) or Z*N*U nu_para *(ds^2 w )

        //Compute perp dissipation  for U
        dg::blas2::gemv( lapperpDIR, y[i+2], lambda);
        dg::blas2::gemv( lapperpDIR, lambda, chi);//nabla_RZ^4 U
        Dperp[i+2] = -z[i]*p.mu[i]*p.nu_perp* dg::blas2::dot(omega, w3d, chi);

    }
    return Dpar[0]+Dperp[0]+Dpar[1]+Dperp[1]+Dpar[2]+Dperp[2]+Dpar[3]+Dperp[3];
}

// #endif
template<class Geometry, class DS, class Matrix, class container>
void Asela<Geometry, DS, Matrix, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{   
    /*  y[0] := N_e - 1
        y[1] := N_i - 1
        y[2] := w_e =  U_e + beta/mu_e Apar_e
        y[3] := w_i =  U_i + beta/mu_i Apar_i
    */
    
    dg::Timer t;
    t.tic();
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    double z[2]    = {-1.0,1.0};
    double S[2]    = {0.0, 0.0};
    double Tpar[2] = {0.0, 0.0};
    
    //compute phi via polarisation
    phi[0] = polarisation( y); //computes phi and Gamma n_i
    phi[1] = compute_psi( phi[0]); //sets omega = u_E^2
    
    //transform n-1 to n and n to logn
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+1)); //npe = N+1
        dg::blas1::transform( npe[i], logn[i], dg::LN<double>());
    }
    //compute A_parallel via induction and compute U_e and U_i from it
    if (p.beta!=0.) apar[0] = induct(y); //computes a_par and needs correct npe
    if (p.flrmode==1) invert_invgammaA(invgammaDIR,apar[1] ,apar[0] );             //chi= Gamma (Ni-1)
    if (p.flrmode==0) dg::blas1::axpby(1.0,apar[0],0.,apar[1]);

    //calculate U from Apar and w

    dg::blas1::axpby( 1., y[2], - p.beta/p.mu[0], apar[0], u[0]); // U_e = w_e -beta/mu_e Apar
    dg::blas1::axpby( 1., y[3], - p.beta/p.mu[1], apar[1], u[1]); // U_i = w_i -beta/mu_i Apar

    
    //Compute U^2  and UN and energies 
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::pointwiseDot( u[i], u[i], u2[i]);               // U^2
        dg::blas1::pointwiseDot( npe[i], u[i], un[i]);             // UN
        S[i]    = z[i]*p.tau[i]*dg::blas2::dot( logn[i], w3d, npe[i]); // S = Z tau N logN
        Tpar[i] = 0.5*z[i]*p.mu[i]*dg::blas2::dot( npe[i], w3d, u2[i]); //Tpar = 0.5 Z mu N U^2  
    }
    mass_ = dg::blas2::dot( one, w3d, y[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( npe[1], w3d, omega);   // Tperp = 0.5 mu_i N_i u_E^2
    poissonN.variationRHS( apar[0], omega); // |nabla_\perp Aparallel|^2 
    double Uapar = 0.5*p.beta*dg::blas2::dot( one, w3d, omega); // Uapar = 0.5 beta |nabla_\perp Aparallel|^2
    energy_ = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1]; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp, evec[3] = Tpar[0], evec[4] = Tpar[1]; evec[5] = Uapar;
    //// the resistive dissipative energy
    dg::blas1::pointwiseDot( npe[0], u[0], chi); //N_e U_e 
    dg::blas1::pointwiseDot( 1., npe[1], u[1], -1., chi);  //N_i U_i - N_e U_e
    dg::blas1::axpby( -1., u[0], 1., u[1], omega); //omega  = - U_e + U_i   
    double Dres = -p.c*dg::blas2::dot(omega, w3d, chi); //- C*(N_i U_i + N_e U_e)(U_i - U_e)

    for( unsigned i=0; i<2; i++)
    {
        //Here we use K(Apar) instead of K_nablaB (Apar) (?)
        vecdotnablaDIR(curvX, curvY,  apar[i], dsblnB[i]);     //K(apar) = K(apar)
        dg::blas1::axpby(  1.,  dslnB,p.beta,  dsblnB[i],dsblnB[i]);  // ds ln B + beta K(apar) 
        
        //ExB dynamics
        poissonN( y[i],   phi[i], yp[i]);                           //[N-1,phi]_RZ
        poissonDIR( y[i+2], phi[i], yp[i+2]);                       //[w,phi]_RZ 
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);               // dtN =1/B [N,phi]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);           // dtw =1/B [w,phi]_RZ  
        
        //Curvature dynamics        
        if (p.curvmode==1) 
        {
            vecdotnablaN(curvX, curvY, y[i], curvy[i]);                          //K(N) = K(N-1)
            vecdotnablaDIR(curvX, curvY,  u[i], curvy[2+i]);                     //K(U) = K(U)
            vecdotnablaDIR(curvX, curvY, phi[i], curvphi[i]);                    //K(phi)
            vecdotnablaN(curvKappaX, curvKappaY, y[i], curvkappay[i]);           //K_kappa(N) = K_kappa(N-1)
            vecdotnablaDIR(curvKappaX, curvKappaY,  u[i], curvkappay[2+i]);      //K_kappa(U)
            vecdotnablaDIR(curvKappaX, curvKappaY, phi[i], curvkappaphi[i]);     //K_kappa(phi)
        
            if (p.bc==dg::DIR)
            {
                dg::blas1::pointwiseDot(u2[i],npe[i],chi); // N U^2
                vecdotnablaN(curvKappaX, curvKappaY, chi, lambda);      //K_kappa( N U^2)
                dg::blas1::pointwiseDot(u2[i],u[i],chi); // U^3
                vecdotnablaN(curvKappaX, curvKappaY, chi, omega);      //K_kappa( U^3)

                dg::blas1::axpby( -p.mu[i], lambda, 1., yp[i]);               //dtN = dtN - (hat(mu))  K_kappa(N U^2)
                dg::blas1::axpby( -p.mu[i]/3., omega, 1., yp[2+i]);           //dtw = dtw -  (hat(mu))/3 K_kappa(U^3)
            }
            if (p.bc==dg::NEU)
            {        
		dg::blas1::pointwiseDot( u[i], curvkappay[2+i], omega);         //U K_kappa(U)
		dg::blas1::pointwiseDot( u[i], omega, chi);                     //chi = U^2 K_kappa(U)
		dg::blas1::pointwiseDot( npe[i], omega, omega);                 //omega = N U K_kappa(U) 
                dg::blas1::axpby( -2.*p.mu[i], omega, 1., yp[i]);              // dtN = dtN - 2 (hat(mu)) N U K_kappa(U)
                dg::blas1::axpby( -p.mu[i], chi, 1., yp[2+i]);                 // dtU = dtU -  (hat(mu)) U^2 K_kappa(U)
                
                dg::blas1::pointwiseDot( y[i+2], curvkappay[i], omega);        // U K_kappa( N)
                dg::blas1::pointwiseDot( y[i+2], omega, chi);                  // U^2 K_kappa( N)
                dg::blas1::axpby( -p.mu[i], chi, 1., yp[i]);                   // dtN = dtN - mu U^2 K_kappa(N)
            }
            vecdotnablaN(curvKappaX, curvKappaY, logn[i], omega);   //K_kappa(ln N)
            dg::blas1::pointwiseDot(u[i], omega, omega);            //U K_kappa(ln N)
            dg::blas1::axpby( -2.*p.tau[i], omega, 1., yp[2+i]);    //dtw = dtw - 2.*tau U K_kappa(lnN)
                
            dg::blas1::axpby( -p.tau[i], curvy[i], 1., yp[i]);         //dtN = dtN - tau K(N)
            dg::blas1::axpby( -p.tau[i], curvy[2+i], 1., yp[2+i]);     //dtw = dtw - tau K(U)
            dg::blas1::pointwiseDot(npe[i],curvphi[i], omega);         //N K(psi)
            dg::blas1::axpby( -1., omega, 1., yp[i]);                  //dtN= dtN - N K(psi)
                
            dg::blas1::pointwiseDot( u[i], curvkappaphi[i], omega);  //U K_kappa(psi)
            dg::blas1::axpby( -1., omega, 1., yp[2+i]);               //dtw = dtw -U K(psi)
            
            dg::blas1::axpby( -2.*p.tau[i], curvkappay[2+i], 1., yp[2+i]); // dtw = dtw -2.*tau K_kappa(U)
            
            //  div(K_kappa) terms
            dg::blas1::pointwiseDot(u[i],divCurvKappa,omega);              // U div(K_kappa)
            dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);                //dtw = dtw -tau U div(K_kappa)
            dg::blas1::pointwiseDot(u[i],omega,omega);                     // U^2 div(K_kappa)
            dg::blas1::pointwiseDot(npe[i],omega,omega);                     // N U^2 div(K_kappa)
            dg::blas1::axpby( -p.mu[i], omega, 1., yp[i]);                //dtw = dtw -hat(mu) N U^2 div(K_kappa)
        }
        if (p.curvmode==0) 
        {
          
            vecdotnablaN(curvX, curvY, y[i], curvy[i]);                          //K(N) = K(N-1)
            vecdotnablaDIR(curvX, curvY,  u[i], curvy[2+i]);                     //K(U) = K(U)
            vecdotnablaDIR(curvX, curvY, phi[i], curvphi[i]);                    //K(phi)
           
            dg::blas1::pointwiseDot( u2[i],curvy[2+i], chi);                //U^2 K(U)
            dg::blas1::pointwiseDot( un[i],curvy[2+i], omega);              //N U K(U)
            
            dg::blas1::axpby( -p.mu[i], omega, 1., yp[i]);                  //dtN = dtN - (hat(mu)) N U K(U)
            dg::blas1::axpby( -0.5*p.mu[i], chi, 1., yp[2+i]);              //dtw = dtw - 0.5 (hat(mu)) U^2 K(U)

            vecdotnablaN(curvX, curvY,  logn[i], omega);                          //K(N) = K(N-1)
            dg::blas1::pointwiseDot(u[i], omega, omega);                    //U K(ln N)
            dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);               //dtw = dtw - tau U K(lnN)
            
            dg::blas1::pointwiseDot( u2[i], curvy[i], omega);                //U^2 K( N)        
            dg::blas1::axpby( -0.5*p.mu[i],omega, 1., yp[i]);                //dtN = dtN - 0.5 mu U^2 K(N)

            dg::blas1::axpby( -p.tau[i], curvy[i], 1., yp[i]);              //dtN = dtN - tau K(N)
            dg::blas1::axpby( -2.*p.tau[i], curvy[2+i], 1., yp[2+i]);       //dtw = dtw - 2 tau K(U)
            dg::blas1::pointwiseDot(npe[i],curvphi[i], omega);              //N K(psi)
            dg::blas1::axpby( -1., omega, 1., yp[i]);                       //dtN= dtN - N K(psi)

            dg::blas1::pointwiseDot( u[i], curvphi[i], omega);              //U K(phi)
            dg::blas1::axpby( -0.5, omega, 1., yp[2+i]);                    //dtw = dtw -0.5 U K(psi)
        }
    }    
   
    //parallel dynamics
    double Dpar_plus_perp = add_parallel_dynamics( y, yp); 
    ediff_= Dpar_plus_perp + Dres;

    for( unsigned i=0; i<2; i++)
    {
        //Resistivity (explicit)
        dg::blas1::axpby( 1., u[1], -1, u[0], omega); //U_i - U_e
        dg::blas1::axpby( -p.c/p.mu[i], omega, 1., yp[i+2]);  //- C/mu_e (U_i - U_e)
        
        //damping 
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]);
        dg::blas1::pointwiseDot( damping, yp[i+2], yp[i+2]); 
    }
    //add particle source to dtN
    //dtN_e

    if (p.omega_source!=0.0)
    {
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
    }
    t.toc();
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif 
    std::cout << "One rhs took "<<t.diff()<<"s\n";
}

//Computes curvature operator
template<class Geometry, class DS, class Matrix, class container>
void Asela<Geometry, DS, Matrix, container>::vecdotnablaN(const container& vecX, const container& vecY, container& src, container& target)
{
    container temp1(src);
    dg::blas2::gemv( poissonN.dxlhs(), src, target); //d_R src
    dg::blas1::pointwiseDot( vecX, target, target); // C^R d_R src
    dg::blas2::gemv( poissonN.dylhs(), src, temp1);  //d_Z src
    dg::blas1::pointwiseDot( 1., vecY, temp1, 1., target);   // C^Z d_Z src + C^R d_R src
}

template<class Geometry, class DS, class Matrix, class container>
void Asela<Geometry, DS, Matrix, container>::vecdotnablaDIR(const container& vecX, const container& vecY,  container& src, container& target)
{
    container temp1(src);
    dg::blas2::gemv( poissonDIR.dxrhs(), src, target); //d_R src
    dg::blas1::pointwiseDot( vecX, target, target); // C^R d_R src
    dg::blas2::gemv( poissonDIR.dyrhs(), src, temp1);  //d_Z src
    dg::blas1::pointwiseDot( 1., vecY, temp1, 1., target);// C^Z d_Z src + C^R d_R src
}

///@endcond

} //namespace eule
