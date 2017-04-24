#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "parameters.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif //DG_BENCHMARK
/*!@file

  Contains the solvers 
  */

namespace eule
{
struct Field
{
    Field( double kappa): kappa_(kappa){}
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        yp[0] = yp[1] = 0;
        yp[2] = 1.;
    }
    void operator()( double x, double y)
    {
        return 1. + kappa_*x; 
    }
    void operator()( double x, double y, double z)
    {
        return 1. + kappa_*x; 
    }
    private:
    double kappa_;

};
///@addtogroup solver
///@{
/**
 * @brief Implicit (perpendicular diffusive) terms for Feltor solver
 *
 \f[
    \begin{align}
     -\nu_\perp\Delta_\perp^2 N \\
    \frac{C}{\mu} (U_e - U_i) - \nu_\perp\Delta_\perp^2 U   
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
    Rolkar( const Geometry& g, eule::Parameters p, DS& dsNEU, DS& dsDIR):
        p(p),
        gp(gp),
        LaplacianM_perpDIR( g, dg::DIR, dg::DIR, dg::normed, dg::centered),
        dsNEU_(dsNEU),
        dsDIR_(dsDIR)
    {
        dg::blas1::transfer( dg::evaluate( dg::zero, g), temp);
    }

    /**
     * @brief Return implicit terms
     *
     * @param x input vector (x[0] := N_e -1, x[1] := N_i-1, x[2] := U_e, x[3] = U_i)
     * @param y output vector
     */
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
           x[1] := N_i - 1
           x[2] := U_e
           x[3] := U_i
        */
        dg::blas1::axpby( 0., x, 0, y);
//         double nu_parallel[] = {-p.mu[0]/p.c, -p.mu[0]/p.c, p.nu_parallel, p.nu_parallel};
        double nu_parallel[] = {p.nu_parallel, p.nu_parallel, 0., 0.};

        for( unsigned i=0; i<2; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perpDIR, x[i], temp);
            dg::blas2::gemv( LaplacianM_perpDIR, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
            dg::blas2::gemv( LaplacianM_perpDIR, x[i+2], temp);
            dg::blas2::gemv( LaplacianM_perpDIR, temp, y[i+2]);
            dg::blas1::scal( y[i+2], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
            if (p.pardiss==0) 
            {
                dg::blas2::symv(dsNEU_, x[i],temp);
                dg::blas1::axpby( nu_parallel[i], temp, 1., y[i]); 
                //let's pray we don't need parallel diffusion on U
                //dg::blas2::symv(dsDIR_, x[i+2],temp);
                //dg::blas1::axpby( nu_parallel[i+2], temp, 1., y[i+2]); 
            }
        }
        //Resistivity
        dg::blas1::axpby( 1., x[3], -1, x[2], temp); //U_i - U_e
        dg::blas1::axpby( -p.c/p.mu[0], temp, 1., y[2]);  //- C/mu_e (U_i - U_e)
        dg::blas1::pointwiseDot( x[0], temp, temp); // N_e/N_i
        dg::blas1::pointwiseDivide( temp, x[1], temp); // N_e/N_i
        dg::blas1::axpby( -p.c/p.mu[1], temp, 1., y[3]);  //- C/mu_i (U_i - U_e)
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
    container temp;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perpDIR;
    DS& dsNEU_,dsDIR_;
};

/**
 * @brief compute explicit terms
 *
 * @tparam Geometry a 3d cartesian geometry
 * @tparam DS the class for parallel derivatives
 * @tparam Matrix matrix class to use
 * @tparam container main container to hold the vectors
 */
template< class Geometry, class DS, class Matrix, class container >
struct Feltor
{
    /**
     * @brief Construct from parameters
     *
     * @tparam Grid3d three-dimensional grid class 
     * @param g The grid
     * @param p the physics parameters
     * @param gp the geometry parameters
     */
    Feltor( const Geometry& g, eule::Parameters p);


    /**
     * @brief Return a ds class for evaluation purposes
     *
     * @return 
     */
    DS& dsNEU(){return dsNEU_;}
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
     * @brief Given N_i-1 initialize N_e -1 such that phi=0
     *
     * @param y N_i -1 
     * @param target N_e -1
     */
    void initializene( const container& y, container& target);
    /**
     * @brief Given N_e-1 initialize N_i -1 such that phi=0
     *
     * @param y N_e -1 
     * @param target N_i -1
     */
    void initializeni( const container& y, container& target);

    /**
     * @brief Compute explicit rhs of Feltor equations
     *
     * @param y y[0] := N_e - 1, y[1] := N_i - 1, y[2] := U_e, y[3] := U_i
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
 E = \partial_t \sum_z \int_V d^3x \left[\frac{1}{2}m NU^2 + \frac{(\nabla_\perp A_\parallel)^2}{2\mu_0} + \frac{1}{2}mN\left(\frac{\nabla_\perp\phi}{B}\right)^2 + T N\ln(N)\right] 
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
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation
    double add_parallel_dynamics( std::vector<container>& y, std::vector<container>& yp);

    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!

    //these should be considered const
    container binv;
    container sourceN, sourceU, one;
    container w3d, v3d;

    std::vector<container> phi, curvphi;
    std::vector<container> npe, logn;
    std::vector<container> dsy, curvy; 

    //matrices and solvers
    DS dsDIR_;
    DS dsNEU_;
    dg::Arakawa<   Geometry, Matrix, container>  poissonDIR; 
    dg::Elliptic<  Geometry, Matrix, container > pol,lapperpDIR;
    dg::Helmholtz< Geometry, Matrix, container > invgamma;

    dg::Invert<container> invert_pol,invert_invgamma,invert_invgammaPhi;

    const eule::Parameters p;

    double mass_, energy_, diff_, ediff_, aligned_;
    std::vector<double> evec;
};
///@}

///@cond
template<class Grid, class DS, class Matrix, class container>
Feltor<Grid, DS, Matrix, container>::Feltor( const Grid& g, eule::Parameters p): 
    dsDIR_( typename DS::FieldAligned( 
                Field( p.kappa), g, 1e-10, dg::DefaultLimiter(), dg::DIR, g.hz()), 
            Field(p.kappa), dg::normed, dg::forward ),
    dsNEU_( typename DS::FieldAligned(
                Field( p.kappa), g, 1e-10, dg::DefaultLimiter(gp), dg::NEU, g.hz()), 
          Field(p.kappa), dg::normed, dg::forward ),
    poissonDIR(g, dg::DIR, dg::DIR), 
    //////////the elliptic and Helmholtz operators//////////////////////////
    pol(          g, dg::DIR, dg::DIR,   dg::not_normed,    dg::centered), 
    lapperpDIR(   g, dg::DIR, dg::DIR,   dg::normed,        dg::centered),
    invgamma(    g, dg::DIR, dg::DIR, -0.5*p.tau[1]*p.mu[1], dg::centered),
    p(p), gp(gp), evec(5)
{
    ////////////////////////////init temporaries///////////////////
    dg::blas1::transfer( dg::evaluate( dg::zero, g), chi ); 
    dg::blas1::transfer( dg::evaluate( dg::zero, g), omega ); 
    dg::blas1::transfer( dg::evaluate( dg::zero, g), lambda ); 
    dg::blas1::transfer( dg::evaluate( dg::one,  g), one);
    phi.resize(2); phi[0] = phi[1] = chi;
    curvphi = npe = logn = phi;
    dsy.resize(4); dsy[0] = dsy[1] = dsy[2] = dsy[3] = chi;
    curvy = dsy;
    //////////////////////////init invert objects///////////////////
    invert_pol.construct(        omega, omega.size(), p.eps_pol  ); 
    invert_invgamma.construct(   omega, omega.size(), p.eps_gamma); 
    //////////////////////////////init fields /////////////////////
    dg::blas1::transfer(  dg::pullback(dg::LinearX( p.kappa,1.),     g), binv);
    dg::blas1::transfer(  dg::pullback(dg::CONSTANT( 2.*sqrt( 1.+p.tau)/p.lpar g), sourceN);
    dg::blas1::transfer(  dg::pullback(dg::LinearZ( 4*(1+p.tau)/p.lpar/p.lpar, 0.), g), sourceU);
    //////////////////////////init weights////////////////////////////
    dg::blas1::transfer( dg::create::volume(g),     w3d);
    dg::blas1::transfer( dg::create::inv_volume(g), v3d);
}

template<class Geometry, class DS, class Matrix, class container>
container& Feltor<Geometry, DS, Matrix, container>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-1) 
    dg::blas1::plus( chi, p.mu[1]);
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
    pol.set_chi( chi);
 
    invert_invgamma(invgamma,chi,y[1]);           //omega= Gamma (Ni-1)    
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);       //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);//Gamma n_i -ne = -nabla chi nabla phi
    if(  number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template< class Geometry, class DS, class Matrix, class container>
container& Feltor<Geometry, DS, Matrix,container>::compute_psi( container& potential)
{
    invert_invgammaPhi(invgamma,chi,potential);                    //chi  Gamma phi
    poissonDIR.variationRHS(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}

template<class Geometry, class DS, class Matrix, class container>
void Feltor<Geometry, DS, Matrix, container>::initializene( const container& src, container& target)
{ 
    invert_invgamma(invgamma,target,src); //=ne-1 = Gamma (ni-1)    
}
template<class Geometry, class DS, class Matrix, class container>
void Feltor<Geometry, DS, Matrix, container>::initializeni( const container& src, container& target)
{ 
    invert_invgamma(invgamma,src target); //=ni = Gamma^-1 (ne)    
    dg::blas1::pointwiseDot(  v3d, target, target);
}


template<class G, class DS, class M, class V>
double Feltor<G, DS, M, V>::add_parallel_dynamics( std::vector<V>& y, std::vector<V>& yp)
{
    double z[2]    = {-1.0,1.0};
    double Dpar[4] = {0.0, 0.0,0.0,0.0};
    double Dperp[4] = {0.0, 0.0,0.0,0.0};
    if (p.pollim==1) dsN_.set_boundaries( p.bc, 0, 0);  //ds N  on limiter
    if (p.pollim==1) dsDIR_.set_boundaries( dg::DIR, 0, 0); //ds psi on limiter
    //Parallel dynamics
    for(unsigned i=0; i<2; i++)
    {
        dsN_.centered(y[i], chi);   
        dg::blas1::pointwiseDot(y[i+2], chi, omega);     // U ds N
        dsDIR_.centered(y[i+2], chi);  
        dg::blas1::pointwiseDot(npe[i], chi,chi);     // N ds U
        dg::blas1::axpby(1.0,chi,1.0,omega,chi); //ds U N
        dg::blas1::pointwiseDot(npe[i], y[i+2], omega);     // U N
        dg::blas1::pointwiseDot(omega, gradlnB, omega);     // U N ds ln B
        dg::blas1::axpby( -1., chi, 1., yp[i]);             // dtN = dtN - ds U N
        dg::blas1::axpby( 1., omega, 1., yp[i]);            // dtN = dtN + U N ds ln B

        dg::blas1::pointwiseDot(y[i+2],y[i+2], omega);      //U^2
        dsDIR_.centered(omega, chi);                                 //ds U^2
        dg::blas1::axpby( -0.5, chi, 1., yp[2+i]);          //dtU = dtU - 0.5 ds U^2
        //parallel force terms
        dsN_.centered(logn[i], omega);                                                //ds lnN
        dg::blas1::axpby( -p.tau[i]/p.mu[i], omega, 1., yp[2+i]); //dtU = dtU - tau/(hat(mu))*ds lnN

        dsDIR_.centered(phi[i], omega);                                             //ds psi
        dg::blas1::axpby( -1./p.mu[i], omega, 1., yp[2+i]);   //dtU = dtU - 1/(hat(mu))  *ds psi  

    }
    //Parallel dissipation
//     double nu_parallel[] = {-p.mu[0]/p.c, -p.mu[0]/p.c, p.nu_parallel, p.nu_parallel};
    double nu_parallel[] = {p.nu_parallel, p.nu_parallel, p.nu_parallel, p.nu_parallel};
    for( unsigned i=0; i<2;i++)
    {
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
            dsDIR_.forward( y[i+2], omega); 
            dsDIR_.forwardTD(omega,lambda);
            dg::blas1::axpby( 0.5*nu_parallel[i+2], lambda, 0., lambda,lambda); //lambda = 0.5 nu_parallel ds^2_f U
            dsDIR_.backward( y[i+2], omega); 
            dsDIR_.backwardTD(omega,chi);
            dg::blas1::axpby( 0.5*nu_parallel[i+2], chi, 1., lambda,lambda);  //lambda = 0.5 nu_parallel ds^2_f U + 0.5 nu_parallel ds^2_b U
            dg::blas1::axpby( 1., lambda, 1., yp[i+2]); //0.5 nu_parallel ds^2_f U + 0.5 nu_parallel ds^2_b U
        }   

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


template<class Geometry, class DS, class Matrix, class container>
void Feltor<Geometry, DS, Matrix, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    /* y[0] := N_e - 1
       y[1] := N_i - 1
       y[2] := U_e - U_{bg}
       y[3] := U_i - U_{bg}
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    //compute phi via polarisation
    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]); //sets omega

    //compute n and logn and energies
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+1)); //npe = N+1
        dg::blas1::transform( npe[i], logn[i], dg::LN<double>());
    }
    //compute energies

    double z[2]    = {-1.0,1.0};
    double S[2]    = {0.0, 0.0};
    double Tpar[2] = {0.0, 0.0};
    //transform compute n and logn and energies
    for(unsigned i=0; i<2; i++)
    {
        S[i]    = z[i]*p.tau[i]*dg::blas2::dot( logn[i], w3d, npe[i]);
        dg::blas1::pointwiseDot( y[i+2], y[i+2], chi); 
        Tpar[i] = z[i]*0.5*p.mu[i]*dg::blas2::dot( npe[i], w3d, chi);
    }
    mass_ = dg::blas2::dot( one, w3d, y[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( npe[1], w3d, omega);   //= 0.5 mu_i N_i u_E^2
    energy_ = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1]; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp, evec[3] = Tpar[0], evec[4] = Tpar[1];
    //// the resistive dissipative energy
    dg::blas1::pointwiseDot( npe[0], y[2], chi); //N_e U_e 
    dg::blas1::pointwiseDot( 1., npe[1], y[3], -1., chi);  //N_i U_i - N_e U_e
    dg::blas1::axpby( -1., y[2], 1., y[3], omega); //omega  = - U_e + U_i   
    double Dres = -p.c*dg::blas2::dot(omega, w3d, chi); //- C*(N_i U_i + N_e U_e)(U_i - U_e)
    
    for( unsigned i=0; i<2; i++)
    {
        //ExB dynamics
        poissonN( y[i], phi[i], yp[i]);  //[N-1,phi]_RZ
        poissonDIR( y[i+2], phi[i], yp[i+2]);//[U,phi]_RZ  
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);                        // dtN =1/B [N,phi]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);                    // dtU =1/B [U,phi]_RZ  
        
        //Curvature dynamics: 
        curveN( y[i], curvy[i]);                                       //K(N) = K(N-1)
        curveDIR( y[i+2], curvy[2+i]);                                 //K(U) 
        curveDIR( phi[i], curvphi[i]);                                 //K(phi) 
        
        dg::blas1::pointwiseDot( y[i+2], curvy[2+i], omega);             //U K(U) 
        dg::blas1::pointwiseDot( y[i+2], omega, chi);                   //U^2 K(U)
        dg::blas1::pointwiseDot( npe[i], omega, omega);                 //N U K(U)
        
        dg::blas1::axpby( -p.mu[i], omega, 1., yp[i]);    //dtN = dtN - (hat(mu)) N U K(U)
        dg::blas1::axpby( -0.5*p.mu[i], chi, 1., yp[2+i]);//dtU = dtU - 0.5 (hat(mu)) U^2 K(U)

        curveN( logn[i], omega);                           //K(ln N) 
        dg::blas1::pointwiseDot(y[i+2], omega, omega);       //U K(ln N)
        dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);    //dtU = dtU - tau U K(lnN)
        
        dg::blas1::pointwiseDot( y[i+2], curvy[i], omega);   //U K( N)
        dg::blas1::pointwiseDot( y[i+2], omega, chi);        //U^2K( N)
        dg::blas1::axpby( -0.5*p.mu[i], chi, 1., yp[i]);     //dtN = dtN - 0.5 mu U^2 K(N)

        dg::blas1::axpby( -p.tau[i], curvy[i], 1., yp[i]);         //dtN = dtN - tau K(N)
        dg::blas1::axpby( -2.*p.tau[i], curvy[2+i], 1., yp[2+i]);  //dtU = dtU - 2 tau K(U)
        dg::blas1::pointwiseDot(npe[i],curvphi[i], omega);         //N K(psi)
        dg::blas1::axpby( -1., omega, 1., yp[i]);                  //dtN= dtN - N K(psi)

        dg::blas1::pointwiseDot( y[i+2], curvphi[i], omega);       //U K(phi)
        dg::blas1::axpby( -0.5, omega, 1., yp[2+i]);               //dtU = dtU -0.5 U K(psi)
    }
    //parallel dynamics
    double Dpar_plus_perp = add_parallel_dynamics( y, yp);
    ediff_= Dpar_plus_perp + Dres;
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

    t.toc();
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif 
    std::cout << "One rhs took "<<t.diff()<<"s\n";


}



///@endcond
} //namespace eule

