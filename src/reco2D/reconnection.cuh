#pragma once

#include "dg/algorithm.h"
#include "parameters.h"

// #define APAR
namespace asela
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
template<class Geometry, class IMatrix, class Matrix, class container>
struct Implicit
{
        /**
     * @brief Construct from parameters
     *
     * @tparam Grid3d three-dimensional grid class 
     * @param g The grid
     * @param p the physics parameters
     * @param gp the geometry parameters
     */
    Implicit( const Geometry& g, asela::Parameters p ):
        p(p),
        LaplacianM_perp  ( g, g.bcx(), g.bcy(), dg::normed, dg::centered)
    {
        dg::blas1::transfer( dg::evaluate( dg::zero, g), temp);
    }
        /**
     * @brief Return implicit terms
     *
     * @param x input vector (x[0] := N_e -1, x[1] := N_i-1, x[2] := w_e, x[3] = w_i)
     * @param y output vector
     */
    void operator()( double time,  const std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
           x[1] := N_i - 1
           x[2] := w_e
           x[3] := w_i
        */
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<2; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
            //dissipation acts on w!
            dg::blas2::gemv( LaplacianM_perp, x[i+2], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i+2]);
            dg::blas1::scal( y[i+2], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ w)  
        }

    }
    /**
     * @brief Return the laplacian with dirichlet BC
     *
     * @return 
     */
    dg::Elliptic<Geometry, Matrix, container>& laplacianM() {return LaplacianM_perp;}

    const container& weights(){return LaplacianM_perp.weights();}
    const container& inv_weights(){return LaplacianM_perp.inv_weights();}
    const container& precond(){return LaplacianM_perp.precond();}
  private:
    const asela::Parameters p;
    container temp;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perp;
    
};

/**
 * @brief compute explicit terms
 *
 * @tparam Matrix matrix class to use
 * @tparam container main container to hold the vectors
 * @tparam container class of the weights
 */
template< class Geometry, class IMatrix, class Matrix, class container >
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
    Asela( const Geometry& g, asela::Parameters p);
    /**
     * @brief Returns phi and psi that belong to the last solve of the polarization equation
     *
     * In a multistep scheme this corresponds to the point HEAD-1
     * unless energies() is called beforehand, then they always belong to HEAD
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
// <<<<<<< HEAD
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
    void operator()( double time, const std::vector<container>& y, std::vector<container>& yp);
    
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

    
  private:
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation
    container& induct(const std::vector<container>& y);//solves induction equation
    double add_parallel_dynamics( const std::vector<container>& y, std::vector<container>& yp);

    container chi, omega,lambda;//1d container

    container one;
    container w2d, v2d;
    
    std::vector<container> phi,apar, u, u2, npe, logn,un; //2d container
    std::vector<container> arakawan,arakawau,arakawaun,arakawalogn,arakawaphi,arakawau2; //2d container

    //matrices and solvers
    dg::ArakawaX< Geometry, Matrix, container > arakawa; 
    dg::Variation< Geometry, Matrix, container > gradient; 
    dg::Elliptic<  Geometry, Matrix, container  > lapperp; //note the host vector    
    
    std::vector<container> multi_chi;
    std::vector<dg::Elliptic<Geometry, Matrix, container> > multi_pol;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > multi_maxwell, multi_invgamma; 
    
    dg::Invert<container> invert_maxwell, invert_pol, invert_invgamma;
    dg::MultigridCG2d<Geometry, Matrix, container> multigrid;
    dg::Extrapolation<container> old_phi, old_psi, old_gammaN, old_gammaNW, old_Apar, old_gammaApar;
    
    const asela::Parameters p;

    double mass_, energy_, diff_, ediff_, aligned_;
    std::vector<double> evec;

};
///@}

template<class Grid, class IMatrix, class Matrix, class container>
Asela<Grid, IMatrix, Matrix, container>::Asela( const Grid& g, Parameters p): 
    //////////the arakawa operators ////////////////////////////////////////
    arakawa(g, g.bcx(), g.bcy()),
    gradient(g, g.bcx(), g.bcy(), dg::centered ),
    //////////the elliptic and Helmholtz operators//////////////////////////
    lapperp (     g, g.bcx(), g.bcy(),   dg::normed,        dg::centered),
    multigrid( g, 3),
    old_phi( 2, dg::evaluate( dg::zero, g)),old_psi( 2, dg::evaluate( dg::zero, g)), old_gammaN( 2, dg::evaluate( dg::zero, g)), old_gammaNW( 2, dg::evaluate( dg::zero, g)), old_Apar( 2, dg::evaluate( dg::zero, g)), old_gammaApar( 2, dg::evaluate( dg::zero, g)), 
    p(p),  evec(6)
{ 
    ////////////////////////////init temporaries///////////////////
    dg::blas1::transfer( dg::evaluate( dg::zero, g), chi ); 
    dg::blas1::transfer( dg::evaluate( dg::zero, g), omega ); 
    dg::blas1::transfer( dg::evaluate( dg::zero, g), lambda ); 
    dg::blas1::transfer( dg::evaluate( dg::one,  g), one); 
    phi.resize(2);apar.resize(2); phi[0] = phi[1] = apar[0]=apar[1] =  chi;
    npe = logn = u = u2 = un =  phi;
    arakawan = arakawau =  phi;
    arakawau2  = arakawaun = arakawalogn =arakawaphi  = phi;
    //////////////////////////init invert objects///////////////////
    invert_pol.construct(        omega, p.Nx*p.Ny*p.n*p.n, p.eps_pol  ); 
    invert_maxwell.construct(    omega, p.Nx*p.Ny*p.n*p.n, p.eps_maxwell ); 
    invert_invgamma.construct(   omega, p.Nx*p.Ny*p.n*p.n, p.eps_gamma); 
    //////////////////////////////init elliptic and helmholtz operators////////////
    multi_chi = multigrid.project( chi);
    multi_pol.resize(3);
    multi_maxwell.resize(3);
    multi_invgamma.resize(3);
    for( unsigned u=0; u<3; u++)
    {
        multi_pol[u].construct(        multigrid.grid(u), g.bcx(), g.bcy(), dg::not_normed, dg::centered, p.jfactor);
        multi_maxwell[u].construct(    multigrid.grid(u), g.bcx(), g.bcy(), 1., dg::centered);
        multi_invgamma[u].construct(   multigrid.grid(u), g.bcx(), g.bcy(), -0.5*p.tau[1]*p.mu[1], dg::centered);
    }
    //////////////////////////init weights////////////////////////////
    dg::blas1::transfer( dg::create::volume(g),     w2d);
    dg::blas1::transfer( dg::create::inv_volume(g), v2d);
}



//computes and modifies expy!!
template<class Geometry, class IMatrix, class Matrix, class container>
container& Asela<Geometry, IMatrix, Matrix, container>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);        //chi =  \mu_i (n_i-1) 
    dg::blas1::plus( chi, p.mu[1]);

    multigrid.project( chi, multi_chi);
    for( unsigned u=0; u<3; u++)
    {
        multi_pol[u].set_chi( multi_chi[u]);
    }

    //Gamma N_i
    if (p.tau[1] == 0.) {
        dg::blas1::axpby( 1., y[1], 0.,chi); //chi = N_i - 1
    } 
    else {
        old_gammaN.extrapolate( chi);
        std::vector<unsigned> numberG = multigrid.direct_solve( multi_invgamma, chi, y[1], p.eps_gamma); //chi= Gamma (Ni-1)
        old_gammaN.update( chi);
        if(  numberG[0] == invert_invgamma.get_max())
            throw dg::Fail( p.eps_gamma);
    }
    //rhs
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);        //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_i - n_e
    //polarisation
    old_phi.extrapolate( phi[0]);
    std::vector<unsigned> number = multigrid.direct_solve( multi_pol, phi[0], chi, p.eps_pol);
    old_phi.update( phi[0]);
    if(  number[0] == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);   
    return phi[0];
}

template<class Geometry, class IMatrix, class Matrix, class container>
container& Asela<Geometry, IMatrix, Matrix, container>::induct(const std::vector<container>& y)
{

        dg::blas1::axpby( p.beta/p.mu[0], npe[0], 0., chi); //chi = beta/mu_e N_e
        //with mass correction (only consistent for cold ions)
        if (p.tau[1] == 0.) {
            dg::blas1::axpby( -p.beta/p.mu[1], npe[1], 1., chi); //chi = beta/mu_e N_e - beta/mu_i N_i 
        } 
        multigrid.project( chi, multi_chi);
        for( unsigned u=0; u<3; u++)
        {
            multi_maxwell[u].set_chi( multi_chi[u]);
        }
        dg::blas1::pointwiseDot( npe[1], y[3], chi);               //chi = N_i w_i
        if (p.tau[1] == 0.) {
            dg::blas1::axpby( 1., chi, 0.,lambda); //lambda = N_i w_i
        } 
        else {
            old_gammaNW.extrapolate( lambda);
            std::vector<unsigned> numberG = multigrid.direct_solve( multi_invgamma, lambda,chi, p.eps_gamma); //lambda= Gamma (Ni wi)
            old_gammaNW.update( lambda);
            if(  numberG[0] == invert_invgamma.get_max())
                throw dg::Fail( p.eps_gamma);
        }
        dg::blas1::pointwiseDot( npe[0], y[2], chi);                 //chi     = n_e w_e
        dg::blas1::axpby( -1.,lambda , 1., chi);  //chi = - Gamma (n_i w_i) + n_e w_e
        //maxwell = (lap_per + beta*( n_e/mu_e)) A_parallel 
        //chi=n_e w_e -Gamma (N_i w_i )
        //induction
        old_Apar.extrapolate( apar[0]);
        std::vector<unsigned> number = multigrid.direct_solve( multi_maxwell, apar[0], chi, p.eps_maxwell);
        old_Apar.update( apar[0]);
        if(  number[0] == invert_maxwell.get_max())
            throw dg::Fail( p.eps_maxwell);  
    return apar[0];
}
template<class Geometry, class IMatrix, class Matrix, class container>
container& Asela<Geometry, IMatrix, Matrix,container>::compute_psi( container& potential)
{
    if (p.tau[1] == 0.) {
        dg::blas1::axpby( 1., potential, 0., phi[1]); 
    } 
    else {
        old_psi.extrapolate( phi[1]);
        std::vector<unsigned> number = multigrid.direct_solve( multi_invgamma, phi[1], potential, p.eps_gamma);
        old_psi.update( phi[1]);
        if(  number[0] == invert_invgamma.get_max())
        throw dg::Fail( p.eps_gamma); 
    }
    gradient.variation(potential, omega); 
    dg::blas1::axpby( 1., phi[1], -0.5, omega,phi[1]);        
    return phi[1];  
}

template<class Geometry, class IMatrix, class Matrix, class container>
void Asela<Geometry, IMatrix, Matrix, container>::initializene( const container& src, container& target)
{ 
    if (p.tau[1] == 0.) {
        dg::blas1::axpby( 1.,src, 0., target); //  ne-1 = N_i -1
    } 
    else {
        std::vector<unsigned> number = multigrid.direct_solve( multi_invgamma, target,src, p.eps_gamma);  //=ne-1 = Gamma (ni-1)  
        if(  number[0] == invert_invgamma.get_max())
        throw dg::Fail( p.eps_gamma);
    }
}

template<class G, class IMatrix, class M, class V>
double Asela<G, IMatrix, M, V>::add_parallel_dynamics(const  std::vector<V>& y, std::vector<V>& yp)
{
    double z[2]     = {-1.0,1.0};
    double Dperp[4] = {0.0, 0.0,0.0,0.0};
    //Parallel dynamics
    for(unsigned i=0; i<2; i++)
    {
        //compute em ArakawaX bracket of ds_b
        arakawa( apar[i], logn[i], arakawalogn[i]);                     // -[Apar,logN]_RZ
        arakawa( apar[i], u2[i],   arakawau2[i]);                       // -[Apar,U^2]_RZ 
        arakawa( apar[i], un[i],   arakawaun[i]);                       // -[Apar,U N]_RZ 

        dg::blas1::axpby( p.beta, arakawaun[i] ,1., yp[i]);                      // dtN += beta [Apar,U N]_RZ 
        dg::blas1::axpby( p.tau[i]/p.mu[i]*p.beta, arakawalogn[i], 1., yp[2+i]); // dtw += tau beta/mu* [Apar,logN]_RZ
        dg::blas1::axpby( 0.5*p.beta, arakawau2[i], 1., yp[2+i]);                // dtw +=  0.5 beta [Apar,U^2]_RZ 

        
        dg::blas1::axpby(1.,one,1., logn[i] ,chi); //chi = (1+lnN_e)
        dg::blas1::pointwiseDot(u[i],u[i], omega);  
        dg::blas1::axpbypgz(0.5*p.mu[i], omega, 1.0, phi[i], p.tau[i], chi); //chi = (tau (1+lnN_e) + psi + 0.5 mu U^2)
        
        //Compute perp dissipation for N
        dg::blas2::gemv( lapperp, y[i], lambda);
        dg::blas2::gemv( lapperp, lambda, omega);                      //nabla_RZ^4 N
        Dperp[i] = -z[i]* p.nu_perp*dg::blas2::dot(chi, w2d, omega);  

        //Compute perp dissipation  for w
        dg::blas2::gemv( lapperp, y[i+2], lambda);
        dg::blas2::gemv( lapperp, lambda, chi);//nabla_RZ^4 w
        Dperp[i+2] = -z[i]*p.mu[i]*p.nu_perp* dg::blas2::dot(un[i], w2d, chi);

    }
    return Dperp[0]+Dperp[1]+Dperp[2]+Dperp[3];
}

// #endif
template<class Geometry, class IMatrix, class Matrix, class container>
void Asela<Geometry, IMatrix, Matrix, container>::operator()( double time,  const std::vector<container>& y, std::vector<container>& yp)
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
        dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+1.)); //npe = N+1
        dg::blas1::transform( npe[i], logn[i], dg::LN<double>());
    }
    //compute A_parallel via induction and compute U_e and U_i from it
    if (p.beta!=0.) apar[0] = induct(y); //computes a_par and needs correct npe
    if (p.tau[1] == 0.) {
        dg::blas1::axpby(1.0,apar[0],0.,apar[1]);
    } 
    else {
        old_gammaApar.extrapolate( apar[1]);
        std::vector<unsigned> number = multigrid.direct_solve( multi_invgamma, apar[1], apar[0] , p.eps_gamma);
        old_gammaApar.update( apar[1]);
        if(  number[0] == invert_invgamma.get_max())
            throw dg::Fail( p.eps_gamma); 
    }
    
    //Compute U, U^2  and UN and energies 
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::axpby( 1., y[i+2], - p.beta/p.mu[i], apar[i], u[i]); // U = w -beta/mu A_G
        dg::blas1::pointwiseDot( u[i], u[i], u2[i]);                    // U^2
        dg::blas1::pointwiseDot( npe[i], u[i], un[i]);                  // UN
        S[i]    = z[i]*p.tau[i]*dg::blas2::dot( logn[i], w2d, npe[i]);  // S = Z tau N logN
        Tpar[i] = 0.5*z[i]*p.mu[i]*dg::blas2::dot( npe[i], w2d, u2[i]); //Tpar = 0.5 Z mu N U^2  
    }
    mass_ = dg::blas2::dot( one, w2d, y[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( npe[1], w2d, omega);   // Tperp = 0.5 mu_i N_i u_E^2
    gradient.variation( apar[0], omega); // |nabla_\perp Aparallel|^2 
    double Uapar = 0.5*p.beta*dg::blas2::dot( one, w2d, omega); // Uapar = 0.5 beta |nabla_\perp Aparallel|^2
    energy_ = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1]; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp, evec[3] = Tpar[0], evec[4] = Tpar[1]; evec[5] = Uapar;

    for( unsigned i=0; i<2; i++)
    {
        //ExB dynamics
        arakawa( y[i],   phi[i], yp[i]);                         //[N-1,phi]_RZ
        arakawa( y[i+2], phi[i], yp[i+2]);                       //[w,phi]_RZ 
    }    
    //parallel dynamics
    ediff_= add_parallel_dynamics( y, yp); 
 

    t.toc();
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif 
    std::cout << "One rhs took "<<t.diff()<<"s\n";
}


///@endcond

} //namespace asela
