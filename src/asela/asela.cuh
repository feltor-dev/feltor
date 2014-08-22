#pragma once

#include "dg/algorithm.h"

#include "parameters.h"
// #include "geometry_circ.h"
#include "geometry.h"
#include "init.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif //DG_BENCHMARK

// #define APAR
namespace eule
{
//diffusive terms (add mu_hat?)
template<class Matrix, class container, class Preconditioner>
struct Rolkar
{
    template<class Grid3d>
    Rolkar( const Grid3d& g, Parameters p, solovev::GeomParameters gp):
        p(p),
        gp(gp),
        w3d_( dg::create::weights(g)), v3d_(dg::create::inv_weights(g)),
        temp( dg::evaluate(dg::zero, g)),
        dampin_( dg::evaluate( solovev::TanhDampingIn(gp ), g)),
        dampout_( dg::evaluate( solovev::TanhDampingOut(gp ), g)),
        dampgauss_( dg::evaluate( solovev::GaussianDamping( gp), g)),
        pupil_( dg::evaluate( solovev::Pupil( gp), g)),
        lapiris_( dg::evaluate( solovev::TanhDampingInv(gp ), g)),
        LaplacianM_perp(  g, dg::normed)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        //we may change x
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::axpby( -p.nu_perp, y[i], 0., y[i]); // - nu_perp lapl_RZ (lapl_RZ (lnN,U)) //factor MISSING!?!

        }
       
        //cut contributions to boundary now with damping on all 4 quantities
        for( unsigned i=0; i<y.size(); i++)
        {
            dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
        }
    }
    dg::Elliptic<Matrix,container,Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const container& weights(){return w3d_;}
    const container& precond(){return v3d_;}
    const container& pupil(){return pupil_;}
    const container& dampin(){return dampin_;}

  private:
    const Parameters p;
    const solovev::GeomParameters gp;
    const Preconditioner w3d_, v3d_;
    container temp;
    const container dampin_;
    const container dampout_;
    const container dampgauss_;
    const container pupil_;
    const container lapiris_;
    dg::Elliptic<  Matrix, container, Preconditioner  > LaplacianM_perp;
};

template< class Matrix, class container=thrust::device_vector<double>, class Preconditioner = thrust::device_vector<double> >
struct Feltor
{
    typedef typename dg::VectorTraits<container>::value_type value_type;

    template<class Grid3d>

    Feltor( const Grid3d& g, Parameters p,solovev::GeomParameters gp);

    void exp( const std::vector<container>& src, std::vector<container>& dst, unsigned);

    void log( const std::vector<container>& src, std::vector<container>& dst, unsigned);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
    void initializene( const container& y, container& target);
    const container& aparallel( ) const { return apar;}
    const std::vector<container>& uparallel( ) const { return u;}

    /**
     * @brief Return the Gamma operator used by this object
     *
     * @return Gamma operator
     */

    void operator()( std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    double energy_diffusion( ){ return ediff_;}

  private:
    void curve( container& y, container& target);
    //use chi and omega as helpers to compute square velocity in omega
    container& compute_vesqr( container& potential);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation
    container& induct(const std::vector<container>& y);//solves induction equation

    container chi, omega;
    container apar,rho,gammani;

    const container binv, curvR, curvZ, gradlnB;
    const container pupil, source, damping;
    const container w3d, v3d, one;
    container curvapar;
    std::vector<container> phi, curvphi, dzphi, expy;
    std::vector<container> dzy, curvy; 
    std::vector<container> arakAN,arakAU,arakAphi,u;

    //matrices and solvers
    //Matrix A; 
    dg::DZ< Matrix, container> dz;
    dg::ArakawaX< Matrix, container>    arakawa; 
    //dg::Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector
    dg::Elliptic<  Matrix, container, Preconditioner  > pol; //note the host vector

    dg::Helmholtz< Matrix, container, Preconditioner > maxwell, invgamma;
    dg::Invert<container> invert_maxwell, invert_pol, invert_invgamma;

    const Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;

};

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),
    rho( chi), apar(chi), curvapar(chi),gammani(chi),
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    curvR( dg::evaluate( solovev::CurvatureR(gp), g)),
    curvZ( dg::evaluate(solovev::CurvatureZ(gp), g)),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    pupil( dg::evaluate( solovev::Pupil( gp), g)),
    source( dg::evaluate(solovev::TanhSource(gp, p.amp_source), g)),
    damping( dg::evaluate( solovev::GaussianDamping(gp ), g)), 
    phi( 2, chi), curvphi( phi), dzphi(phi), expy(phi),  
    arakAN( phi), arakAU( phi), arakAphi(phi),u(phi),   
    dzy( 4, chi), curvy(dzy),
    //A (dg::create::laplacianM_perp( g, dg::not_normed, dg::symmetric)),
    dz(solovev::Field(gp), g, gp.rk4eps),
    arakawa( g), 
    pol(     g), 
    invgamma(g,-0.5*p.tau[1]*p.mu[1]),
    invert_pol( omega, omega.size(), p.eps_pol), 
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)), one(dg::evaluate( dg::one, g)),
    maxwell(g, 1.), //sign is already correct!
    invert_maxwell(rho, rho.size(), p.eps_maxwell),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    p(p),
    gp(gp)
{ }
template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_vesqr( container& potential)
{
    arakawa.bracketS( potential, potential, chi);
    dg::blas1::pointwiseDot( binv, binv, omega);
    dg::blas1::pointwiseDot( chi, omega, omega);
    return omega;
}
template<class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_psi( container& potential)
{
    //without FLR
//     dg::blas1::axpby( 1., potential, -0.5, compute_vesqr( potential), phi[1]);
    //with FLR
    invert_invgamma(invgamma,chi,potential);
    dg::blas1::axpby( 1., chi, -0.5, compute_vesqr( potential),phi[1]);    
    return phi[1];
}

template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::initializene( const container& src, container& target)
{ 

    dg::blas1::transform( src,omega, dg::PLUS<double>(-1)); //n_i -1
    invert_invgamma(invgamma,target,omega); //=ne-1 = Gamma (ni-1)    
    dg::blas1::transform( target,target, dg::PLUS<double>(+1)); //n_i

}

//computes and modifies expy!!
template<class Matrix, class container, class P>
container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
// #ifdef DG_BENCHMARK
//     dg::Timer t; 
//     t.tic();
// #endif
    //compute chi and polarisation
    exp( y, expy, 2);
    dg::blas1::axpby( 1., expy[1], 0., chi); //\chi = a_i \mu_i n_i
    //correction
    dg::blas1::axpby( -p.mu[0], expy[0], 1., chi); //\chi = a_i \mu_i n_i -a_e \mu_e n_e
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi); //chi/= B^2

    //A = pol.create( chi);
    pol.set_chi( chi);
//     dg::blas1::transform( expy[0], expy[0], dg::PLUS<double>(-1)); //n_e -1
    dg::blas1::transform( expy[1], omega,   dg::PLUS<double>(-1)); //n_i -1
    //with FLR
    unsigned numberg =  invert_invgamma(invgamma,chi,omega);    //chi= Gamma (Omega) = Gamma (ni-1)
    dg::blas1::transform(  chi, gammani, dg::PLUS<double>(1)); // Gamma N_i = Gamma (Ni-1)+1
/*    if( numberg == invert_invgamma.get_max())
        throw dg::Fail( p.eps_gamma);*/  
// #ifdef DG_BENCHMARK
//     t.toc();
//     std::cout<< "Polarisation assembly took "<<t.diff()<<"s\n";
// #endif 
    dg::blas1::axpby( -1., expy[0], 1., gammani,chi); //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_1 - n_e
    unsigned number = invert_pol( pol, phi[0], chi); //Gamma n_i -ne = -nabla chi nabla phi

    if( number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

// #if def APAR
template<class Matrix, class container, class P>
container& Feltor< Matrix, container, P>::induct(const std::vector<container>& y)
{
    exp( y, expy, 2);
    dg::blas1::axpby( p.beta/p.mu[0], expy[0], 0., chi); //chi = beta/mu_e N_e
//     dg::blas1::axpby(- p.beta/p.mu[1], expy[1], 1., chi); //chi =beta/mu_e N_e-beta/mu_i N_i
    dg::blas1::axpby(- p.beta/p.mu[1], gammani, 1., chi); //chi =beta/mu_e N_e-beta/mu_i Gamma N_i
    maxwell.set_chi(chi);
    dg::blas1::pointwiseDot( expy[0], y[2], rho);                 //rho = n_e w_e
    dg::blas1::pointwiseDot( expy[1], y[3], omega);               //omega = n_i w_i
    dg::blas1::axpby( -1.,omega , 1., rho);  //rho = -n_i w_i + n_e w_e
    //maxwell = (lap_per - beta*(N_i/hatmu_i - n_e/hatmu_e)) A_parallel 
    //rho=n_e w_e -N_i w_i
    unsigned number = invert_maxwell( maxwell, apar, rho); //omega equals a_parallel
    if( number == invert_maxwell.get_max())
        throw dg::Fail( p.eps_maxwell);
    return apar;
}

// #endif
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    //compute phi via polarisation
    phi[0] = polarisation( y);//transformed exp()
    phi[1] = compute_psi( phi[0]);
    //compute A_parallel via induction and compute U_e and U_i from it
    apar = induct(y);

    //calculate U from Apar and w
    dg::blas1::axpby( 1., y[2], - p.beta/p.mu[0], apar, u[0]); // U_e = w_e -beta/mu_e A_parallel
    dg::blas1::axpby( 1., y[3], - p.beta/p.mu[1], apar, u[1]); // U_i = w_i -beta/mu_i A_parallel

    //update energetics, 2% of total time
    exp( y, expy, 2);
    mass_ = dg::blas2::dot( one, w3d, expy[0] ); //take real ion density which is electron density!!
    double Ue = p.tau[0]*dg::blas2::dot( y[0], w3d, expy[0]); // tau_e n_e ln(n_e)
    double Ui = p.tau[1]*dg::blas2::dot( y[1], w3d, expy[1]);// tau_i n_i ln(n_i)
    double Uphi = 0.5*p.mu[1]*dg::blas2::dot( expy[1], w3d, omega); 
    dg::blas1::pointwiseDot( u[0], u[0], omega); //U_e^2
    double Upare = -0.5*p.mu[0]*dg::blas2::dot( expy[0], w3d, omega); 
    dg::blas1::pointwiseDot(u[1], u[1], omega); //U_i^2
    double Upari =  0.5*p.mu[1]*dg::blas2::dot( expy[1], w3d, omega); 
    arakawa.variation(apar,omega); // (dx A_parallel)^2 + (dy A_parallel)^2
    double Uapar = p.beta*sqrt(dg::blas2::dot( omega, w3d, omega));
    energy_ = Ue + Ui + Uphi + Upare + Upari + Uapar;
    
    
    curve( apar, curvapar); //K(A_parallel)
    dg::blas1::axpby(  1.,  gradlnB,p.beta,  curvapar);  // gradlnB + beta K(A_parallel) factor 0.5 or not?
    for( unsigned i=0; i<2; i++)
    {
        //Compute RZ poisson  brackets
        arakawa( y[i], phi[i], yp[i]);  //[lnN,phi]_RZ
        arakawa( u[i], phi[i], yp[i+2]);//[U,phi]_RZ  
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);                        // dtlnN =1/B [phi,lnN]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);                    // dtU =1/B [phi,U]_RZ
        // compute A_parallel terms of the parallel derivatuve
        arakawa( apar, y[i], arakAN[i]); // [A_parallel,lnN]_RZ
        arakawa( apar, u[i], arakAU[i]); // [A_parallel,U]_RZ
        arakawa( apar, phi[i], arakAphi[i]); // [A_parallel,phi]_RZ
        dg::blas1::pointwiseDot(arakAN[i],binv,arakAN[i]);   //1/B [A_parallel,lnN]_RZ
        dg::blas1::pointwiseDot(arakAU[i],binv,arakAU[i]);   //1/B [A_parallel,U]_RZ
        dg::blas1::pointwiseDot(arakAphi[i],binv,arakAphi[i]); //1/B [A_parallel,phi]_RZ

        //compute parallel derivatives
        dz(y[i], dzy[i]);     //dz(lnN)
        dz(phi[i], dzphi[i]); //dz(phi)
        dz(u[i], dzy[2+i]);   //dz(U)
        //add A_parallel terms to the parallel derivatives
        dg::blas1::axpby(-0.5*p.beta,arakAN[i]   ,1.,dzy[i]);  //= dz lnN -beta/B [A_parallel,lnN ] 
        dg::blas1::axpby(-0.5*p.beta,arakAphi[i] ,1.,dzphi[i]);//= dz phi -beta/B [A_parallel,phi ] 
        dg::blas1::axpby(-0.5*p.beta,arakAU[i]   ,1.,dzy[2+i]);//= dz U-beta/B [A_parallel,U ] 


        //parallel advection terms
        dg::blas1::pointwiseDot(u[i], dzy[i], omega); //Udz lnN 
        dg::blas1::axpby( -1., omega, 1., yp[i]);                            //dtlnN = dtlnN - Udz lnN
        dg::blas1::axpby( -1., dzy[2+i], 1., yp[i]);                         //dtlnN = dtlnN - dz U
        dg::blas1::pointwiseDot(u[i], curvapar, omega);                     
        dg::blas1::axpby( 1., omega, 1., yp[i]);                            //dtlnN = dtlnN + U dz ln B
        dg::blas1::pointwiseDot(u[i], dzy[2+i], omega);                    
        dg::blas1::axpby( -1., omega, 1., yp[2+i]);                         //dtw = dtw - U dz U

        //parallel force terms
        dg::blas1::axpby( -p.tau[i]/p.mu[i]/p.eps_hat, dzy[i], 1., yp[2+i]); //dtw = dtw - tau/(hat(mu))*dz lnN
        dg::blas1::axpby( -1./p.mu[i]/p.eps_hat, dzphi[i], 1., yp[2+i]);     //dtw = dtw - 1/(hat(mu))*dz phi

        //curvature terms
        curve( y[i], curvy[i]);     //K(N)
        curve( u[i], curvy[2+i]);  //K(U)
        curve( phi[i], curvphi[i]);//K(phi)

        dg::blas1::pointwiseDot( u[i], curvy[2+i], omega); //U K(U)
        dg::blas1::pointwiseDot( u[i], omega, chi); //U^2 K(U)
        dg::blas1::axpby( -p.mu[i]*p.eps_hat, omega, 1., yp[i]);             //dtlnN = dtlnN - (hat(mu)) U K(U)
        dg::blas1::axpby( -0.5*p.mu[i]*p.eps_hat, chi, 1., yp[2+i]);         //dtw = dtw - 0.5 (hat(mu)) U^2 K(U)

        dg::blas1::pointwiseDot(u[i], curvy[i], omega); //U K(ln N)
        dg::blas1::pointwiseDot( u[i], omega, chi); //U^2K(ln N)
        dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);                    //dtw = dtw - tau U K(lnN)
        dg::blas1::axpby( -0.5*p.mu[i]*p.eps_hat, chi, 1., yp[i]);           //dtlnN = dtlnN -0.5mu U^2K(lnN)

        dg::blas1::axpby( -p.tau[i], curvy[i], 1., yp[i]);                   //dtlnN = dtlnN - tau K(lnN)
        dg::blas1::axpby( -2.*p.tau[i], curvy[2+i], 1., yp[2+i]);            //dtw = dtw - 2 tau K(U)
        dg::blas1::axpby( -1., curvphi[i], 1., yp[i]);                       //dtlnN= dtlnN - K(psi)

        dg::blas1::pointwiseDot( u[i], curvphi[i], omega);  //U K(phi)
        dg::blas1::axpby( -0.5, omega, 1., yp[2+i]);                         //dtw = dtw -0.5 U K(psi)


    }
    //add parallel resistivity now explicit
    dg::blas1::pointwiseDot( expy[0], u[0], omega); //N_e U_e 
    dg::blas1::pointwiseDot( expy[1], u[1], chi); //N_i U_i
    dg::blas1::axpby( -1., omega, 1., chi); //J_par = -N_e U_e + N_i U_i
    dg::blas1::pointwiseDivide( chi, expy[0], omega);//J_par/N_e
//         //for 1/N_i
// //         dg::blas1::pointwiseDivide( chi, expy[1], chi); //J_par/N_i    now //J_par/N_e  //n_e instead of n_i
    dg::blas1::axpby( -p.c/p.mu[0]/p.eps_hat, omega, 1., yp[2]);  // dtU_e =- C/hat(mu)_e J_par/N_e
    dg::blas1::axpby( -p.c/p.mu[1]/p.eps_hat, omega, 1., yp[3]);  // dtU_e =- C/hat(mu)_e J_par/N_e
        
    //add parallel diffusion with naive implementation
    for( unsigned i=0; i<4; i++)
    {
        dz(dzy[i], omega); //dz (dz (N,U))
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i]);                     //dt(lnN,U) = dt(lnN,U) + dz (dz (lnN,U))
    }
    //add particle source to dtN
//     for( unsigned i=0; i<2; i++)
//     {
//         dg::blas1::pointwiseDivide( source, expy[i], omega); //source/N
//         dg::blas1::axpby( 1., omega, 1, yp[i]  );       //dtlnN = dtlnN + source/N
//     }
    for( unsigned i=0; i<2; i++) //pupil on U for nicer plot <- does not contribute to dynamics
    {
        dg::blas1::pointwiseDot( damping, u[i], u[i]); 
    }
    for( unsigned i=0; i<4; i++) //damping  on N and w
    {
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]); 
    }
}

//Computes curvature operator
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::curve( container& src, container& target)
{
    dg::blas2::gemv( arakawa.dx(), src, target); //d_R src
    dg::blas2::gemv( arakawa.dy(), src, omega);  //d_Z src
    dg::blas1::pointwiseDot( curvR, target, target); // C^R d_R src
    dg::blas1::pointwiseDot( curvZ, omega, omega);   // C^Z d_Z src
    dg::blas1::axpby( 1., omega, 1., target ); // (C^R d_R + C^Z d_Z) src
}
//Exp
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::exp( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        dg::blas1::transform( y[i], target[i], dg::EXP<value_type>());
}
//Log
template< class M, class container, class P>
void Feltor<M, container, P>::log( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        dg::blas1::transform( y[i], target[i], dg::LN<value_type>());
}


} //namespace eule
