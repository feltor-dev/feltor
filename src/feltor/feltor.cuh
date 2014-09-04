#pragma once

#include "dg/algorithm.h"

#include "parameters.h"
// #include "geometry_circ.h"
#include "geometry.h"
#include "init.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif //DG_BENCHMARK
/*!@file

  Contains the solvers 
  */

namespace eule
{
///@addtogroup solver
///@{
/**
 * @brief Diffusive terms for Feltor solver
 *
 * @tparam Matrix The Matrix class
 * @tparam container The Vector class 
 * @tparam Preconditioner The Preconditioner class
 */
template<class Matrix, class container, class Preconditioner>
struct Rolkar
{
    template<class Grid3d>
    Rolkar( const Grid3d& g, eule::Parameters p, solovev::GeomParameters gp):
        p(p),
        gp(gp),
        temp( dg::evaluate(dg::zero, g)), chi(temp), omega(chi),
        expy(2, temp),
        dampin_( dg::evaluate( solovev::TanhDampingIn(gp ), g)),
        dampout_( dg::evaluate( solovev::TanhDampingOut(gp ), g)),
        dampgauss_( dg::evaluate( solovev::GaussianDamping( gp), g)),
        pupil_( dg::evaluate( solovev::Pupil( gp), g)),
        lapiris_( dg::evaluate( solovev::TanhDampingInv(gp ), g)),
        LaplacianM_perp ( g, dg::normed)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        dg::blas1::axpby( 0., x, 0, y);
//         for( unsigned i=0; i<x.size(); i++)
//         {
//             dg::blas2::gemv( LaplacianM_perp, x[i], temp);
//             dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
//             dg::blas1::axpby( -p.nu_perp, y[i], 0., y[i]); //  nu_perp lapl_RZ (lapl_RZ (N,U)) 
//         }
      // Resistivity
//         dg::blas1::pointwiseDot( x[0], x[2], omega); //N_e U_e 
//         dg::blas1::pointwiseDot( x[1], x[3], chi); //N_i U_i
//         dg::blas1::axpby( -1., omega, 1., chi); //J_par = -N_e U_e + N_i U_i
//         dg::blas1::pointwiseDivide( chi, x[0], omega);//J_par/N_e
//         dg::blas1::pointwiseDivide( chi, expy[0], chi); //J_par/N_i    now //J_par/N_e  //n_e instead of n_i
//         dg::blas1::axpby( -p.c/p.mu[0]/p.eps_hat, omega, 1., y[2]);  // dtU_e =- C/hat(mu)_e J_par/N_e
//         dg::blas1::axpby( -p.c/p.mu[1]/p.eps_hat,omega, 1., y[3]); 
        // dtU_e =- C/hat(mu)_i J_par/N_i   //n_e instead of n_i 
//         cut contributions to boundary now with damping on all 4 quantities
//         for( unsigned i=0; i<y.size(); i++){
//             dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
//         }
    }
    dg::Elliptic<Matrix, container, Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const Preconditioner& weights(){return LaplacianM_perp.weights();}
    const Preconditioner& precond(){return LaplacianM_perp.precond();}
    const container& pupil(){return pupil_;}
    const container& dampin(){return dampin_;}
  private:
    const eule::Parameters p;
    const solovev::GeomParameters gp;
    container temp, chi, omega;
    std::vector<container> expy;
    const container dampin_;
    const container dampout_;
    const container dampgauss_;
    const container pupil_;
    const container lapiris_;
    
    dg::Elliptic<Matrix, container, Preconditioner> LaplacianM_perp;

};

template< class Matrix, class container=thrust::device_vector<double>, class Preconditioner = thrust::device_vector<double> >
struct Feltor
{
    //typedef std::vector<container> Vector;
    typedef typename dg::VectorTraits<container>::value_type value_type;
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    //typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    template<class Grid3d>
    Feltor( const Grid3d& g, eule::Parameters p,solovev::GeomParameters gp);

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

    container chi, omega, gammani;

    const container binv, curvR, curvZ, gradlnB;
    const container pupil, source, damping, one;
    const Preconditioner w3d, v3d;

    std::vector<container> phi, curvphi,curvlogn, dzphi,dzun,dzlogn,dzu2, expy,logy;
    std::vector<container> dzy, curvy; 

    //matrices and solvers
    dg::DZ<Matrix, container> dz;
    dg::ArakawaX< Matrix, container>    arakawa; 
    //dg::Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector

    dg::Elliptic< Matrix, container, Preconditioner > pol; //note the host vector
    dg::Helmholtz< Matrix, container, Preconditioner > invgamma;
    dg::Invert<container> invert_pol,invert_invgamma;

    const eule::Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;

};

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, eule::Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi), gammani(chi),
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    curvR( dg::evaluate( solovev::CurvatureR(gp), g)),
    curvZ( dg::evaluate(solovev::CurvatureZ(gp), g)),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    pupil( dg::evaluate( solovev::Pupil( gp), g)),
    source( dg::evaluate(solovev::TanhSource(gp, p.amp_source), g)),
    damping( dg::evaluate( solovev::GaussianDamping(gp ), g)), 
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)), 
    phi( 2, chi), curvphi( phi),curvlogn(phi), dzphi(phi), dzun(phi),dzlogn(phi),dzu2(phi),expy(phi),  logy(phi),
    dzy( 4, chi),curvy(dzy), 
    dz(solovev::Field(gp), g, gp.rk4eps, dg::DefaultLimiter()),
// dz(solovev::Field(gp), g, gp.rk4eps, dg::NoLimiter()),
//     dz(solovev::Field(gp), g, gp.rk4eps,solovev::PsiLimiter(gp)),
    arakawa( g), 
    pol(     g), 
    invgamma(g,-0.5*p.tau[1]*p.mu[1]),
    invert_pol( omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    p(p),
    gp(gp)
{ }

template<class Matrix, class container, class P>
container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( -p.mu[0], y[0], p.mu[1], y[1], chi);    //chi =  \mu_i n_i - \mu_e n_e
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);                 //(\mu_i n_i - \mu_e n_e) /B^2
    pol.set_chi( chi);
    dg::blas1::transform( y[1], omega,   dg::PLUS<double>(-1));//omega= Ni-1 
    unsigned numberg    =  invert_invgamma(invgamma,chi,omega); //omega= Gamma (Ni-1)
    dg::blas1::transform(  chi, gammani, dg::PLUS<double>(+1));
    dg::blas1::axpby( -1., y[0], 1.,gammani,chi);                   //chi=  Gamma (n_i-1) +1  - (n_e) = Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);        //Gamma n_i -ne = -nabla chi nabla phi
    if( number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_vesqr( container& potential)
{
    arakawa.bracketS( potential, potential, chi);                           //dR phi dR phi + Dz phi Dz phi
    dg::blas1::pointwiseDot( binv, binv, omega);
    dg::blas1::pointwiseDot( chi, omega, omega);
    return omega;                                                           //u_E = (dR phi dR phi + Dz phi Dz phi)/B^2
}
template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_psi( container& potential)
{
    invert_invgamma(invgamma,chi,potential);                               //chi = Gamma phi
    dg::blas1::axpby( 1., chi, -0.5, compute_vesqr( potential),phi[1]);    //psi = Gamma phi - 0.5 u_E^2
    return phi[1];    
}
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::initializene( const container& src, container& target)
{ 
    dg::blas1::transform( src,omega, dg::PLUS<double>(-1)); //n_i -1
    invert_invgamma(invgamma,target,omega); //=ne-1 = Gamma (ni-1)    
    dg::blas1::transform( target,target, dg::PLUS<double>(+1)); //n_i
}


template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    //compute phi via polarisation
    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    //update energetics, 2% of total time
    log( y, logy, 2);
    mass_ = dg::blas2::dot( one, w3d, y[0] ); //take real ion density which is electron density!!
    double Ue = p.tau[0]*dg::blas2::dot( logy[0], w3d, y[0]); // tau_e n_e ln(n_e)
    double Ui = p.tau[1]*dg::blas2::dot( logy[1], w3d, y[1]);// tau_i n_i ln(n_i)
    double Uphi = 0.5*p.mu[1]*dg::blas2::dot( y[1], w3d, omega); 
        dg::blas1::pointwiseDot( y[2], y[2], omega); //U_e^2
    double Upare = -0.5*p.mu[0]*dg::blas2::dot( y[0], w3d, omega); //N_e U_e^2
        dg::blas1::pointwiseDot(y[3], y[3], omega); //U_i^2
    double Upari =  0.5*p.mu[1]*dg::blas2::dot( y[1], w3d, omega); //N_i U_i^2
    energy_ = Ue + Ui  + Uphi + Upare + Upari;

    // the resistive dissipation without FLR      
        dg::blas1::pointwiseDot(y[0], y[2], omega); //N_e U_e 
        dg::blas1::pointwiseDot( y[1], y[3], chi); //N_i U_i
        dg::blas1::axpby( -1., omega, 1., chi); //-N_e U_e + N_i U_i                  //dt(lnN,U) = dt(lnN,U) + dz (dz (lnN,U))
    //double Dres = -p.c*dg::blas2::dot(chi, w3d, chi); //- C*J_parallel^2

    //Dissipative terms without FLR
//         dg::blas1::axpby(1.,dg::evaluate( dg::one, g),1., y[0] ,chi); //(1+lnN_e)
//         dg::blas1::axpby(1.,phi[0],p.tau[0], chi); //(1+lnN_e-phi)
//         dg::blas1::pointwiseDot( expy[0], chi, omega); //N_e phi_e     
//         dg::blas2::gemv( lapperp, y[0],chi);
//         dg::blas2::gemv( lapperp, chi,chi);//nabla_RZ^4 lnN_e
//     double Dperpne =  p.nu_perp*dg::blas2::dot(omega, w3d, chi);
//         dg::blas1::axpby(1.,dg::evaluate( dg::one, g),1., y[1] ,chi); //(1+lnN_i)
//         dg::blas1::axpby(1.,phi[1],p.tau[1], chi); //(1+lnN_i-phi)
//         dg::blas1::pointwiseDot( expy[1], chi, omega); //N_i phi_i     
//         dg::blas2::gemv( lapperp, y[1], chi);
//         dg::blas2::gemv( lapperp, chi,chi);//nabla_RZ^4 lnN_i
//     double Dperpni = - p.nu_perp*dg::blas2::dot(omega, w3d, chi);
//         dg::blas1::pointwiseDot( expy[0], y[2], omega); //N_e U_e     
//         dg::blas2::gemv( lapperp, y[2], chi);
//         dg::blas2::gemv( lapperp, chi,chi);//nabla_RZ^4 U_e
//     double Dperpue = p.nu_perp*p.mu[0]* dg::blas2::dot(omega, w3d, chi);
//         dg::blas1::pointwiseDot( expy[1], y[3], omega); //N_e U_e     
//         dg::blas2::gemv( lapperp, y[3], chi);
//         dg::blas2::gemv( lapperp, chi,chi);//nabla_RZ^4 U_i
//     double Dperpui = - p.nu_perp*p.mu[1]* dg::blas2::dot(omega, w3d, chi);
//     ediff_ = Dres + Dperpne + Dperpni + Dperpue + Dperpui;
    //the parallel part is done elsewhere
    
    for( unsigned i=0; i<2; i++)
    {
        //ExB dynamics
        arakawa( y[i], phi[i], yp[i]);  //[N,phi]_RZ
        arakawa( y[i+2], phi[i], yp[i+2]);//[U,phi]_RZ  
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);                        // dtN =1/B [N,phi]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);                    // dtU =1/B [U,phi]_RZ  
        
        //Parallel dynamics
//         dz.set_boundaries( dg::NEU, 0, 0);
        dz(y[i], dzy[i]);                                                       //dz N
//         dz.set_boundaries( dg::DIR, -1., 1.);
        dz(y[i+2], dzy[2+i]);                                                   //dz U
        dg::blas1::pointwiseDot(y[i],y[i+2], omega);                            //U N
        dz.set_boundaries( dg::NEU, 0, 0);
        dz(omega, dzun[i]);                                                     //dz UN
        dg::blas1::axpby( -1., dzun[i], 1., yp[i]);                             //dtN = dtN - dz U N
        dg::blas1::pointwiseDot(omega, gradlnB, omega);                         //U N dz ln B
        dg::blas1::axpby( 1., omega, 1., yp[i]);                                //dtN = dtN + U N dz ln B
        //parallel force terms
//         dz.set_boundaries( dg::NEU, 0, 0);
        dz(phi[i], dzphi[i]);                                                   //dz psi
//         dz.set_boundaries( dg::NEU, 0, 0);
        dz(logy[i], dzlogn[i]);                                                 //dz lnN
        dg::blas1::axpby( -p.tau[i]/p.mu[i]/p.eps_hat, dzlogn[i], 1., yp[2+i]); //dtU = dtU - tau/(hat(mu))*dz lnN
        dg::blas1::axpby( -1./p.mu[i]/p.eps_hat, dzphi[i], 1., yp[2+i]);        //dtU = dtU - 1/(hat(mu))  *dz phi  
         
        dg::blas1::pointwiseDot(y[i+2],y[i+2], omega);                          //U^2
//         dz.set_boundaries( dg::DIR, -1., 1.);
        dz(omega, dzu2[i]);                                                     //dz u^2
        dg::blas1::axpby( -0.5, dzu2[i], 1., yp[2+i]);                          //dtU = dtU - 0.5 dz U^2
        
        //Curvature dynamics
        curve( y[i], curvy[i]);                                                 //K(N) 
        curve( y[i+2], curvy[2+i]);                                             //K(U) 
        curve( phi[i], curvphi[i]);                                             //K(phi) 
        curve( logy[i], curvlogn[i]);                                           //K(logn) 
        
        dg::blas1::pointwiseDot(y[i+2], curvy[2+i], omega);                     //U K(U) 
        dg::blas1::pointwiseDot( y[i+2], omega, chi);                           //U^2 K(U)
        dg::blas1::pointwiseDot( y[i], omega, omega);                           //N U K(U)
        
        dg::blas1::axpby( -p.mu[i]*p.eps_hat, omega, 1., yp[i]);                //dtN = dtN - (hat(mu)) N U K(U)
        dg::blas1::axpby( -0.5*p.mu[i]*p.eps_hat, chi, 1., yp[2+i]);            //dtU = dtU - 0.5 (hat(mu)) U^2 K(U)

        dg::blas1::pointwiseDot(y[i+2], curvlogn[i], omega);                    //U K(ln N)
        dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);                       //dtU = dtU - tau U K(lnN)
        
        dg::blas1::pointwiseDot(y[i+2], curvy[i], omega);                       //U K( N)
        dg::blas1::pointwiseDot( y[i+2], omega, chi);                           //U^2K( N)
        dg::blas1::axpby( -0.5*p.mu[i]*p.eps_hat, chi, 1., yp[i]);              //dtN = dtN - 0.5 mu U^2 K(N)

        dg::blas1::axpby( -p.tau[i], curvy[i], 1., yp[i]);                      //dtN = dtN - tau K(N)
        dg::blas1::axpby( -2.*p.tau[i], curvy[2+i], 1., yp[2+i]);               //dtU = dtU - 2 tau K(U)
        dg::blas1::pointwiseDot(y[i],curvphi[i], omega);                        //N K(psi)
        dg::blas1::axpby( -1., omega, 1., yp[i]);                               //dtN= dtN - N K(psi)

        dg::blas1::pointwiseDot( y[i+2], curvphi[i], omega);                    //U K(phi)
        dg::blas1::axpby( -0.5, omega, 1., yp[2+i]);                            //dtU = dtU -0.5 U K(psi)

        //Parallel dissipation
//        dz.set_boundaries( dg::NEU, 0, 0);
        dz.dzz(y[i],omega);                                                     //dz^2 N 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i]);               
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB,dzy[i], omega);                         // dz lnB dz N    
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i]);    
        //         dz.set_boundaries( dg::DIR, -1., 1.);       
        dz.dzz(y[i+2],omega);                                                   //dz^2 U 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i+2]);               
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB,dzy[i+2], omega);                       // dz lnB dz U
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i+2]);    
    }

    //add particle source to dtN
//     for( unsigned i=0; i<2; i++)
//     {
//         dg::blas1::pointwiseDivide( source, expy[i], omega); //source/N
//         dg::blas1::axpby( 1., omega, 1, yp[i]  );       //dtlnN = dtlnN + source/N
//     }

    for( unsigned i=0; i<4; i++) //damping and pupil on N and w
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

///@}

} //namespace eule
