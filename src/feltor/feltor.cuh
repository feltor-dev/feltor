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
        dampgauss_( dg::evaluate( solovev::GaussianDamping( gp), g)),
        lapiris_( dg::evaluate( solovev::TanhDampingInv(gp ), g)),
        LaplacianM_perp ( g, dg::normed, dg::symmetric)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
           x[1] := N_i - 1
           x[2] := U_e
           x[3] := U_i
        */
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<4; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 

        }
        //Resistivity
        //dg::blas1::pointwiseDot( x[0], x[2], omega); //N_e U_e 
        //dg::blas1::pointwiseDot( x[1], x[3], chi); //N_i U_i
        //dg::blas1::axpby( -1., omega, 1., chi); //J_par = -N_e U_e + N_i U_i
        //dg::blas1::pointwiseDivide( chi, x[0], omega);//J_par/N_e
        //dg::blas1::axpby( -p.c/p.mu[0], omega, 1., y[2]);   // dt U_e =- C/hat(mu)_e J_par/N_e
        //dg::blas1::axpby( -p.c/p.mu[1], omega, 1., y[3]);   // dt U_i =- C/hat(mu)_i J_par/N_i   //n_e instead of n_i now
        //or U_i - U_e
        dg::blas1::axpby( 1., x[3], -1, x[2], omega);
        dg::blas1::axpby( -p.c/p.mu[0], omega, 1., y[2]);   
        dg::blas1::axpby( -p.c/p.mu[1], omega, 1., y[3]);   
        //damping
        for( unsigned i=0; i<y.size(); i++){
           dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
        }
    }
    dg::Elliptic<Matrix, container, Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const Preconditioner& weights(){return LaplacianM_perp.weights();}
    const Preconditioner& precond(){return LaplacianM_perp.precond();}
    const container& damping(){return dampin_;}
  private:
    const eule::Parameters p;
    const solovev::GeomParameters gp;
    container temp, chi, omega;
    std::vector<container> expy;
    const container dampin_;
    const container dampgauss_;
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

    dg::DZ<Matrix, container> dz(){return dz_;}

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
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation

    container chi, omega; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!

    const container binv, curvR, curvZ, gradlnB;
    const container source, damping, one;
    const Preconditioner w3d, v3d;

    std::vector<container> phi, curvphi;
    std::vector<container> expy, npe, logn, ush; 
    std::vector<container> dzy, curvy; 

    //matrices and solvers
    dg::DZ<Matrix, container> dz_;
    dg::ArakawaX< Matrix, container>    arakawa; 
    //dg::Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector

    dg::Elliptic< Matrix, container, Preconditioner > pol; 
    dg::Helmholtz< Matrix, container, Preconditioner > invgamma;
    dg::Invert<container> invert_pol,invert_invgamma;

    const eule::Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;

};

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, eule::Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi), 
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    curvR( dg::evaluate( solovev::CurvatureR(gp), g)),
    curvZ( dg::evaluate(solovev::CurvatureZ(gp), g)),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    source( dg::evaluate(solovev::TanhSource(gp, p.amp_source), g)),
    damping( dg::evaluate( solovev::GaussianDamping(gp ), g)), 
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)), 
    phi( 2, chi), curvphi( phi), expy(phi), npe(phi), logn(phi),ush(phi),
    dzy( 4, chi),curvy(dzy), 
//  dz(solovev::Field(gp), g, gp.rk4eps, dg::DefaultLimiter()),
//  dz(solovev::Field(gp), g, gp.rk4eps, dg::NoLimiter()),
    dz_(solovev::Field(gp), g, gp.rk4eps,solovev::PsiLimiter(gp)),
    arakawa( g), 
    pol(     g, dg::not_normed, dg::symmetric), 
    invgamma(g,-0.5*p.tau[1]*p.mu[1]),
    invert_pol( omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    p(p),
    gp(gp)
{ }

template<class Matrix, class container, class P>
container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( -p.mu[0], y[0], p.mu[1], y[1], chi);      //chi =  \mu_i (n_i-1) - \mu_e (n_e-1)
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]-p.mu[0]));
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);                   //(\mu_i n_i - \mu_e n_e) /B^2
    pol.set_chi( chi);

    unsigned numberg    =  invert_invgamma(invgamma,chi,y[1]); //omega= Gamma (Ni-1)
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi

    if( number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_psi( container& potential)
{
    invert_invgamma(invgamma,chi,potential);                    //chi  Gamma phi
    arakawa.bracketS( potential, potential, omega);             //dR phi dR phi + Dz phi Dz phi
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}

template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::initializene( const container& src, container& target)
{ 
    invert_invgamma(invgamma,target,src); //=ne-1 = Gamma (ni-1)    
}

template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    /* y[0] := N_e - 1
       y[1] := N_i - 1
       y[2] := U_e
       y[3] := U_i
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    //compute phi via polarisation
    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    //update energetics, 2% of total time
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+1)); //npe = N+1
        dg::blas1::transform( npe[i], logn[i], dg::LN<value_type>());
    }
    mass_ = dg::blas2::dot( one, w3d, y[0] ); //take real ion density which is electron density!!
    double Ue = p.tau[0]*dg::blas2::dot( logn[0], w3d, y[0]); // tau_e n_e ln(n_e)
    double Ui = p.tau[1]*dg::blas2::dot( logn[1], w3d, y[1]);// tau_i n_i ln(n_i)
    double Uphi = 0.5*p.mu[1]*dg::blas2::dot( y[1], w3d, omega); 
        dg::blas1::pointwiseDot( y[2], y[2], omega); //U_e^2
    double Upare = -0.5*p.mu[0]*dg::blas2::dot( y[0], w3d, omega); //N_e U_e^2
        dg::blas1::pointwiseDot(y[3], y[3], omega); //U_i^2
    double Upari =  0.5*p.mu[1]*dg::blas2::dot( y[1], w3d, omega); //N_i U_i^2
    energy_ = Ue + Ui  + Uphi + Upare + Upari;

    //// the resistive dissipation without FLR      
    //    dg::blas1::pointwiseDot(y[0], y[2], omega); //N_e U_e 
    //    dg::blas1::pointwiseDot( y[1], y[3], chi); //N_i U_i
    //    dg::blas1::axpby( -1., omega, 1., chi); //-N_e U_e + N_i U_i                  //dt(lnN,U) = dt(lnN,U) + dz (dz (lnN,U))
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
    //set U_sheath
    dg::blas1::axpby( -1.0, phi[0], 0., ush[0]);                       //U_sh_e = - phi
    dg::blas1::transform(  ush[0],ush[0], dg::EXP<value_type>());      //U_sh_e = EXP(-phi)                  
    dg::blas1::scal(ush[0], 1.0/sqrt(-2.*M_PI*p.mu[0]));               //U_sh_e = 1./sqrt(-2.*M_PI*mu[0])*EXP(-phi)
    dg::blas1::axpby( 1.0, one, 0., ush[1]);                           //U_sh_i = 1.

    for( unsigned i=0; i<2; i++)
    {
        //ExB dynamics
        arakawa( y[i], phi[i], yp[i]);  //[N-1,phi]_RZ
        arakawa( y[i+2], phi[i], yp[i+2]);//[U,phi]_RZ  
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);                        // dtN =1/B [N,phi]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);                    // dtU =1/B [U,phi]_RZ  
        
        //Parallel dynamics
        dz_.set_boundaries( dg::NEU, 0, 0);                                  //dz N = 0 on limiter
        dz_(y[i], dzy[i]);       
//         dz_.set_boundaries( dg::DIR,  ush[i],-1.0,1.0);                      //dz U = {1./sqrt(-2.*M_PI*mu[0])*EXP(-phi),1} on limiter
        dz_.set_boundaries( dg::NEU, 0, 0);                                  //dz N = 0 on limiter
        dz_(y[i+2], dzy[i+2]);                                               //dz U

//         dg::blas1::pointwiseDot(npe[i], ush[i], omega);                      // U N on limiter
//         dz_.set_boundaries( dg::DIR, omega, -1.0,1.0);                       // dz U N = {ne/sqrt(-2.*M_PI*mu[0])*EXP(-phi),ne}  on
        dz_.set_boundaries( dg::NEU, 0, 0);  
        dg::blas1::pointwiseDot(npe[i], y[i+2], omega);                      // U N
        dz_(omega, chi);                                                     // dz UN
        dg::blas1::pointwiseDot(omega, gradlnB, omega);                      // U N dz ln B
        dg::blas1::axpby( -1., chi, 1., yp[i]);                              // dtN = dtN - dz U N
        dg::blas1::axpby( 1., omega, 1., yp[i]);                             // dtN = dtN + U N dz ln B
        //parallel force terms
        dz_.set_boundaries( dg::NEU, 0, 0); 
        dz_(logn[i], omega);                                                //dz lnN
        dg::blas1::axpby( -p.tau[i]/p.mu[i], omega, 1., yp[2+i]); //dtU = dtU - tau/(hat(mu))*dz lnN
        dz_.set_boundaries( dg::DIR, 0, 0); 

        dz_(phi[i], omega);                                             //dz psi
        dg::blas1::axpby( -1./p.mu[i], omega, 1., yp[2+i]);   //dtU = dtU - 1/(hat(mu))  *dz psi  

//         dg::blas1::pointwiseDot( ush[i], ush[i], omega); 
//         dz_.set_boundaries( dg::DIR, omega, 1.0,1.0); 
                dz_.set_boundaries( dg::NEU, 0, 0); 
        dg::blas1::pointwiseDot(y[i+2],y[i+2], omega);                  //U^2
        dz_(omega, chi);                                                //dz U^2
        dg::blas1::axpby( -0.5, chi, 1., yp[2+i]);                      //dtU = dtU - 0.5 dz U^2
        
        //Curvature dynamics: 
        curve( y[i], curvy[i]);                                          //K(N) = K(N-1)
        curve( y[i+2], curvy[2+i]);                                     //K(U) 
        curve( phi[i], curvphi[i]);                                     //K(phi) 
        
        dg::blas1::pointwiseDot(y[i+2], curvy[2+i], omega);             //U K(U) 
        dg::blas1::pointwiseDot( y[i+2], omega, chi);                   //U^2 K(U)
        dg::blas1::pointwiseDot( npe[i], omega, omega);                   //N U K(U)
        
        dg::blas1::axpby( -p.mu[i], omega, 1., yp[i]);        //dtN = dtN - (hat(mu)) N U K(U)
        dg::blas1::axpby( -0.5*p.mu[i], chi, 1., yp[2+i]);    //dtU = dtU - 0.5 (hat(mu)) U^2 K(U)

        curve( logn[i], omega);                                         //K(ln N) 
        dg::blas1::pointwiseDot(y[i+2], omega, omega);                  //U K(ln N)
        dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);               //dtU = dtU - tau U K(lnN)
        
        dg::blas1::pointwiseDot( y[i+2], curvy[i], omega);              //U K( N)
        dg::blas1::pointwiseDot( y[i+2], omega, chi);                   //U^2K( N)
        dg::blas1::axpby( -0.5*p.mu[i], chi, 1., yp[i]);      //dtN = dtN - 0.5 mu U^2 K(N)

        dg::blas1::axpby( -p.tau[i], curvy[i], 1., yp[i]);              //dtN = dtN - tau K(N)
        dg::blas1::axpby( -2.*p.tau[i], curvy[2+i], 1., yp[2+i]);       //dtU = dtU - 2 tau K(U)
        dg::blas1::pointwiseDot(npe[i],curvphi[i], omega);                //N K(psi)
        dg::blas1::axpby( -1., omega, 1., yp[i]);                       //dtN= dtN - N K(psi)

        dg::blas1::pointwiseDot( y[i+2], curvphi[i], omega);            //U K(phi)
        dg::blas1::axpby( -0.5, omega, 1., yp[2+i]);                    //dtU = dtU -0.5 U K(psi)

        //Parallel dissipation
        dz_.set_boundaries( dg::NEU, 0, 0);
        dz_.dzz(y[i],omega);                                            //dz^2 N 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i]);             
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB, dzy[i], omega);                // dz lnB dz N
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i]);    
        
        
//         dz_.set_boundaries( dg::DIR,ush[i],-1.0,1.0);  
                dz_.set_boundaries( dg::NEU, 0, 0);

        dz_.dzz(y[i+2],omega);                                          //dz^2 U 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i+2]);           
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB,dzy[i+2], omega);               // dz lnB dz U
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i+2]);    
        
        //damping 
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]);
        dg::blas1::pointwiseDot( damping, yp[i+2], yp[i+2]); 

    }
    t.toc();
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif 
    std::cout << "One rhs took "<<t.diff()<<"s\n";


    //add particle source to dtN
//     for( unsigned i=0; i<2; i++)
//     {
//         dg::blas1::pointwiseDivide( source, expy[i], omega); //source/N
//         dg::blas1::axpby( 1., omega, 1, yp[i]  );       //dtlnN = dtlnN + source/N
//     }

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

///@}

} //namespace eule
