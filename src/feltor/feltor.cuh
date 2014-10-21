#pragma once

#include "dg/algorithm.h"

#include "parameters.h"
// #include "geometry_circ.h"
#include "solovev/geometry.h"
#include "solovev/init.h"

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
        dampprof_( dg::evaluate( solovev::GaussianProfDamping( gp), g)),
        dampgauss_( dg::evaluate( solovev::GaussianDamping( gp), g)),
        LaplacianM_perp ( g, dg::normed, dg::centered)
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
        dg::blas1::axpby( 1., x[3], -1, x[2], omega); //U_i - U_e
        dg::blas1::axpby( -p.c/p.mu[0], omega, 1., y[2]);  //- C/mu_e (U_e - U_i)
        dg::blas1::axpby( -p.c/p.mu[1], omega, 1., y[3]);  //- C/mu_i (U_e - U_i)
        //damping
        for( unsigned i=0; i<y.size(); i++){
           dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
        }
    }
    dg::Elliptic<Matrix, container, Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const Preconditioner& weights(){return LaplacianM_perp.weights();}
    const Preconditioner& precond(){return LaplacianM_perp.precond();}
    const container& damping(){return dampprof_;}
  private:
    const eule::Parameters p;
    const solovev::GeomParameters gp;
    container temp, chi, omega;
    std::vector<container> expy;
    const container dampprof_;
    const container dampgauss_;
    
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
    std::vector<double> energy_vector( ) {return evec;}
    double energy_diffusion( ){ return ediff_;}

  private:
    void curve( container& y, container& target);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation

    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!


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

    dg::Elliptic< Matrix, container, Preconditioner > pol,lapperp; 
    dg::Helmholtz< Matrix, container, Preconditioner > invgamma;
    dg::Invert<container> invert_pol,invert_invgamma;

    const eule::Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;
    std::vector<double> evec;

};

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, eule::Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi), 
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    curvR( dg::evaluate( solovev::CurvatureR(gp), g)),
    curvZ( dg::evaluate(solovev::CurvatureZ(gp), g)),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    source( dg::evaluate(solovev::TanhSource(p, gp), g)),
    damping( dg::evaluate( solovev::GaussianDamping(gp ), g)), 
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)), 
    phi( 2, chi), curvphi( phi), expy(phi), npe(phi), logn(phi),ush(phi),
    dzy( 4, chi),curvy(dzy), 
    dz_(solovev::Field(gp), g, gp.rk4eps,solovev::PsiLimiter(gp)),
    arakawa( g), 
    pol(     g, dg::not_normed, dg::centered), 
    lapperp ( g, dg::normed, dg::centered),
    invgamma(g,-0.5*p.tau[1]*p.mu[1], dg::centered),
    invert_pol( omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    p(p),
    gp(gp),
    evec(5)
{ }

template<class Matrix, class container, class P>
container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-1) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]));
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
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
    phi[1] = compute_psi( phi[0]); //sets omega

    double z[2]    = {-1.0,1.0};
    double S[2]    = {0.0, 0.0};
    double Tpar[2] = {0.0, 0.0};
    double Dpar[4] = {0.0, 0.0,0.0,0.0};
    double Dperp[4] = {0.0, 0.0,0.0,0.0};
    //transform compute n and logn and energies
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+1)); //npe = N+1
        dg::blas1::transform( npe[i], logn[i], dg::LN<value_type>());
        S[i]    = z[i]*p.tau[i]*dg::blas2::dot( logn[i], w3d, npe[i]);
        dg::blas1::pointwiseDot( y[i+2], y[i+2], chi); 
        Tpar[i] = z[i]*0.5*p.mu[i]*dg::blas2::dot( npe[i], w3d, chi);
    }
    mass_ = dg::blas2::dot( one, w3d, y[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( npe[1], w3d, omega);   //= 0.5 mu_i N_i u_E^2
    //energytheorem
    energy_ = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1]; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp, evec[3] = Tpar[0], evec[4] = Tpar[1];
     
    //// the resistive dissipative energy
    dg::blas1::pointwiseDot( npe[0], y[2], omega); //N_e U_e 
    dg::blas1::pointwiseDot( npe[1], y[3], chi);  //N_i U_i
    dg::blas1::axpby( -1., omega, 1., chi); //chi  = + N_i U_i -N_e U_e
    dg::blas1::axpby( -1., y[2], 1., y[3], omega); //lambda  = -N_e U_e + N_e U_i   
    double Dres = -p.c*dg::blas2::dot(omega, w3d, chi); //- C*(N_i U_i + N_e U_e)(U_i - U_e)

    //the parallel part is done elsewhere
    //set U_sheath
//     dg::blas1::axpby( -1.0, phi[0], 0., ush[0]);                       //U_sh_e = - phi
//     dg::blas1::transform(  ush[0],ush[0], dg::EXP<value_type>());      //U_sh_e = EXP(-phi)                  
//     dg::blas1::scal(ush[0], 1.0/sqrt(-2.*M_PI*p.mu[0]));               //U_sh_e = 1./sqrt(-2.*M_PI*mu[0])*EXP(-phi)
//     dg::blas1::axpby( 1.0, one, 0., ush[1]);                           //U_sh_i = 1.
   

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
        dz_.set_boundaries( dg::NEU, 0, 0);                                  //dz U = 0 on limiter
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

        //Parallel dissipation + dissipation in energytheorem(perp &para dissi)
        dz_.set_boundaries( dg::NEU, 0, 0);
        dz_.dzz(y[i],omega);                                            //dz^2 N 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i]);       

        // for energytheorem 
        dg::blas1::axpby( p.nu_parallel, omega, 0., lambda,lambda);     //lambda = nu_para*dz^2 N 
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB, dzy[i], omega);                // dz lnB dz N
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i]);    
        // for energytheorem 
        dg::blas1::axpby(-p.nu_parallel, omega, 1., lambda,lambda);     // lambda += nu_para*dz lnB dz N
        //Compute parallel and perp dissipative energy for N
        dg::blas1::axpby(1.,one,1., logn[i] ,chi); //chi = (1+lnN_e)
        dg::blas1::axpby(1.,phi[i],p.tau[i], chi); //chi = (tau_e(1+lnN_e)+phi)
        Dpar[i] = z[i]*dg::blas2::dot(chi, w3d, lambda); //Z*(tau (1+lnN )+psi) nu_para *(dz^2 N -dz lnB dz N)
        dg::blas2::gemv( lapperp, y[i], lambda);
        dg::blas2::gemv( lapperp, lambda, omega);//nabla_RZ^4 N_e
        Dperp[i] = -z[i]* p.nu_perp*dg::blas2::dot(chi, w3d, omega);  
        
//         dz_.set_boundaries( dg::DIR,ush[i],-1.0,1.0);  
        dz_.set_boundaries( dg::NEU, 0, 0);
        dz_.dzz(y[i+2],omega);                                          //dz^2 U 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i+2]);      

        // for energytheorem 
        dg::blas1::axpby( p.nu_parallel, omega, 0., lambda,lambda);     //lambda = nu_para*dz^2 U 
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB,dzy[i+2], omega);               // dz lnB dz U
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i+2]);    

         // for energytheorem 
        dg::blas1::axpby(-p.nu_parallel, omega, 1., lambda,lambda);     // lambda += nu_para*dz lnB dz N     
        //Compute parallel and perp dissipative energy for U
        dg::blas1::pointwiseDot( npe[i], y[i+2], omega); //N U   
        Dpar[i+2] = z[i]*p.mu[i]*dg::blas2::dot(omega, w3d, lambda);      //Z*N*U nu_para *(dz^2 U -dz lnB dz U)  
        dg::blas2::gemv( lapperp, y[i+2], lambda);
        dg::blas2::gemv( lapperp, lambda,chi);//nabla_RZ^4 U
        Dperp[i+2] = -z[i]*p.mu[i]*p.nu_perp* dg::blas2::dot(omega, w3d, chi);

        //damping 
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]);
        dg::blas1::pointwiseDot( damping, yp[i+2], yp[i+2]); 

    }
    //Compute rhs of energy theorem
    ediff_= Dpar[0]+Dperp[0]+Dpar[1]+Dperp[1]+Dpar[2]+Dperp[2]+Dpar[3]+Dperp[3] + Dres;
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

