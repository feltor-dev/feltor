#pragma once

#include "dg/algorithm.h"

#include "feltor/parameters.h"
// #include "geometry_circ.h"
#include "solovev/geometry.h"
#include "solovev/init.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif //DG_BENCHMARK

// #define APAR
namespace eule
{
template<class Matrix, class container, class Preconditioner>
struct Rolkar
{
    template<class Grid3d>
    Rolkar( const Grid3d& g, eule::Parameters p, solovev::GeomParameters gp):
        p(p),
        gp(gp),
        temp( dg::evaluate(dg::zero, g)),
        dampprof_( dg::evaluate( solovev::GaussianProfDamping( gp), g)),
        dampgauss_( dg::evaluate( solovev::GaussianDamping( gp), g)),
        LaplacianM_perp ( g, dg::normed, dg::centered)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
           x[1] := N_i - 1
           x[2] := w_e
           x[3] := w_i
        */
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<4; i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
        }
        //cut contributions to boundary now with damping on all 4 quantities
        for( unsigned i=0; i<y.size(); i++)
        {
            dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
        }
    }
    dg::Elliptic<Matrix,container,Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const Preconditioner& weights(){return LaplacianM_perp.weights();}
    const Preconditioner& precond(){return LaplacianM_perp.precond();}
    const container& damping(){return dampprof_;}
  private:
    const eule::Parameters p;
    const solovev::GeomParameters gp;
    container temp;
    const container dampprof_;
    const container dampgauss_;
    dg::Elliptic<Matrix, container, Preconditioner> LaplacianM_perp;
};

template< class Matrix, class container=thrust::device_vector<double>, class Preconditioner = thrust::device_vector<double> >
struct Feltor
{
    typedef typename dg::VectorTraits<container>::value_type value_type;

    template<class Grid3d>

    Feltor( const Grid3d& g,  eule::Parameters,solovev::GeomParameters gp);
    dg::DZ<Matrix, container> dz(){return dz_;}

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
    const container& aparallel( ) const { return apar;}
    const std::vector<container>& uparallel( ) const { return u;}
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
    container& induct(const std::vector<container>& y);//solves induction equation

    container chi, omega,lambda;//1d container
    container apar,rho;//1d container

    const container binv, curvR, curvZ, gradlnB;
    const container source, damping, one;
    const Preconditioner w3d, v3d;
    container curvapar;
    std::vector<container> phi,u, u2, npe, logn,un,curvphi, dzphi, dzlogn,dzun,dzu2; //2d container
    std::vector<container> arakan,arakaw,arakaun,arakalogn,arakAphi,arakau2; //2d container
    std::vector<container> dzy, curvy;  //4d container

    //matrices and solvers
    dg::DZ<Matrix, container> dz_;
    dg::ArakawaX< Matrix, container>    arakawa; 
    //dg::Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector
    dg::Elliptic<  Matrix, container, Preconditioner > pol,lapperp; //note the host vector
    dg::Helmholtz< Matrix, container, Preconditioner > maxwell, invgamma;
    dg::Invert<container> invert_maxwell, invert_pol, invert_invgamma;

    const  eule::Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;
    std::vector<double> evec;

};

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),lambda(chi),  //1d container
    rho( chi), apar(chi), curvapar(chi),
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    curvR( dg::evaluate( solovev::CurvatureR(gp), g)),
    curvZ( dg::evaluate(solovev::CurvatureZ(gp), g)),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    source( dg::evaluate(solovev::TanhSource(p, gp), g)),
    damping( dg::evaluate( solovev::GaussianDamping(gp ), g)), 
    phi( 2, chi), u(phi),   u2(phi), npe(phi), logn(phi), un(phi),  //2d container
    curvphi( phi), dzphi(phi),dzlogn(phi), dzun(phi),dzu2(phi),
    arakan( phi), arakaw( phi),arakaun(phi),arakalogn(phi), arakAphi(phi),arakau2(phi), 
    dzy( 4, chi), curvy(dzy), //4d container
    dz_(solovev::Field(gp), g, gp.rk4eps,solovev::PsiLimiter(gp)),
    arakawa( g), 
    pol(     g, dg::not_normed, dg::centered), 
    lapperp ( g, dg::normed, dg::centered),
    invgamma(g,-0.5*p.tau[1]*p.mu[1],dg::centered),
    invert_pol( omega, omega.size(), p.eps_pol), 
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)), one(dg::evaluate( dg::one, g)),
    maxwell(g, 1., dg::centered), //sign is already correct!
    invert_maxwell(omega, omega.size(), p.eps_maxwell),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    p(p),
    gp(gp),
    evec(5)
{ }

template<class Matrix, class container, class P>
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

//computes and modifies expy!!
template<class Matrix, class container, class P>
container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-1) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]));
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
    pol.set_chi( chi);

    unsigned numberg    =  invert_invgamma(invgamma,chi,y[1]); //chi= Gamma (Ni-1)
    //Set gamma_ni <- do not change afterwards
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi

    if( number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template<class Matrix, class container, class P>
container& Feltor< Matrix, container, P>::induct(const std::vector<container>& y)
{
    dg::blas1::axpby( p.beta/p.mu[0], npe[0], 0., chi); //chi = beta/mu_e N_e
    dg::blas1::axpby(- p.beta/p.mu[1],  npe[1], 1., chi); //chi =beta/mu_e N_e-beta/mu_i  N_i
    maxwell.set_chi(chi);
    dg::blas1::pointwiseDot( npe[0], y[2], chi);                 //chi = n_e w_e
    dg::blas1::pointwiseDot( npe[1], y[3], lambda);               //lambda = n_i w_i
    dg::blas1::axpby( -1.,lambda , 1., chi);  //chi = -n_i w_i + n_e w_e
    //maxwell = (lap_per - beta*(N_i/hatmu_i - n_e/hatmu_e)) A_parallel 
    //chi=n_e w_e -N_i w_i
    unsigned number = invert_maxwell( maxwell, apar, chi); //omega equals a_parallel
    if( number == invert_maxwell.get_max())
        throw dg::Fail( p.eps_maxwell);
    return apar;
}

// #endif
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{   
    /*  y[0] := N_e - 1
        y[1] := N_i - 1
        y[2] := w_e
        y[3] := w_i
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    
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
    }

    //compute phi via polarisation
    phi[0] = polarisation( y); //computes phi and Gamma n_i
    phi[1] = compute_psi( phi[0]); //sets omega = u_E^2
    //compute A_parallel via induction and compute U_e and U_i from it
    apar = induct(y); //computes a_par and needs correct npe

    //calculate U from Apar and w
    dg::blas1::axpby( 1., y[2], - p.beta/p.mu[0], apar, u[0]); // U_e = w_e -beta/mu_e A_parallel
    dg::blas1::axpby( 1., y[3], - p.beta/p.mu[1], apar, u[1]); // U_i = w_i -beta/mu_i A_parallel

    for(unsigned i=0; i<2; i++)
    {
        S[i]    = z[i]*p.tau[i]*dg::blas2::dot( logn[i], w3d, npe[i]);
        dg::blas1::pointwiseDot( u[i], u[i], chi); 
        Tpar[i] = z[i]*0.5*p.mu[i]*dg::blas2::dot( npe[i], w3d, chi);
    }
    mass_ = dg::blas2::dot( one, w3d, y[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( npe[1], w3d, omega);   //= 0.5 mu_i N_i u_E^2
     arakawa.bracketS( apar,apar, omega);    
//     arakawa.variation(apar,omega); // (dx A_parallel)^2 + (dy A_parallel)^2
    double Uapar = p.beta*dg::blas2::dot( one, w3d, omega);
    //energytheorem
    energy_ = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1] + Uapar; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp, evec[3] = Tpar[0], evec[4] = Tpar[1];
     
    //// the resistive dissipative energy
    dg::blas1::pointwiseDot( npe[0], u[0], omega); //N_e U_e 
    dg::blas1::pointwiseDot( npe[1], u[1], chi);  //N_i U_i
    dg::blas1::axpby( -1., omega, 1., chi); //chi  = + N_i U_i -N_e U_e
    dg::blas1::axpby( -1., u[0], 1., u[1], omega); //lambda  = -N_e U_e + N_e U_i   
    double Dres = -p.c*dg::blas2::dot(omega, w3d, chi); //- C*(N_i U_i + N_e U_e)(U_i - U_e)    
    
    curve( apar, curvapar); //K(A_parallel)
    dg::blas1::axpby(  1.,  gradlnB,0.5*p.beta,  curvapar,curvapar);  // dz ln B + 0.5 beta K(A_parallel) factor 0.5 or not?
    for( unsigned i=0; i<2; i++)
    {
        //ExB dynamics
        arakawa( y[i], phi[i], yp[i]);                             //[N-1,phi]_RZ
        arakawa( y[i+2], phi[i], yp[i+2]);                         //[U+beta/nu A_parallel,phi]_RZ 
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);              // dtN =1/B [N,phi]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);          // dtU =1/B [w,phi]_RZ  
        // compute A_parallel terms of the parallel derivatuve
        arakawa( apar, y[i], arakan[i]);                           // [A_parallel,N]_RZ
        arakawa( apar, y[i+2], arakaw[i]);                         // [A_parallel,w]_RZ
        arakawa( apar, logn[i], arakalogn[i]);                     // [A_parallel,logN]_RZ
        dg::blas1::pointwiseDot( npe[i], u[i], un[i]);             // UN
        arakawa( apar, un[i], arakaun[i]);                         // [A_parallel,UN]_RZ
        dg::blas1::pointwiseDot( u[i], u[i], u2[i]);               // U^2
        arakawa( apar, u2[i], arakau2[i]);                         // [A_parallel,U^2]_RZ
        dg::blas1::pointwiseDot(arakan[i],binv,arakan[i]);         //1/B [A_parallel,N]_RZ
        dg::blas1::pointwiseDot(arakaw[i],binv,arakaw[i]);         //1/B [A_parallel,U]_RZ
        dg::blas1::pointwiseDot(arakalogn[i],binv, arakalogn[i]);  //1/B [A_parallel,logN]_RZ
        dg::blas1::pointwiseDot(arakaun[i],binv,arakaun[i]);       //1/B [A_parallel,UN]_RZ
        dg::blas1::pointwiseDot(arakau2[i],binv,arakau2[i]);       //1/B [A_parallel,U^2]_RZ

        //Parallel dynamics
        dz_.set_boundaries( dg::NEU, 0, 0);                                  // dz N = 0 on limiter
        dz_(y[i], dzy[i]);                                                   // dz N   (for dissi)
        dz_.set_boundaries( dg::NEU, 0, 0);                                  // dz w = 0 on limiter
        dz_(y[i+2], dzy[i+2]);                                               // dz w  (for dissi)
        dz_.set_boundaries( dg::NEU, 0, 0);                                  // dz UN = 0 on limiter  
        dz_(un[i], dzun[i]);                                                 // dz UN
        dz_.set_boundaries( dg::NEU, 0, 0);                                  // dz lnN = 0 on Limiter
        dz_(logn[i], dzlogn[i]);                                             // dz lnN
        dz_.set_boundaries( dg::DIR, 0, 0);                                  // psi=0 on limiter
        dz_(phi[i], dzphi[i]);                                                // dz psi
        dz_.set_boundaries( dg::NEU, 0, 0);                                  // dz U^2 = 0 on limiter  
        dz_(u2[i],dzu2[i]);                                                  // dz U^2
        
        //add A_parallel terms to the parallel derivatives to obtain dz^b
        dg::blas1::axpby(-p.beta,arakan[i]   ,1.,dzy[i]);     // dz^b N    = dz N -beta/B [A_parallel,lnN ] //enters dissi
        dg::blas1::axpby(-p.beta,arakaw[i]   ,1., dzy[i+2]);   // dz^b U    = dz U -beta/B [A_parallel,w ] //enters dissi
        dg::blas1::axpby(-p.beta,arakalogn[i],1.,dzlogn[i]);// dz^b logN = dz logN   -beta/B [A_parallel,logN ]
        dg::blas1::axpby(-p.beta,arakaun[i]  ,1.,dzun[i]);  // dz^b UN= dz UN   -beta/B [A_parallel,UN ]
        dg::blas1::axpby(-p.beta,arakau2[i]  ,1.,dzu2[i]);  // dz^b U^2= dz U^2   -beta/B [A_parallel,U^2 ] 

        //Add terms         
        dg::blas1::pointwiseDot(un[i], curvapar, omega);                     // U N dz^b ln B
        dg::blas1::axpby( 1. , omega, 1., yp[i]);                           // dtN = dtN + U N dz^b ln B
        dg::blas1::axpby( -1., dzun[i], 1., yp[i]);                         // dtN = dtN - dz^b U N
        dg::blas1::axpby( -p.tau[i]/p.mu[i], dzlogn[i], 1., yp[2+i]);       // dtw = dtw - tau/(hat(mu))*dz^b lnN
        dg::blas1::axpby( -1./p.mu[i],dzphi[i], 1., yp[2+i]);               // dtw = dtw - 1/(hat(mu))  *dz^b psi  
        dg::blas1::axpby( -0.5, dzu2[i], 1., yp[2+i]);                      // dtw = dtw - 0.5 dz U^2

        //curvature terms
        curve( npe[i], curvy[i]);                                       //K(N) = K(N)
        curve( u[i], curvy[2+i]);                                       //K(U) 
        curve( phi[i], curvphi[i]);                                     //K(phi) 
        
        dg::blas1::pointwiseDot( u2[i],curvy[2+i], chi);                //U^2 K(U)
        dg::blas1::pointwiseDot( un[i],curvy[2+i], omega);              //N U K(U)
        
        dg::blas1::axpby( -p.mu[i], omega, 1., yp[i]);                  //dtN = dtN - (hat(mu)) N U K(U)
        dg::blas1::axpby( -0.5*p.mu[i], chi, 1., yp[2+i]);              //dtw = dtw - 0.5 (hat(mu)) U^2 K(U)

        curve( logn[i], omega);                                         //K(ln N) 
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
        dz_.dzz(y[i+2],omega);                                          //dz^2 w 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i+2]);      

        // for energytheorem 
        dg::blas1::axpby( p.nu_parallel, omega, 0., lambda,lambda);     //lambda = nu_para*dz^2 w 
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB,dzy[i+2], omega);               // dz lnB dz w
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i+2]);    

         // for energytheorem 
        dg::blas1::axpby(-p.nu_parallel, omega, 1., lambda,lambda);     // lambda += nu_para*dz lnB dz N     
        //Compute parallel and perp dissipative energy for U
        dg::blas1::pointwiseDot( npe[i], y[i+2], omega); //N U   
        Dpar[i+2] = z[i]*p.mu[i]*dg::blas2::dot(omega, w3d, lambda);      //Z*N*U nu_para *(dz^2 w -dz lnB dz w)  
        dg::blas2::gemv( lapperp, y[i+2], lambda);
        dg::blas2::gemv( lapperp, lambda,chi);//nabla_RZ^4 U
        Dperp[i+2] = -z[i]*p.mu[i]*p.nu_perp* dg::blas2::dot(omega, w3d, chi);

        //Resistivity (explicit)
        dg::blas1::axpby( 1., u[1], -1, u[0], omega); //U_i - U_w
        dg::blas1::axpby( -p.c/p.mu[i], omega, 1., yp[i+2]);  //- C/mu_e (U_i - U_e)
        
        //damping 
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]);
//         dg::blas1::pointwiseDot( damping, u[i], u[i]); 
        dg::blas1::pointwiseDot( damping, yp[i+2], yp[i+2]); 

    }

    ediff_= Dpar[0]+Dperp[0]+Dpar[1]+Dperp[1]+Dpar[2]+Dperp[2]+Dpar[3]+Dperp[3] + Dres;

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




} //namespace eule
