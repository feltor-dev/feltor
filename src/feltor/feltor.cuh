#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
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
 * @brief Implicit (perpendicular diffusive) terms for Feltor solver
 *
 * @tparam Matrix The Matrix class
 * @tparam container The Vector class 
 * @tparam Preconditioner The Preconditioner class
 */
template<class Matrix, class container, class Preconditioner>
struct Rolkar
{

    /**
     * @brief 
     *
     * @tparam Grid3d
     * @param g
     * @param p
     * @param gp
     */
    template<class Grid3d>
    Rolkar( const Grid3d& g, eule::Parameters p, solovev::GeomParameters gp):
        p(p),
        gp(gp),
        temp( dg::evaluate(dg::zero, g)),  omega(temp),        
        dampprof_( dg::evaluate( solovev::GaussianProfXDamping( gp), g)),
        dampgauss_( dg::evaluate( solovev::GaussianDamping( gp), g)),
        LaplacianM_perpN ( g,g.bcx(),g.bcy(), dg::normed, dg::centered),
        LaplacianM_perpDIR ( g,dg::DIR, dg::DIR, dg::normed, dg::centered)
    {
    }

    /**
     * @brief Return implicit terms
     *
     * @param x input vector
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
        for( unsigned i=0; i<2; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perpN, x[i], temp);
            dg::blas2::gemv( LaplacianM_perpN, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
            dg::blas2::gemv( LaplacianM_perpDIR, x[i+2], temp);
            dg::blas2::gemv( LaplacianM_perpDIR, temp, y[i+2]);
            dg::blas1::scal( y[i+2], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 

        }
        //Resistivity
        dg::blas1::axpby( 1., x[3], -1, x[2], omega); //U_i - U_e
        dg::blas1::axpby( -p.c/p.mu[0], omega, 1., y[2]);  //- C/mu_e (U_i - U_e)
        dg::blas1::axpby( -p.c/p.mu[1], omega, 1., y[3]);  //- C/mu_i (U_i - U_e)
        //damping
        for( unsigned i=0; i<y.size(); i++){
           dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
        }
    }

    dg::Elliptic<Matrix, container, Preconditioner>& laplacianM() {return LaplacianM_perpDIR;}

    const Preconditioner& weights(){return LaplacianM_perpDIR.weights();}
    const Preconditioner& precond(){return LaplacianM_perpDIR.precond();}
    const container& damping(){return dampprof_;}
  private:
    const eule::Parameters p;
    const solovev::GeomParameters gp;
    container temp, omega;
    const container dampprof_;
    const container dampgauss_;
    
    dg::Elliptic<Matrix, container, Preconditioner> LaplacianM_perpN,LaplacianM_perpDIR;

};

/**
 * @brief compute explicit terms
 *
 * @tparam Matrix matrix class to use
 * @tparam container main container to hold the vectors
 * @tparam Preconditioner class of the weights
 */
template< class Matrix, class container=thrust::device_vector<double>, class Preconditioner = thrust::device_vector<double> >
struct Feltor
{
    template<class Grid3d>
    Feltor( const Grid3d& g, eule::Parameters p,solovev::GeomParameters gp);

    dg::DZ<Matrix, container> dz(){return dzN_;}

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

    void energies( std::vector<container>& y);

  private:
//     void curve( container& y, container& target);
    void curveN( container& y, container& target);
    void curveDIR( container& y, container& target);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation
    void add_parallel_dynamics( std::vector<container>& y, std::vector<container>& yp);

    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!

    const container binv, curvR, curvZ;
    container gradlnB;
    const container source, damping, one;
    const Preconditioner w3d, v3d;

    std::vector<container> phi, curvphi;
    std::vector<container> npe, logn;
    std::vector<container> dzy, curvy; 

    //matrices and solvers
    dg::DZ<Matrix, container> dzDIR_;
    dg::DZ<Matrix, container> dzN_;
    dg::Poisson< Matrix, container> poissonN,poissonDIR; 

    dg::Elliptic< Matrix, container, Preconditioner > pol,lapperpN,lapperpDIR; 
    dg::Helmholtz< Matrix, container, Preconditioner > invgammaDIR;
    dg::Helmholtz< Matrix, container, Preconditioner > invgammaN;

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
    phi( 2, chi), curvphi( phi),  npe(phi), logn(phi),
    dzy( 4, chi),curvy(dzy), 
    dzDIR_(solovev::Field(gp), g, 2.*M_PI/(double)p.Nz, gp.rk4eps,solovev::PsiLimiter(gp), dg::DIR),
    dzN_(solovev::Field(gp), g, 2.*M_PI/(double)p.Nz, gp.rk4eps,solovev::PsiLimiter(gp), g.bcx()),
    poissonN(g, g.bcx(), g.bcy(), dg::DIR, dg::DIR), //first N/U then phi BCC
    poissonDIR(g, dg::DIR, dg::DIR, dg::DIR, dg::DIR), //first N/U then phi BCC
    pol(    g, dg::DIR, dg::DIR, dg::not_normed,          dg::centered), 
    lapperpN ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
    lapperpDIR ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
    invgammaDIR( g,dg::DIR, dg::DIR,-0.5*p.tau[1]*p.mu[1],dg::centered),
    invgammaN(  g,g.bcx(), g.bcy(),-0.5*p.tau[1]*p.mu[1],dg::centered),
    invert_pol(      omega, omega.size(), p.eps_pol),
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

    invert_invgamma(invgammaN,chi,y[1]); //omega= Gamma (Ni-1)    
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi
        if(  number == invert_pol.get_max())
            throw dg::Fail( p.eps_pol);
    return phi[0];
}

template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_psi( container& potential)
{
    invert_invgamma(invgammaDIR,chi,potential);                    //chi  Gamma phi
    poissonN.variationRHS(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}

template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::initializene( const container& src, container& target)
{ 
    invert_invgamma(invgammaN,target,src); //=ne-1 = Gamma (ni-1)    
}

template<class M, class V, class P>
void Feltor<M, V, P>::energies( std::vector<V>& y)
{
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
        dg::blas1::transform( npe[i], logn[i], dg::LN<double>());
        S[i]    = z[i]*p.tau[i]*dg::blas2::dot( logn[i], w3d, npe[i]);
        dg::blas1::pointwiseDot( y[i+2], y[i+2], chi); 
        Tpar[i] = z[i]*0.5*p.mu[i]*dg::blas2::dot( npe[i], w3d, chi);
    }
    mass_ = dg::blas2::dot( one, w3d, y[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( npe[1], w3d, omega);   //= 0.5 mu_i N_i u_E^2
    energy_ = S[0] + S[1]  + Tperp + Tpar[0] + Tpar[1]; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp, evec[3] = Tpar[0], evec[4] = Tpar[1];
    //// the resistive dissipative energy
    dg::blas1::pointwiseDot( npe[0], y[2], omega); //N_e U_e 
    dg::blas1::pointwiseDot( npe[1], y[3], chi);  //N_i U_i
    dg::blas1::axpby( -1., omega, 1., chi); //chi  = + N_i U_i -N_e U_e
    dg::blas1::axpby( -1., y[2], 1., y[3], omega); //omega  = - U_e + U_i   
    double Dres = -p.c*dg::blas2::dot(omega, w3d, chi); //- C*(N_i U_i + N_e U_e)(U_i - U_e)
    for( unsigned i=0; i<2;i++)
    {
        //Compute parallel dissipative energy for N/////////////////////////////
        if( p.nu_parallel != 0)
        {
                if (p.pollim==1) dzN_.set_boundaries( p.bc, 0, 0);                
                if (p.pardiss==0)
                {
                    dzN_.forward(y[i], omega); 
                    dzN_.forwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 0.,lambda,lambda); 

                    dzN_.backward( y[i], omega); 
                    dzN_.backwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda,  1., lambda,lambda); 
                }
                if (p.pardiss==1)
                {
                    dzN_.dzz(y[i],omega);                                            //dz^2 N 
                    dg::blas1::axpby( p.nu_parallel, omega, 0., lambda,lambda);     //lambda = nu_para*dz^2 N 
                    dzN_(y[i], dzy[i]);       
                    dg::blas1::pointwiseDot(gradlnB, dzy[i], omega);                // dz lnB dz N
                    dg::blas1::axpby(-p.nu_parallel, omega, 1., lambda,lambda);     // lambda += nu_para*dz lnB dz N
                }           
        }
        dg::blas1::axpby(1.,one,1., logn[i] ,chi); //chi = (1+lnN_e)
        dg::blas1::axpby(1.,phi[i],p.tau[i], chi); //chi = (tau_e(1+lnN_e)+phi)
        dg::blas1::pointwiseDot(y[i+2],y[i+2], omega);  
        dg::blas1::axpby(0.5*p.mu[i], omega,1., chi); //chi = (tau_e(1+lnN_e)+phi + 0.5 mu U^2)
        if( p.nu_parallel != 0)
            Dpar[i] = z[i]*dg::blas2::dot(chi, w3d, lambda); //Z*(tau (1+lnN )+psi) nu_para *(dz^2 N -dz lnB dz N)
        else 
            Dpar[i] = 0;

        //Compute perp dissipation 
        dg::blas2::gemv( lapperpN, y[i], lambda);
        dg::blas2::gemv( lapperpN, lambda, omega);//nabla_RZ^4 N_e
        Dperp[i] = -z[i]* p.nu_perp*dg::blas2::dot(chi, w3d, omega);  

        //Compute parallel dissipative energy for U/////////////////////////////
        if( p.nu_parallel !=0)
        {
                if (p.pollim==1) dzDIR_.set_boundaries( dg::DIR, 0, 0);    
                if (p.pardiss==0)
                {
                    dzDIR_.forward(y[i+2], omega); 
                    dzDIR_.forwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 0.,lambda,lambda); 
                    dzDIR_.backward( y[i+2], omega); 
                    dzDIR_.backwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda,  1., lambda,lambda); 
                }
                if (p.pardiss==1)
                {
                    dzDIR_.dzz(y[i+2],omega);                                          //dz^2 U 
                    dg::blas1::axpby( p.nu_parallel, omega, 0., lambda,lambda);     //lambda = nu_para*dz^2 U 
                    //gradlnBcorrection
                    dzDIR_(y[i+2], dzy[i+2]);                                               //dz U
                    dg::blas1::pointwiseDot(gradlnB,dzy[i+2], omega);               // dz lnB dz U
                    dg::blas1::axpby(-p.nu_parallel, omega, 1., lambda,lambda);     // lambda += nu_para*dz lnB dz N   
                }   
        }
        dg::blas1::pointwiseDot( npe[i], y[i+2], omega); //N U   
        if( p.nu_parallel !=0)
            Dpar[i+2] = z[i]*p.mu[i]*dg::blas2::dot(omega, w3d, lambda);      //Z*N*U nu_para *(dz^2 U -dz lnB dz U)  
        else
            Dpar[i+2] = 0;

        //Compute perp dissipation 
        dg::blas2::gemv( lapperpDIR, y[i+2], lambda);
        dg::blas2::gemv( lapperpDIR, lambda,chi);//nabla_RZ^4 U
        Dperp[i+2] = -z[i]*p.mu[i]*p.nu_perp* dg::blas2::dot(omega, w3d, chi);


    }
    //Compute rhs of energy theorem
    ediff_= Dpar[0]+Dperp[0]+Dpar[1]+Dperp[1]+Dpar[2]+Dperp[2]+Dpar[3]+Dperp[3] + Dres;
}
template<class M, class V, class P>
void Feltor<M, V, P>::add_parallel_dynamics( std::vector<V>& y, std::vector<V>& yp)
{
    for(unsigned i=0; i<2; i++)
    {
        //Parallel dynamics
        if (p.pollim==1) dzN_.set_boundaries( p.bc, 0, 0);  //dz N  on limiter
        dzN_(y[i], chi);   
        dg::blas1::pointwiseDot(y[i+2], chi, omega);     // U dz N
        if (p.pollim==1) dzDIR_.set_boundaries( dg::DIR, 0, 0);  //dz U  on limiter
        dzDIR_(y[i+2], chi);  
        dg::blas1::pointwiseDot(npe[i], chi,chi);     // N dz U
        dg::blas1::axpby(1.0,chi,1.0,omega,chi);
        dg::blas1::pointwiseDot(npe[i], y[i+2], omega);     // U N
        dg::blas1::pointwiseDot(omega, gradlnB, omega);     // U N dz ln B
        dg::blas1::axpby( -1., chi, 1., yp[i]);             // dtN = dtN - dz U N
        dg::blas1::axpby( 1., omega, 1., yp[i]);            // dtN = dtN + U N dz ln B

        dg::blas1::pointwiseDot(y[i+2],y[i+2], omega);      //U^2
        dzDIR_(omega, chi);                                 //dz U^2
        dg::blas1::axpby( -0.5, chi, 1., yp[2+i]);          //dtU = dtU - 0.5 dz U^2
        //parallel force terms
        if (p.pollim==1) dzN_.set_boundaries( p.bc, 0, 0);  //dz N  on limiter
        dzN_(logn[i], omega);                                                //dz lnN
        dg::blas1::axpby( -p.tau[i]/p.mu[i], omega, 1., yp[2+i]); //dtU = dtU - tau/(hat(mu))*dz lnN

        if (p.pollim==1)   dzDIR_.set_boundaries( dg::DIR, 0, 0); //dz psi on limiter
        dzDIR_(phi[i], omega);                                             //dz psi
        dg::blas1::axpby( -1./p.mu[i], omega, 1., yp[2+i]);   //dtU = dtU - 1/(hat(mu))  *dz psi  

        //Parallel dissipation
        if( p.nu_parallel != 0)
        {
                if (p.pardiss==0)
                {
                    if (p.pollim==1) dzN_.set_boundaries( p.bc, 0, 0);
                    dzN_.forward(y[i], omega); 
                    dzN_.forwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[i]); 

                    dzN_.backward( y[i], omega); 
                    dzN_.backwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[i]); 
                    if (p.pollim==1) dzDIR_.set_boundaries( dg::DIR, 0, 0);         
                    dzDIR_.forward(y[i+2], omega); 
                    dzDIR_.forwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[i+2]); 

                    dzDIR_.backward( y[i+2], omega); 
                    dzDIR_.backwardT(omega,lambda);
                    dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[i+2]); 
                }
                if (p.pardiss==1)
                {
                    if (p.pollim==1) dzN_.set_boundaries( p.bc, 0, 0);
                    dzN_.dzz(y[i],omega);                                          //dz^2 N 
                    dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i]);       
        
                    //gradlnBcorrection
                    dzN_(y[i], dzy[i]);       
                    dg::blas1::pointwiseDot(gradlnB, dzy[i], omega);            // dz lnB dz N
                    dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i]);    
                    
                    if (p.pollim==1) dzDIR_.set_boundaries( dg::DIR, 0, 0);      
                    dzDIR_.dzz(y[i+2],omega);                                   
                    dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i+2]);      
        
                    //gradlnBcorrection
                    dzDIR_(y[i+2], dzy[i+2]);                                   //dz U
                    dg::blas1::pointwiseDot(gradlnB,dzy[i+2], omega);           // dz lnB dz U
                    dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i+2]); 
                }
        }
    }
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

    //transform compute n and logn and energies
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+1)); //npe = N+1
        dg::blas1::transform( npe[i], logn[i], dg::LN<double>());
    }

    for( unsigned i=0; i<2; i++)
    {
        //ExB dynamics

        poissonN( y[i], phi[i], yp[i]);  //[N-1,phi]_RZ
        poissonDIR( y[i+2], phi[i], yp[i+2]);//[U,phi]_RZ  
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);                        // dtN =1/B [N,phi]_RZ
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);                    // dtU =1/B [U,phi]_RZ  
        
        //Curvature dynamics: 
        curveN( y[i], curvy[i]);                                     //K(N) = K(N-1)
        curveDIR( y[i+2], curvy[2+i]);                                 //K(U) 
        curveDIR( phi[i], curvphi[i]);                                 //K(phi) 
        
        dg::blas1::pointwiseDot(y[i+2], curvy[2+i], omega);             //U K(U) 
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
    add_parallel_dynamics( y, yp);
    for( unsigned i=0; i<2; i++)
    {
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
void Feltor<Matrix, container, P>::curveN( container& src, container& target)
{
    container temp1(src);
    dg::blas2::gemv( poissonN.dxlhs(), src, target); //d_R src
    dg::blas2::gemv( poissonN.dylhs(), src, temp1);  //d_Z src
    dg::blas1::pointwiseDot( curvR, target, target); // C^R d_R src
    dg::blas1::pointwiseDot( curvZ, temp1, temp1);   // C^Z d_Z src
    dg::blas1::axpby( 1., temp1, 1., target ); // (C^R d_R + C^Z d_Z) src
}
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::curveDIR( container& src, container& target)
{
    container temp1(src);
    dg::blas2::gemv( poissonN.dxrhs(), src, target); //d_R src
    dg::blas2::gemv( poissonN.dyrhs(), src, temp1);  //d_Z src
    dg::blas1::pointwiseDot( curvR, target, target); // C^R d_R src
    dg::blas1::pointwiseDot( curvZ, temp1, temp1);   // C^Z d_Z src
    dg::blas1::axpby( 1., temp1, 1., target ); // (C^R d_R + C^Z d_Z) src
}
///@}

} //namespace eule

