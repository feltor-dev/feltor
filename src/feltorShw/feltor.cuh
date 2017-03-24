#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "parameters.h"
#include "dg/average.h"

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
 * @tparam container The container class
 */

template<class Geometry, class Matrix, class container>
struct Rolkar
{
    Rolkar( const Geometry& g, eule::Parameters p):
        p(p),
        temp( dg::evaluate(dg::zero, g)),
        LaplacianM_perp ( g,g.bcx(),g.bcy(), dg::normed, dg::centered),
        LaplacianM_perp_phi ( g,p.bc_x_phi,g.bcy(), dg::normed, dg::centered)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - (bgamp+profamp)
           x[1] := N_i - (bgamp+profamp)
        */
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<2; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
        }
    }
    dg::Elliptic<Geometry, Matrix, container>& laplacianM() {return LaplacianM_perp_phi;}
    const container& weights(){return LaplacianM_perp.weights();}
    const container& precond(){return LaplacianM_perp.precond();}
  private:
    const eule::Parameters p;
    container temp;    
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perp,LaplacianM_perp_phi;

};

template< class Geometry, class Matrix, class container >
struct Feltor
{
    //typedef std::vector<container> Vector;
    typedef typename dg::VectorTraits<container>::value_type value_type;
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    //typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    Feltor( const Geometry& g, eule::Parameters p);


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
    double coupling( ) {return coupling_;}
    double charge( ) {return charge_;}

    std::vector<double> energy_vector( ) {return evec;}
//     std::vector<container>& probe_vector( ) {return probevec;}

    double energy_diffusion( ){ return ediff_;}
    double radial_transport( ){ return gammanex_;}

  private:
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation

    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!
    container neavg,netilde,nedelta,lognedelta,phiavg,phitilde,phidelta,Niavg; //dont use them as helper
    const container binv;
    const container one;
    const container w2d, v2d;
    std::vector<container> phi;
    std::vector<container> npe, logn; 

    //matrices and solvers
    dg::Poisson< Geometry, Matrix, container> poisson; 

    dg::Elliptic< Geometry, Matrix, container > pol,lapperp; 
    dg::Helmholtz< Geometry, Matrix, container > invgammaPhi,invgammaN;

    dg::Invert<container> invert_pol,invert_invgammaN,invert_invgammaPhi;
    
    dg::PoloidalAverage<container, container > polavg; //device int vectors would be better for cuda

    const eule::Parameters p;

    double mass_, energy_, diff_, ediff_,gammanex_,coupling_,charge_;
    std::vector<double> evec;
    
    container lhs,profne,profNi;
};

template<class Grid, class Matrix, class container>
Feltor<Grid, Matrix, container>::Feltor( const Grid& g, eule::Parameters p): 
    chi( dg::evaluate( dg::zero, g)), omega(chi),  lambda(chi), 
    neavg(chi),netilde(chi),nedelta(chi),lognedelta(chi),
    phiavg(chi),phitilde(chi),phidelta(chi),    Niavg(chi),
    binv( dg::evaluate( dg::LinearX( 0.0, 1.), g) ),
    one( dg::evaluate( dg::one, g)),    
    w2d( dg::create::weights(g)), v2d( dg::create::inv_weights(g)), 
    phi( 2, chi), npe(phi), logn(phi),
    poisson(g, g.bcx(), g.bcy(), p.bc_x_phi, g.bcy()), //first N/U then phi BCC
    pol(    g, p.bc_x_phi, g.bcy(), dg::not_normed,          dg::centered), 
    lapperp ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
    invgammaPhi( g,p.bc_x_phi, g.bcy(),-0.5*p.tau[1]*p.mu[1],dg::centered),
    invgammaN(  g,g.bcx(), g.bcy(),-0.5*p.tau[1]*p.mu[1],dg::centered),
    invert_pol(         omega, p.Nx*p.Ny*p.n*p.n, p.eps_pol),
    invert_invgammaN(   omega, p.Nx*p.Ny*p.n*p.n, p.eps_gamma),
    invert_invgammaPhi( omega, p.Nx*p.Ny*p.n*p.n, p.eps_gamma),
    polavg(g),
    p(p),
    evec(3),
    //damping and sources
    lhs(dg::evaluate(dg::TanhProfX(p.lx*p.sourceb,p.sourcew,-1.0,0.0,1.0),g)),
    profne(dg::evaluate(dg::ExpProfX(p.nprofileamp, p.bgprofamp,p.invkappa),g)),
    profNi(profne)
{
    dg::blas1::transform(profNi,profNi, dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); 
    initializene(profNi,profne); //ne = Gamma N_i
    dg::blas1::transform(profne,profne, dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); 
    dg::blas1::transform(profNi,profNi, dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); 
}

template< class Grid, class Matrix, class container>
container& Feltor<Grid, Matrix, container>::polarisation( const std::vector<container>& y)
{
   if (p.modelmode==0) {
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-(bgamp+profamp)) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]*(p.bgprofamp + p.nprofileamp))); //mu_i n_i
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
    pol.set_chi( chi);
    invert_invgammaN(invgammaN,chi,y[1]); //chi= Gamma (Ni-(bgamp+profamp))    
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-(bgamp+profamp)) -(n_e-(bgamp+profamp))
    charge_ = dg::blas2::dot(one,w2d,chi);
    //= Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi
        if(  number == invert_pol.get_max())
            throw dg::Fail( p.eps_pol);
  }
  if (p.modelmode==1) {
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-1) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]*(p.bgprofamp + p.nprofileamp)));
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, omega);       //omega = (\mu_i n_i ) /B^2

    invert_invgammaN(invgammaN,chi,y[1]); //chi= Gamma (Ni-1)    
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-(bgamp+profamp)) - (n_e-(bgamp+profamp)) = Gamma n_i - n_e
    charge_ = dg::blas2::dot(one,w2d,chi);

//     dg::blas1::pointwiseDivide(chi,omega,chi);              // B^2(Gamma n_i - n_e )/  (\mu_i n_i )
    polavg(omega,lambda);
    dg::blas1::pointwiseDivide(chi,lambda,chi);              // (Gamma n_i - n_e )/  (\mu_i <n_i/B^2> )
    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi
        if(  number == invert_pol.get_max())
            throw dg::Fail( p.eps_pol);
  }
  if (p.modelmode==2) {
    dg::blas1::axpby( p.mu[1], y[1], 0, omega);      //omega =  \mu_i (N_i_tilde)/B_0^2
    
    invert_invgammaN(invgammaN,chi,y[1]); //chi= Gamma (N_i_tilde)    
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-(bgamp+profamp)) - (n_e-(bgamp+profamp)) = Gamma n_i - n_e
    charge_ = dg::blas2::dot(one,w2d,chi);

    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma N_i_tilde -N_e_tilde = -nabla^2 phi
        if(  number == invert_pol.get_max())
            throw dg::Fail( p.eps_pol);
  }
  return phi[0];
}

template< class Grid, class Matrix, class container>
container& Feltor<Grid, Matrix, container>::compute_psi( container& potential)
{
    if (p.modelmode==0 || p.modelmode==1)
    {
        invert_invgammaPhi(invgammaPhi,chi,potential);                 //chi  = Gamma phi
        poisson.variationRHS(potential, omega);
        dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    }
    if (p.modelmode==2)
    {
        invert_invgammaPhi(invgammaPhi,phi[1],potential);                 //chi  = Gamma phi
    }
    return phi[1];    
}

template<class Grid, class Matrix, class container>
void Feltor<Grid, Matrix, container>::initializene( const container& src, container& target)
{ 
    invert_invgammaN(invgammaN,target,src); //=ne-1 = Gamma (ni-1)    
}

template<class Grid, class Matrix, class container>
void Feltor<Grid, Matrix, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{

    dg::Timer t;
    t.tic();
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //compute phi via polarisation
    
    if (p.modelmode==0 || p.modelmode==1)
    {
        /* y[0] := N_e - (p.bgprofamp + p.nprofileamp)
           y[1] := N_i - (p.bgprofamp + p.nprofileamp)
        */
        phi[0] = polarisation( y);
        phi[1] = compute_psi( phi[0]); //sets omega

        //transform compute n and logn
        for(unsigned i=0; i<2; i++)
        {
            dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //npe = N+1
            dg::blas1::transform( npe[i], logn[i], dg::LN<value_type>());
        }    
        
        // START computation of energies
        double z[2]    = {-1.0,1.0};
        double S[2]    = {0.0, 0.0};
        double Dperp[2] = {0.0, 0.0};
        double Dsource[2] = {0.0, 0.0};
        //transform compute n and logn and energies
        for(unsigned i=0; i<2; i++)
        {
            S[i]    = z[i]*p.tau[i]*dg::blas2::dot( logn[i], w2d, npe[i]); // N LN N
        }
        mass_ = dg::blas2::dot( one, w2d, y[0] ); //take real ion density which is electron density!!
        double Tperp = 0.5*p.mu[1]*dg::blas2::dot( npe[1], w2d, omega);   //= 0.5 mu_i N_i u_E^2
        energy_ = S[0] + S[1]  + Tperp; 
        evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp;
        //Compute the perp dissipative energy 
        for( unsigned i=0; i<2;i++)
        {

            dg::blas1::axpby(1.,one,1., logn[i] ,chi); //chi = (1+lnN)
            dg::blas1::axpby(1.,phi[i],p.tau[i], chi); //chi = (tau_z(1+lnN)+phi)

            //---------- perp dissipation 
            dg::blas2::gemv( lapperp, y[i], lambda);
            dg::blas2::gemv( lapperp, lambda, omega);//nabla_RZ^4 N_e
            Dperp[i] = -z[i]* p.nu_perp*dg::blas2::dot(chi, w2d, omega);  //  tau_z(1+lnN)+phi) nabla_RZ^4 N_e
        }                
        //compute the radial electron density  transport
        dg::blas2::gemv( poisson.dyrhs(), phi[0], omega); //dy phi
        dg::blas1::pointwiseDot(omega,binv,omega); //1/B dy phi
        gammanex_ =-1.* dg::blas2::dot(npe[0],w2d,omega);//int(1/B N dy phi)       
        
        
        for( unsigned i=0; i<2; i++)
        {
            //ExB dynamics
            poisson( y[i], phi[i], yp[i]);  //[N-1,phi]_RZ
            dg::blas1::pointwiseDot( yp[i], binv, yp[i]);                        // dtN =1/B [N,phi]_RZ                
        }        
        //Coupling term for the electrons
        polavg(npe[0],neavg);
        dg::blas1::axpby(1.,npe[0],-1.,neavg,nedelta); // delta(ne) = ne-<ne> = <ne>tilde(ne)
        dg::blas1::scal(omega,0.0);
        if (p.hwmode==0) {
            dg::blas1::pointwiseDivide(npe[0],neavg,lambda); //lambda = ne/<ne> = 1+ tilde(ne)
            dg::blas1::transform(lambda, lambda, dg::LN<value_type>()); //lambda = ln(N/<N> )
            dg::blas1::axpby(1.,phi[0],p.tau[0],lambda,omega); //omega = phi - <phi> -  ln(N/<N> )
        }

        if (p.hwmode==1) {
            polavg(logn[0],lambda);       //<ln(ne)> 
            polavg(phi[0],phiavg);        //<phi>
            dg::blas1::axpby(1.,phi[0],-1.,phiavg,phidelta); // delta(phi) = phi - <phi>
            dg::blas1::axpby(1.,logn[0],-1.,lambda,lognedelta); // delta(ln(ne)) = ln(ne)- <ln(ne)>         
            dg::blas1::axpby(1.,phidelta,p.tau[0],lognedelta,omega); //omega = phi - lnNe
        }
        if (p.cmode==1) {
            dg::blas1::pointwiseDot(omega,npe[0],omega);  //(coupling)*Ne for constant resi
        }
        dg::blas1::axpby(p.alpha,omega,1.0,yp[0]);
        
        //---------- coupling energy
        dg::blas1::axpby(1.,one,1., logn[0] ,chi); //chi = (1+lnN)
        dg::blas1::axpby(1.,phi[0],p.tau[0], chi); //chi = (tau_e(1+lnN)+phi)   
        coupling_ =  z[0]*p.alpha* dg::blas2::dot(chi, w2d, omega);
        //Compute rhs of energy theorem
        ediff_= Dperp[0]+Dperp[1]+ coupling_ ;

        //Density source terms
        if (p.omega_source>1e-14) 
        {
            dg::blas1::axpby(1.0,profne,-1.0,neavg,lambda); //lambda = ne0p - <ne>
            //dtN_e
            dg::blas1::pointwiseDot(lambda,lhs,omega); //omega =lhs*(ne0p - <ne>)
            dg::blas1::transform(omega,omega, dg::POSVALUE<value_type>()); //= P [lhs*(n0ep - <ne>) ]
            dg::blas1::axpby(p.omega_source,omega,1.0,yp[0]);// dtne+= - omega_s P [lhs*(ne0p - <ne>) ]
            
            dg::blas1::axpby(1.,one,1., logn[0] ,chi); //chi = (1+lnNe)
            dg::blas1::axpby(1.,phi[0],p.tau[0], chi); //chi = (tau_e(1+lnNe)+phi)
            Dsource[0]=z[0]* p.omega_source*dg::blas2::dot(chi, w2d, omega);
            //add the FLR term (tanh and postrans before lapl seems to work because of cancelation) (LWL vorticity correction)
    //         dg::blas1::pointwiseDot(lambda,lhs,lambda);
    //         dg::blas1::transform(lambda,lambda, dg::POSVALUE<value_type>());   
    //         dg::blas2::gemv( lapperp, lambda, omega);
    //         dg::blas1::axpby(-p.omega_source*0.5*p.tau[1]*p.mu[1],omega,1.0,yp[0]); 

            //dt Ni without FLR
            dg::blas1::axpby(p.omega_source,omega,1.0,yp[1]);  //dtNi += omega_s* P [lhs*(ne0p - <ne>)]
            dg::blas1::axpby(1.,one,1., logn[1] ,chi); //chi = (1+lnNi)
            dg::blas1::axpby(1.,phi[1],p.tau[1], chi); //chi = (tau_i(1+lnNi)+psi)
            Dsource[1]=z[1]* p.omega_source*dg::blas2::dot(chi, w2d, omega);
            //add the FLR term (tanh and postrans before lapl seems to work because of cancelation)
            dg::blas1::pointwiseDot(lambda,lhs,lambda);
            dg::blas1::transform(lambda,lambda, dg::POSVALUE<value_type>());         
            dg::blas2::gemv( lapperp, lambda, omega);
            dg::blas1::axpby(p.omega_source*0.5*p.tau[1]*p.mu[1],omega,1.0,yp[1]); //dtNi +=- omega_s*0.5*tau_i nabla_perp^2 P [lhs*(ne0p - <ne>)]
            Dsource[1]+=z[1]* p.omega_source*0.5*p.tau[1]*p.mu[1]*dg::blas2::dot(chi, w2d, omega);

        }
        ediff_+=Dsource[0]+ Dsource[1];
    }
    if (p.modelmode==2 )
    {
        /* y[0] := n_e_tilde
           y[1] := N_i_tilde
        */
        phi[0] = polarisation( y);
        phi[1] = compute_psi( phi[0]); //sets omega
        
        
        // START computation of energies
        double z[2]    = {-1.0,1.0};
        double S[2]    = {0.0, 0.0};
        double Dperp[2] = {0.0, 0.0};
        double Dgrad[2] = {0.0, 0.0};
        //transform compute n and logn and energies
        for(unsigned i=0; i<2; i++)
        {
            S[i]    = 0.5*z[i]*p.tau[i]*dg::blas2::dot( y[i], w2d, y[i]); // N_tilde^2
        }
        mass_ = dg::blas2::dot( one, w2d, y[0] ); //take real ion density which is electron density!!
        poisson.variationRHS(phi[0], omega);
        double Tperp = 0.5*p.mu[1]*dg::blas2::dot( one, w2d, omega);   //= 0.5 mu_i u_E^2
        energy_ = S[0] + S[1]  + Tperp; 
        evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp;
        //Compute the perp dissipative energy 
        for( unsigned i=0; i<2;i++)
        {
            dg::blas1::axpby(1.,phi[i],p.tau[i],y[i], chi); //chi = (tau_z N_tilde +phi)

            //---------- perp dissipation 
            dg::blas2::gemv( lapperp, y[i], lambda);
            dg::blas2::gemv( lapperp, lambda, omega);//nabla_RZ^4 N_e
            Dperp[i] = -z[i]* p.nu_perp*dg::blas2::dot(chi, w2d, omega);  // - Z (tau N_tilde +psi ) nabla_RZ^4 N
        }   
        
        for( unsigned i=0; i<2; i++)
        {
            //ExB dynamics
            poisson( y[i], phi[i], yp[i]);  //dt N = 1/B[N_tilde,phi]_RZ
            
            //density gradient term
            dg::blas2::gemv( poisson.dyrhs(), phi[i], omega); //lambda = dy psi
            dg::blas1::axpby(-1./p.invkappa,omega,1.0,yp[i]);   // dt N_tilde += - kappa dy psi    
            
            Dgrad[i] = - z[i]*p.tau[i]/p.invkappa*dg::blas2::dot(y[i], w2d, omega);
            dg::blas1::pointwiseDot(omega,binv,omega); //1/B dy phi
            gammanex_ =-1.* dg::blas2::dot(one,w2d,omega);//int(1/B  dy phi)

        }
        
        //Coupling term for the electrons
        polavg(y[0],neavg); //<n_e_tilde>
        dg::blas1::axpby(1.,y[0],-1.,neavg,nedelta); // delta(n_e_tilde) = n_e_tilde-<n_e_tilde> 
        dg::blas1::scal(omega,0.0);
        if (p.hwmode==0) {
            dg::blas1::axpby(1.,phi[0],p.tau[0],y[0],omega); //omega = phi  - n_e_tilde )
        }
        if (p.hwmode==1) {
            polavg(phi[0],phiavg);        //<phi>
            dg::blas1::axpby(1.,phi[0], -1.,phiavg,phidelta); // delta(phi) = phi - <phi>
            dg::blas1::axpby(1.,phidelta,p.tau[0],nedelta,omega); //omega = delta(phi) - delta(n_e_tilde)
        }
        dg::blas1::axpby(p.alpha,omega,1.0,yp[0]);
        
        //compute coupling energy
        dg::blas1::axpby(1.,phi[0],p.tau[0],y[0], chi); //chi = (tau_z(1+invkappaN)+phi)
        coupling_ =  z[0]*p.alpha* dg::blas2::dot(chi, w2d, omega);
        //Compute rhs of energy theorem
        ediff_= Dperp[0]+Dperp[1]+ Dgrad[0]+Dgrad[1]+coupling_ ;
        
    }
    
    t.toc();
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif 
    std::cout << "One rhs took "<<t.diff()<<"s\n";
    
}

///@}

} //namespace eule

