#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "parameters.h"
#include <cusp/multiply.h>
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
 */

template<class Geometry, class Matrix, class container>
struct Rolkar
{
    Rolkar( const Geometry& g,eule::Parameters p):
        p(p),
        temp( dg::evaluate(dg::zero, g)),
        LaplacianM_perp ( g,g.bcx(),g.bcy(), dg::normed, dg::centered)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - (bgamp+profamp)
           x[1] := N_i - (bgamp+profamp)
           x[2] := T_e - (bgamp+profamp)
           x[3] := T_i - (bgamp+profamp)

        */
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<4; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
        }


    }
    dg::Elliptic<Geometry, Matrix, container>& laplacianM() {return LaplacianM_perp;}
    const container& weights(){return LaplacianM_perp.weights();}
    const container& precond(){return LaplacianM_perp.precond();}
  private:
    const eule::Parameters p;
    container temp;    
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perp;

};

template< class Geometry, class Matrix, class container>
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
    void initializene( const container& y, const container& helper, container& target);
    void initializepi( const container& y, const container& helper, container& target);

    void operator()( std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    std::vector<double> energy_vector( ) {return evec;}
    double energy_diffusion( ){ return ediff_;}

  private:
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi(const container& helper, container& potential);
    container& compute_chii(const container& helper, container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation

    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!
    container chii,uE2; //dont use them as helper
    const container binv;
    const container one;
    container B2;
    const container w2d, v2d;
    std::vector<container> phi; // =(phi,psi_i), (0,chi_i)
    std::vector<container> ype, logype; 

    //matrices and solvers
    dg::Poisson< Geometry, Matrix, container> poisson; 

    dg::Elliptic<   Geometry, Matrix, container> pol,lapperpM; 
    dg::Helmholtz<  Geometry, Matrix, container> invgamma1;    
    dg::Helmholtz2< Geometry, Matrix, container> invgamma2;
    dg::Invert<container> invert_pol,invert_invgammadag,invert_invgamma,invert_invgamma2;
    const eule::Parameters p;

    double mass_, energy_, diff_, ediff_;
    std::vector<double> evec;
};     

template<class Grid, class Matrix, class container>
Feltor<Grid, Matrix, container>::Feltor( const Grid& g, eule::Parameters p): 
    chi( dg::evaluate( dg::zero, g)), omega(chi),  lambda(chi), 
    binv( dg::evaluate( dg::LinearX( p.mcv, 1.), g) ),
    one( dg::evaluate( dg::one, g)),    
    B2( dg::evaluate( dg::one, g)),    
    w2d( dg::create::weights(g)), v2d( dg::create::inv_weights(g)), 
    phi( 2, chi),chii(chi),uE2(chi),// (phi,psi), (chi_i), u_ExB
    ype(4,chi), logype(ype), // y+(bgamp+profamp) , log(ype)
    poisson(g, g.bcx(), g.bcy(), g.bcx(), g.bcy()), //first N/U then phi BCC
    pol(    g, g.bcx(), g.bcy(), dg::not_normed,          dg::centered), 
    lapperpM ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
    invgamma1( g,g.bcx(), g.bcy(), -0.5*p.tau[1]*p.mu[1],dg::centered),
    invgamma2( g,g.bcx(), g.bcy(), -0.5*p.tau[1]*p.mu[1],dg::centered) ,
    invert_pol(      omega, omega.size(), p.eps_pol),
    invert_invgammadag( omega, omega.size(), p.eps_gamma),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    invert_invgamma2( omega, omega.size(), p.eps_gamma),
    p(p),
    evec(3)
{
    dg::blas1::pointwiseDivide(one,binv,B2);
    dg::blas1::pointwiseDivide(B2,binv,B2);
}

template<class G, class Matrix, class container>
container& Feltor<G, Matrix, container>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-(bgamp+profamp)) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]*(p.bgprofamp + p.nprofileamp))); //mu_i n_i
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
    pol.set_chi( chi);                              //set chi of polarisation: nabla_perp (chi nabla_perp )

    if (p.flrmode == 1)
    {
        dg::blas1::transform( y[3], chi, dg::PLUS<>( (p.bgprofamp + p.nprofileamp))); //Ti
        dg::blas1::pointwiseDivide(B2,chi,lambda); //B^2/T_i
        invgamma1.set_chi(lambda);                //(B^2/T - 0.5*tau_i nabla_perp^2)
        
        dg::blas1::pointwiseDivide(y[3],B2,chi); //chi=t_i_tilde/b^2    
        dg::blas2::gemv(lapperpM,chi,omega);
        dg::blas1::axpby(1.0,y[1],-(p.bgprofamp + p.nprofileamp)*0.5*p.tau[1],omega,omega);    
        dg::blas1::axpby(1.0,omega,(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv*p.tau[1],one,omega);        
        invert_invgammadag(invgamma1,chi,omega); //chi= Gamma (Ni-(bgamp+profamp))    
        dg::blas1::pointwiseDot(chi,lambda,chi);   //chi = B^2/T_i chi Gamma (Ni-(bgamp+profamp))   
    }
    if (p.flrmode == 0)
    {
        invgamma1.set_chi(one); ////(B^2/T - 0.5*tau_i nabla_perp^2)
        dg::blas1::axpby(1.0,y[1],0.0,omega,omega);
        invert_invgammadag(invgamma1,chi,omega); //chi= Gamma (Ni-(bgamp+profamp))    
    }  

    dg::blas1::axpby( -1., y[0], 1.,chi,chi);  //chi= Gamma1^dagger (n_i-(bgamp+profamp)) -(n_e-(bgamp+profamp))

    unsigned number = invert_pol( pol, phi[0], chi);   //Gamma1^dagger( N_i) -ne = -nabla ( chi nabla phi)
    if(  number == invert_pol.get_max())
     throw dg::Fail( p.eps_pol);
    return phi[0];
}

template<class G, class Matrix, class container>
container& Feltor<G, Matrix,container>::compute_psi(const container& ti,container& potential)
{
    if (p.flrmode == 1)
    {   
        dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/T
        invgamma1.set_chi(lambda);//invgamma1bar = (B^2/T - 0.5*tau_i nabla_perp^2)
        dg::blas1::pointwiseDot(lambda,potential,lambda); //lambda= B^2/T phi
        invert_invgamma(invgamma1,chi,lambda);    //(B^2/T - 0.5*tau_i nabla_perp^2) chi  =  B^2/T phi
    }
    if (p.flrmode == 0)
    {
        invgamma1.set_chi(one);//invgamma1bar = (1 - 0.5*tau_i nabla_perp^2)
        invert_invgamma(invgamma1,chi,potential);    //(B^2/T - 0.5*tau_i nabla_perp^2) chi  =  B^2/T phi
    }

    poisson.variationRHS(potential, omega); // (nabla_perp phi)^2
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, uE2);// (nabla_perp phi)^2/B^2
    dg::blas1::axpby( 1., chi, -0.5, uE2,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}
template< class G, class Matrix, class container>
container& Feltor<G, Matrix,container>::compute_chii(const container& ti,container& potential)
{    
//  setup rhs
    dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/T
    invgamma2.set_chi(lambda); //(B^2/T - tau_i nabla_perp^2 +  0.25*tau_i^2 nabla_perp^2 T/B^2  nabla_perp^2)
//  set up the lhs
    dg::blas2::gemv(lapperpM,potential,lambda); //lambda = - nabla_perp^2 phi
    dg::blas1::scal(lambda,-0.5*p.tau[1]*p.mu[1]); // lambda = 0.5*tau_i*nabla_perp^2 phi 
    invert_invgamma2(invgamma2,chii,lambda);
    return chii;
}
template<class G, class Matrix, class container>
void Feltor<G, Matrix, container>::initializene( const container& src, const container& ti,container& target)
{   
    if (p.flrmode == 1)
    {
        dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/T    
        invgamma1.set_chi(lambda);
        
        dg::blas1::transform( ti, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //t_i_tilde
        dg::blas1::pointwiseDivide(chi,B2,chi); //chi=t_i_tilde/b^2    
        dg::blas2::gemv(lapperpM,chi,omega); //- lap t_i_tilde/b^2    
        dg::blas1::axpby(1.0,src ,-(p.bgprofamp + p.nprofileamp)*0.5*p.tau[1],omega,omega);  //omega = Ni_tilde +a tau/2 lap t_i_tilde/b^2    
        dg::blas1::axpby(1.0,omega,(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv*p.tau[1],one,omega);   
        invert_invgammadag(invgamma1,target,omega); //=ne-1 = bar(Gamma)_dagger (ni-1)    
        dg::blas1::pointwiseDot(target,lambda,target);
    }
    if (p.flrmode == 0)
    {
        invgamma1.set_chi(one);
        invert_invgammadag(invgamma1,target,src); //=ne-1 = bar(Gamma)_dagger (ni-1)    
    }
}

template<class G, class Matrix, class container>
void Feltor<G, Matrix, container>::initializepi( const container& src, const container& ti,container& target)
{   
    //src =Pi-bg = (N_i-bg)*(T_i-bg) + bg(N_i-bg) + bg(T_i-bg)
    //target =pi-bg =  (n_i-bg)*(t_i-bg) + bg(n_i-bg) + bg(t_i-bg)
    dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/Ti
    if (p.flrmode == 1)
    {
        invgamma2.set_chi(lambda); //(B^2/Ti - tau_i nabla_perp^2 +  0.25*tau_i^2 nabla_perp^2 Ti/B^2  nabla_perp^2)  
        dg::blas1::transform( ti, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //t_i_tilde
        dg::blas1::pointwiseDivide(chi,B2,chi); //chi=t_i_tilde/b^2    
        dg::blas2::gemv(lapperpM,chi,omega);
        dg::blas1::pointwiseDivide(omega,lambda,chi); //-Ti/B^2 lap T_i_tilde/B^2
        dg::blas2::gemv(lapperpM,chi,target);//+ lap (Ti/B^2 lap T_i_tilde/B)
        dg::blas1::axpby(1.0, src ,-(1.-p.tau[1]*0.5*p.mcv*p.mcv*(p.bgprofamp + p.nprofileamp))*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1],omega,omega);  
        dg::blas1::axpby(1.0, omega, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*p.tau[1]*0.25,target, omega); 
        dg::blas1::axpby(1.0, omega, (p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*2.*p.mcv*p.mcv*(1.-0.5*p.tau[1]*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv),one, omega); 
        invert_invgamma2(invgamma2,target,omega); //=(p_i_tilde) = bar(Gamma)_dagger { P_i_tilde + a tau_i lap T_i_tilde/B^2  - a tau^2 /4 lap (Ti/B^2 lap T_i_tilde/B^2)   }
        dg::blas1::pointwiseDot(target,lambda,target); //target = B^2/Ti target
    }    
    if (p.flrmode == 0)
    {     
        invgamma2.set_chi(one); //(B^2/Ti - tau_i nabla_perp^2 +  0.25*tau_i^2 nabla_perp^2 Ti/B^2  nabla_perp^2)   
        dg::blas1::transform( one, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //t_i_tilde
        dg::blas1::pointwiseDivide(chi,B2,chi); //chi=t_i_tilde/b^2    
        dg::blas2::gemv(lapperpM,chi,omega);
        dg::blas1::pointwiseDivide(omega,lambda,chi); //-Ti/B^2 lap T_i_tilde/B^2
        dg::blas2::gemv(lapperpM,chi,target);//+ lap (Ti/B^2 lap T_i_tilde/B)
        dg::blas1::axpby(1.0, src ,-(1.-p.tau[1]*0.5*p.mcv*p.mcv*(p.bgprofamp + p.nprofileamp))*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1],omega,omega);  
        dg::blas1::axpby(1.0, omega, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*p.tau[1]*0.25,target, omega); 
        dg::blas1::axpby(1.0, omega, (p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*2.*p.mcv*p.mcv*(1.-0.5*p.tau[1]*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv),one, omega); 
        invert_invgamma2(invgamma2,target,omega); //=(p_i_tilde) = bar(Gamma)_dagger { P_i_tilde + a tau_i lap T_i_tilde/B^2  - a tau^2 /4 lap (Ti/B^2 lap T_i_tilde/B^2)   }
        dg::blas1::pointwiseDot(target,one,target); //target = B^2/Ti target
    }
}

template<class G, class Matrix, class container>
void Feltor<G, Matrix, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
 /* y[0] := N_e - (p.bgprofamp + p.nprofileamp)
       y[1] := N_i - (p.bgprofamp + p.nprofileamp)
       y[2] := T_e - (p.bgprofamp + p.nprofileamp)
       y[3] := T_i - (p.bgprofamp + p.nprofileamp)
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    //transform compute n and logn and energies
    if (p.iso == 1)    {
        dg::blas1::scal( y[2], 0.0);
        dg::blas1::scal( y[3], 0.0); 
    }
    
    for(unsigned i=0; i<4; i++)
    {
        dg::blas1::transform( y[i], ype[i], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //ype = y +p.bgprofamp + p.nprofileamp
        dg::blas1::transform( ype[i], logype[i], dg::LN<value_type>()); //log(ype)
    }
    //compute phi via polarisation
    phi[0] = polarisation( y);  
    //compute psi
    phi[1] = compute_psi(ype[3], phi[0]); //sets omega for T_perp
    //compute chii
    if (p.iso == 0)    {
    chii   = compute_chii(ype[3], phi[0]);  
    }
    if (p.iso == 1 || p.flrmode==0)    {
        dg::blas1::scal(chii, 0.0);
    }

    
    //Compute energies
    double z[2]    = {-1.0,1.0};
    double S[2]    = {0.0, 0.0};
    double Dperp[4] = {0.0, 0.0,0.0, 0.0};
    //transform compute n and logn and energies
    for(unsigned i=0; i<2; i++)
    {
        if (p.iso == 1) S[i] = z[i]*p.tau[i]*dg::blas2::dot( logype[i], w2d, ype[i]); //N LN N
        if (p.iso == 0) S[i] = z[i]*p.tau[i]*dg::blas2::dot( ype[i+2], w2d, ype[i]); // N T
    }
    mass_ = dg::blas2::dot( one, w2d, ype[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( ype[1], w2d, uE2);   //= 0.5 mu_i N_i u_E^2
    energy_ = S[0] + S[1]  + Tperp; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp;
    for(unsigned i=0; i<2; i++)
    {
        if (p.iso == 1) {
            dg::blas1::axpby(1.,one,1., logype[i] ,chi); //chi = (1+lnN_e)
            dg::blas1::axpby(1.,phi[i],p.tau[i], chi); //chi = (tau_e(1+lnN_e)+phi)
        }
        if (p.iso == 0) dg::blas1::axpby(1., phi[i], p.tau[i], ype[i+2],chi); //chi = (tau_z T + psi)
        dg::blas2::gemv( lapperpM, y[i], lambda);
        dg::blas2::gemv( lapperpM, lambda, omega);//nabla_RZ^4 N_e
        Dperp[i] = -z[i]* p.nu_perp*dg::blas2::dot(chi, w2d, omega);  // ( tau_z T+psi) nabla_RZ^4 N    
    }
    dg::blas2::gemv( lapperpM, y[2], lambda);
    dg::blas2::gemv( lapperpM, lambda, omega);//nabla_RZ^4 t_e-1
    Dperp[2] = -z[0]*p.tau[0]*p.nu_perp*dg::blas2::dot(ype[0], w2d, omega);  // n_e nabla_RZ^4 te-1
    
    //(1+ chii/tau)*
    dg::blas2::gemv( lapperpM, y[3], lambda);
    dg::blas2::gemv( lapperpM, lambda, omega);//nabla_RZ^4 Ti-1
    Dperp[3] = -z[1]*p.tau[1]*p.nu_perp*dg::blas2::dot(ype[1], w2d, omega);  // nu*Z*taui*Ni nabla_RZ^4 T    
    
    dg::blas1::pointwiseDot(chii,ype[1],chi); //chi = Ni*chii
    dg::blas1::pointwiseDivide(chi,ype[3],chi); //chi = Ni*chii/Ti
    dg::blas2::gemv( lapperpM, y[3], lambda);
    dg::blas2::gemv( lapperpM, lambda, omega);//nabla_RZ^4 T
    Dperp[3] += -z[1]*p.nu_perp*dg::blas2::dot(chi, w2d, omega);  // nu*Z(N chii/ T) nabla_RZ^4 T   
    ediff_= Dperp[0]+Dperp[1]+ Dperp[2]+Dperp[3];   
    
    //ExB dynamics
    for(unsigned i=0; i<2; i++)
    {
        poisson( y[i], phi[i], yp[i]);  //[N-1,psi]_xy
        poisson( y[i+2], phi[i], yp[i+2]);  //[T-1,psi]_xy
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);  //dt N = 1/B [N-1,psi]_xy
        dg::blas1::pointwiseDot( yp[i+2], binv, yp[i+2]);  //dt T = 1/B [T-1,psi]_xy
    }
    //add 2nd order FLR terms to ExB dynamics 
    //[2 chi,Ti] terms 
    poisson( y[3], chii, omega);  //omega = [T-1,chi_i]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //omega = 1/B [T-1,chii]_xy
    dg::blas1::axpby(2.,omega,1.0,yp[3]); //dt T_i += 1/B [T-1, 2 chii]_xy
    
    //Moment mixing terms
    //[lnTi,Ni chii] term 
    poisson( logype[3], chii, omega);  //omega = [ln(Ti),chii]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //omega = 1/B [ln(Ti),chii]_xy
    dg::blas1::pointwiseDot(omega, ype[1], omega); //omega = Ni/B [ln(Ti),chii]_xy
    dg::blas1::axpby(1.,omega,1.0,yp[1]); //dt N_i += Ni/B [ln(Ti),chii]_xy
    poisson( logype[3],  y[1], omega);  //omega = [ln(Ti),Ni]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //omega = 1/B [ln(Ti),Ni]_xy
    dg::blas1::pointwiseDot(omega, chii, omega); //omega = chii/B [ln(Ti),Ni]_xy
    dg::blas1::axpby(1.,omega,1.0,yp[1]); //dt N_i += chii/B [ln(Ti),Ni]_xy
    //Ti chii [ln(chii)- ln(Ti),ln(Ni)] term
    poisson( logype[1],chii, omega);  //omega = [ln(Ni),chii]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //omega = 1/B [ln(Ni),chii]_xy
    dg::blas1::pointwiseDot(omega, ype[3], omega); //omega = Ti/B [ln(Ni),chii]_xy
    dg::blas1::axpby(1.,omega,1.0,yp[3]);   //dt T_i += Ti/B [ln(Ni),chii]_xy
    poisson( logype[1],y[3], omega);  //omega = [ln(Ni),Ti-1]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //omega = 1/B [ln(Ni),Ti-1]_xy
    dg::blas1::pointwiseDot(omega, chii, omega); //omega = chii/B [ln(Ni),Ti-1]_xy
    dg::blas1::axpby(-1.,omega,1.0,yp[3]);   //dt T_i += - chii/B [ln(Ni),Ti-1]_xy

    //curvature dynamics
    for(unsigned i=0; i<2; i++)
    {
        //N*K(psi) and T*K(psi)  terms
        dg::blas2::gemv( poisson.dyrhs(), phi[i], lambda); //lambda = dy psi
        dg::blas1::pointwiseDot(lambda,ype[i],omega); //omega =  n dy psi
        dg::blas1::axpby(p.mcv,omega,1.0,yp[i]);   // dtN +=  mcv* N dy psi
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); // T dy psi
        dg::blas1::axpby(p.mcv,omega,1.0,yp[i+2]);   // dtT +=  mcv* T dy psi        
        // K(T N) terms
        dg::blas2::gemv( poisson.dyrhs(), y[i], lambda); //lambda = dy (N-1)
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); //omega =  T dy (N-1)
        dg::blas1::axpby(p.tau[i]*p.mcv,omega,1.0,yp[i]); //dt N += tau*mcv*T dy (N-1)
        dg::blas2::gemv( poisson.dyrhs(), y[i+2], lambda); //lambda = dy (T-1)
        dg::blas1::pointwiseDot(lambda,ype[i],omega); // omega = N dy (T-1)
        dg::blas1::axpby(p.tau[i]*p.mcv,omega,1.0,yp[i]);  //dt N += tau*mcv*N dy (T-1)        
        //3 T*K(T) terms
        dg::blas2::gemv( poisson.dyrhs(), y[i+2], lambda); //lambda = dy (T-1)
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); // omega = T dy (T-1)
        dg::blas1::axpby(3.*p.tau[i]*p.mcv,omega,1.0,yp[i+2]); //dt T +=  3 tau*mcv* T dy (T-1)
        //T^2*K(ln(N)) terms
        dg::blas2::gemv( poisson.dyrhs(), logype[i], lambda); //lambda = dy (ln(N))
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); //omega = T dy (ln(N))
        dg::blas1::pointwiseDot(omega,ype[i+2],omega); //omega =  T^2 dy (ln(N))
        dg::blas1::axpby(p.tau[i]*p.mcv,omega,1.0,yp[i+2]); //dt T += tau mcv T^2 dy (ln(N))         
    }   
    //add FLR correction to curvature dynamics
    //Ni K(chii) and Ti K(3 chii) term
    dg::blas2::gemv( poisson.dyrhs(), chii, lambda); //lambda = dy chii
    dg::blas1::pointwiseDot(lambda,ype[1],omega); //omega = Ni dy chii
    dg::blas1::axpby(p.mcv,omega,1.0,yp[1]);   // dtNi +=  mcv* Ni dy chii
    dg::blas1::pointwiseDot(lambda,ype[3],omega); // omega = Ti dy chii
    dg::blas1::axpby(3.*p.mcv,omega,1.0,yp[3]);   // dtTi += 3.* mcv* Ti dy chii    
    //Ni chii K(lnTi - lnNi) term
    dg::blas1::axpby(1.,logype[1],-1.0,logype[3],omega); //omega = -ln(Ti)+ln(Ni)
    dg::blas2::gemv( poisson.dyrhs(), omega, lambda); //lambda = dy(-ln(Ti)+ln(Ni))
    dg::blas1::pointwiseDot(lambda,ype[1],omega); // omega = Ni dy(-ln(Ti)+ln(Ni))
    dg::blas1::pointwiseDot(omega,chii,omega); //omega =  Ni  chii dy(-ln(Ti)+ln(Ni))
    dg::blas1::axpby(p.mcv,omega,1.0,yp[1]);   // dtNi +=  mcv* Ni  chii dy(-ln(Ti)+ln(Ni))    
    //chii K(Ti) term
    dg::blas2::gemv( poisson.dyrhs(), y[3], lambda); //lambda = dy (Ti-1)
    dg::blas1::pointwiseDot(lambda,chii,omega); //omega =  chii dy (Ti-1)
    dg::blas1::axpby(-p.mcv,omega,1.0,yp[3]);   // dtTi +=-  mcv*  chii dy (Ti-1)
    //Ti chii K(lnNi)) term
    dg::blas2::gemv( poisson.dyrhs(), logype[1], lambda); //lambda = dy (ln(Ni))
    dg::blas1::pointwiseDot(lambda,chii,omega); // omega = chii dy (ln(Ni))
    dg::blas1::pointwiseDot(omega,ype[3],omega); // omega =Ti chii dy (ln(Ni))
    dg::blas1::axpby(p.mcv,omega,1.0,yp[3]);   // dtTi +=  mcv*  chii dy (ln(Ni))    

     if (p.iso == 1) {
        dg::blas1::scal( yp[2], 0.0);
        dg::blas1::scal( yp[3], 0.0); 
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


