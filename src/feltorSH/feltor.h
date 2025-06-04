#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
/*!@file

  Contains the solvers 
  */

namespace eule
{
///@addtogroup solver
///@{
/**
 * @brief Diffusive terms for Explicit solver
 *
 * @tparam Matrix The Matrix class
 * @tparam container The Vector class 
 */

template<class Geometry, class Matrix, class container>
struct Implicit
{
    Implicit( const Geometry& g,eule::Parameters p):
        p(p),
        temp( dg::evaluate(dg::zero, g)),
        LaplacianM_perp ( g,g.bcx(),g.bcy(),  dg::centered)
    {
    }
    void operator()(double, const std::vector<container>& x, std::vector<container>& y)
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
struct Explicit
{
    using value_type = dg::get_value_type<container>;

    Explicit( const Geometry& g, eule::Parameters p);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
    void initializene( const container& y, const container& helper, container& target);
    void initializepi( const container& y, const container& helper, container& target);

    void operator()( double t, const std::vector<container>& y, std::vector<container>& yp);

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

    container chi, omega, lambda,iota; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!
    const container binv;
    const container one;
    container B2;
    const container w2d;
    std::vector<container> phi; // =(phi,psi_i), (0,chi_i)
    container chii,uE2; //dont use them as helper
    std::vector<container> ype, logype; 

    //matrices and solvers
    dg::Poisson< Geometry, Matrix, container> poisson; 

    dg::Elliptic<   Geometry, Matrix, container> lapperpM; 
    std::vector<container> multi_chi;
    std::vector<dg::Elliptic<   Geometry, Matrix, container> > multi_pol;     
    std::vector<dg::Helmholtz<  Geometry, Matrix, container> > multi_invgamma1;    
    std::vector<dg::Helmholtz2< Geometry, Matrix, container> > multi_invgamma2;
    
    dg::MultigridCG2d<Geometry, Matrix, container> multigrid;
    dg::Extrapolation<container> old_phi, old_psi, old_gammaN, old_chiia, old_chiib;
    
    const eule::Parameters p;

    double mass_, energy_, diff_, ediff_;
    std::vector<double> evec;
};     

template<class Grid, class Matrix, class container>
Explicit<Grid, Matrix, container>::Explicit( const Grid& g, eule::Parameters p): 
    chi( dg::evaluate( dg::zero, g)), omega(chi),  lambda(chi), iota(chi), 
    binv( dg::evaluate( dg::LinearX( p.mcv, 1.), g) ),
    one( dg::evaluate( dg::one, g)),    
    B2( dg::evaluate( dg::one, g)),    
    w2d( dg::create::weights(g)),
    phi( 2, chi),chii(chi),uE2(chi),// (phi,psi), (chi_i), u_ExB
    ype(4,chi), logype(ype), // y+(bgamp+profamp) , log(ype)
    poisson(g, g.bcx(), g.bcy(), g.bcx(), g.bcy()), //first N/U then phi BCC
    lapperpM ( g,g.bcx(), g.bcy(),              dg::centered),
    multigrid( g, 3),
    old_phi( 2, chi), old_psi( 2, chi), old_gammaN( 2, chi), old_chiia( 2, chi),old_chiib( 2, chi),
    p(p),
    evec(3)
{
    multi_chi= multigrid.project( chi);
    multi_pol.resize(3);
    multi_invgamma2.resize(3);
    for( unsigned u=0; u<3; u++)
    {
        multi_pol[u].construct(       multigrid.grid(u), g.bcx(), g.bcy(),  dg::centered, 1.0);
        multi_invgamma1.push_back( {-0.5*p.tau[1]*p.mu[1], {multigrid.grid(u), g.bcx(), g.bcy(), dg::centered}});
        multi_invgamma2[u].construct( multigrid.grid(u), g.bcx(), g.bcy(), -0.5*p.tau[1]*p.mu[1], dg::centered);
    }
    dg::blas1::pointwiseDivide(one,binv,B2);
    dg::blas1::pointwiseDivide(B2,binv,B2);
}

template<class G, class Matrix, class container>
container& Explicit<G, Matrix, container>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-(bgamp+profamp)) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]*(p.bgprofamp + p.nprofileamp))); //mu_i n_i
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
    multigrid.project( chi, multi_chi);
    for( unsigned u=0; u<3; u++)
    {
        multi_pol[u].set_chi( multi_chi[u]); //set chi of polarisation: nabla_perp (chi nabla_perp )
    }   
    if (p.tau[1] == 0.) {
        dg::blas1::axpby( 1., y[1], 0.,chi); //chi = N_i - 1
    } 
    else {
        if (p.flrmode == 1)
        {
            dg::blas1::transform( y[3], chi, dg::PLUS<>( (p.bgprofamp + p.nprofileamp))); //Ti
            dg::blas1::pointwiseDivide(B2,chi,lambda); //B^2/T_i
            multigrid.project( lambda, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma1[u].set_chi( multi_chi[u]); //(B^2/T - 0.5*tau_i nabla_perp^2)
            }        
            dg::blas1::pointwiseDivide(y[3],B2,chi); //chi=t_i_tilde/b^2    
            dg::blas2::gemv(lapperpM,chi,omega);
            dg::blas1::axpby(1.0,y[1],-(p.bgprofamp + p.nprofileamp)*0.5*p.tau[1],omega,omega);    
            dg::blas1::axpby(1.0,omega,(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv*p.tau[1],one,omega);        
            old_gammaN.extrapolate( chi);
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, chi,omega, p.eps_gamma);
            old_gammaN.update( chi);
            dg::blas1::pointwiseDot(chi,lambda,chi);   //chi = B^2/T_i chi Gamma (Ni-(bgamp+profamp))   
        }
        if (p.flrmode == 0)
        {
            multigrid.project( one, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma1[u].set_chi( multi_chi[u]); //(1 - 0.5*tau_i nabla_perp^2)
            }
            dg::blas1::axpby(1.0,y[1],0.0,omega,omega);
            old_gammaN.extrapolate( chi);
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, chi,omega, p.eps_gamma);
            old_gammaN.update( chi);
        }  
    }

    dg::blas1::axpby( -1., y[0], 1.,chi,chi);  //chi= Gamma1^dagger (n_i-(bgamp+profamp)) -(n_e-(bgamp+profamp))
     //invert pol
    old_phi.extrapolate( phi[0]);
    std::vector<unsigned> number = multigrid.solve( multi_pol, phi[0], chi, p.eps_pol);
    old_phi.update( phi[0]);
    if(  number[0] == multigrid.max_iter())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template<class G, class Matrix, class container>
container& Explicit<G, Matrix,container>::compute_psi(const container& ti,container& potential)
{
    if (p.tau[1] == 0.) {
        dg::blas1::axpby( 1., potential, 0., phi[1]); 
    } 
    else {
        if (p.flrmode == 1)
        {   
            dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/T
            multigrid.project( lambda, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma1[u].set_chi( multi_chi[u]);
            }
            dg::blas1::pointwiseDot(lambda,potential,lambda); //lambda= B^2/T phi
            old_psi.extrapolate( phi[1]);
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, phi[1], lambda, p.eps_gamma);
            old_psi.update( phi[1]);    
        }
        if (p.flrmode == 0)
        {
            multigrid.project( one, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma1[u].set_chi( multi_chi[u]);
            }
            old_psi.extrapolate( phi[1]);
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, phi[1], potential, p.eps_gamma);
            old_psi.update( phi[1]);
        }
    }

    multi_pol[0].variation(binv, potential, uE2); // (nabla_perp phi)^2
    dg::blas1::axpby( 1., phi[1], -0.5, uE2,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}
template< class G, class Matrix, class container>
container& Explicit<G, Matrix,container>::compute_chii(const container& ti,container& potential)
{    
    if (p.tau[1] == 0.) {
        dg::blas1::scal(chii,0.0); 
    } 
    else {
        if (p.flrmode==1)
        {
                //  setup rhs
            dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/T
            multigrid.project( lambda, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma1[u].set_chi( multi_chi[u]);
            }       //  set up the lhs
            dg::blas2::gemv(lapperpM,potential,lambda); //lambda = - nabla_perp^2 phi
            dg::blas1::scal(lambda,-0.5*p.tau[1]*p.mu[1]); // lambda = 0.5*tau_i*nabla_perp^2 phi 
            old_chiia.extrapolate( chii);
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, chii, lambda, p.eps_gamma);
            old_chiia.update( chii);        dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/T
            dg::blas1::pointwiseDot(chii,lambda,lambda);
            old_chiib.extrapolate( chii);
            number = multigrid.solve( multi_invgamma1, chii, lambda, p.eps_gamma);
            old_chiib.update( chii);
        }
    }
    return chii;
}
template<class G, class Matrix, class container>
void Explicit<G, Matrix, container>::initializene( const container& src, const container& ti,container& target)
{   
    if (p.flrmode == 1)
    {
        dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/T    
        multigrid.project( lambda, multi_chi);
        for( unsigned u=0; u<3; u++)
        {
            multi_invgamma1[u].set_chi( multi_chi[u]);
        }        
        dg::blas1::transform( ti, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //t_i_tilde
        dg::blas1::pointwiseDivide(chi,B2,chi); //chi=t_i_tilde/b^2    
        dg::blas2::gemv(lapperpM,chi,omega); //- lap t_i_tilde/b^2    
        dg::blas1::axpby(1.0,src ,-(p.bgprofamp + p.nprofileamp)*0.5*p.tau[1],omega,omega);  //omega = Ni_tilde +a tau/2 lap t_i_tilde/b^2    
        dg::blas1::axpby(1.0,omega,(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv*p.tau[1],one,omega);   
        std::vector<unsigned> number = multigrid.solve( multi_invgamma1, target,omega, p.eps_gamma);  //=ne-1 = Gamma (ni-1)  
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( p.eps_gamma);
        dg::blas1::pointwiseDot(target,lambda,target);
    }
    if (p.flrmode == 0)
    {
        multigrid.project( one, multi_chi);
        for( unsigned u=0; u<3; u++)
        {
            multi_invgamma1[u].set_chi( multi_chi[u]);
        }
        std::vector<unsigned> number = multigrid.solve( multi_invgamma1, target,src, p.eps_gamma);  //=ne-1 = Gamma (ni-1)  
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( p.eps_gamma);
    }
}

template<class G, class Matrix, class container>
void Explicit<G, Matrix, container>::initializepi( const container& src, const container& ti,container& target)
{   
    //src =Pi-bg = (N_i-bg)*(T_i-bg) + bg(N_i-bg) + bg(T_i-bg)
    //target =pi-bg =  (n_i-bg)*(t_i-bg) + bg(n_i-bg) + bg(t_i-bg)
//src =Pi-bg^2 = (N_i-bg)*(T_i-bg) + bg(N_i-bg) + bg(T_i-bg)
    //target =pi-bg^2 =  (n_i-bg)*(t_i-bg) + bg(n_i-bg) + bg(t_i-bg)
    if (p.init==0)        
    {
        if (p.flrmode == 1)
        {
            dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/Ti
            multigrid.project( lambda, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma2[u].set_chi( multi_chi[u]);//(B^2/Ti - tau_i nabla_perp^2 +  0.25*tau_i^2 nabla_perp^2 Ti/B^2  nabla_perp^2)  
            }
            dg::blas1::transform( ti, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //t_i_tilde
            
            
            //RHS + "correction terms":
            dg::blas1::pointwiseDivide(chi,B2,chi); //chi=t_i_tilde/b^2    
            dg::blas2::gemv(lapperpM,chi,omega); //-lap T_i_tilde/B^2
            // chi= Pi_tilde + 2*a^3 tau*mcv^2        
            dg::blas1::axpby(1.0, src,   (p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*2.*p.mcv*p.mcv,one, chi); 
            // chi += a^2 tau lap ( T_i_tilde/B^2)
            dg::blas1::axpby(1.0, chi, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1],omega, chi); 
            dg::blas2::gemv(lapperpM,omega,target);//+ lap (lap T_i_tilde/B)
            // chi+= - a^2 tau^2*mcv^2 *0.5* lap^2 (T_i_tilde/B)
            dg::blas1::axpby(1.0, chi, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*p.tau[1]*p.mcv*p.mcv*0.5,target, chi);            
            dg::blas1::pointwiseDivide(omega,lambda,omega); //-Ti/B^2 lap T_i_tilde/B^2
            dg::blas2::gemv(lapperpM,omega,target);//+ lap (Ti/B^2 lap T_i_tilde/B)
            // chi+= - a^2 tau^2/4 lap (Ti/B^2 lap T_i_tilde/B)
            dg::blas1::axpby(1.0, chi, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*p.tau[1]*0.25,target, chi);    

            std::vector<unsigned> number = multigrid.solve( multi_invgamma2, target,chi, p.eps_gamma);  //=(p_i_tilde) = bar(Gamma)_dagger { P_i_tilde + a^2 tau_i lap T_i_tilde/B^2  - a^2 tau^2 /4 lap (Ti/B^2 lap T_i_tilde/B^2)   }
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_gamma);
            dg::blas1::pointwiseDot(target,lambda,target); //target = B^2/Ti target
        }
        if (p.flrmode == 0)
        {     
            multigrid.project( one, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma1[u].set_chi( multi_chi[u]);
            }
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, target,src, p.eps_gamma);//=(p_i_tilde) = bar(Gamma)_dagger { P_i_tilde + a^2 tau_i lap T_i_tilde/B^2  - a^2 tau^2 /4 lap (Ti/B^2 lap T_i_tilde/B^2)}
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_gamma);
        }
    }    

    
    if (p.init==1)
    {
        if (p.flrmode==1)
        {
            //solve polarisation for phi with Ti=Ni=ne
            dg::blas1::pointwiseDot( ti, binv, chi);        //chi = (T_i ) /B
            dg::blas1::pointwiseDot( chi, binv, chi);       //chi = (T_i ) /B^2
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_pol[u].set_chi( multi_chi[u]); //set chi of polarisation: nabla_perp (chi nabla_perp )
            }
                dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/Ti
            multigrid.project( lambda, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
                multi_invgamma1[u].set_chi( multi_chi[u]); //(B^2/T - 0.5*tau_i nabla_perp^2)
            }         
            dg::blas1::transform( ti, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //chi = T_i_tilde
            dg::blas1::pointwiseDivide(chi,B2,chi); //chi=T_i_tilde/B^2    
            dg::blas2::gemv(lapperpM,chi,omega);    //omega = -lap T_i_tilde/B^2    
            dg::blas1::transform( ti, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //chi = T_i_tilde
            dg::blas1::axpby(1.0,chi,-(p.bgprofamp + p.nprofileamp)*0.5*p.tau[1],omega,omega); //omega = T_i_tilde + a^2 tau*0.5* lap T_i_tilde/B^2   
            dg::blas1::axpby(1.0,omega,(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.mcv*p.mcv*p.tau[1],one,omega);  //omega = T_i_tilde + a^2 tau*0.5* lap T_i_tilde/B^2 + a^2 mcv^2 tau      
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, chi,omega, p.eps_gamma);// chi = Gamma^-1 omega
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_gamma);// chi = Gamma^-1 omega
            dg::blas1::pointwiseDot(chi,lambda,chi);   //chi = B^2/T_i chi Gamma^-1 omega 

            dg::blas1::transform( ti, lambda, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //T_i_tilde
            dg::blas1::axpby( -1., lambda, 1.,chi,chi);  //chi= Gamma1^dagger (Ti-(bgamp+profamp)) -(Ti-(bgamp+profamp))

            
            number = multigrid.solve( multi_pol, omega, chi, p.eps_pol); //(Gamma1^dagger -1) T_i = -nabla ( chi nabla phi)
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_pol);
            
            //set up polarisation term of pressure equation
            dg::blas1::pointwiseDot( ti, ti, chi); //Pi=ti^2
            dg::blas1::pointwiseDot( chi, binv, chi);
            dg::blas1::pointwiseDot( chi, binv, chi);       //t_i^2 /B^2
            lapperpM.set_chi(chi);
            dg::blas2::symv(lapperpM,omega,iota); //- nabla( P/B^2 nabla phi) with omega=phi
            lapperpM.set_chi(one);
            
            
            
            //solve gamma terms of pressure equation
            dg::blas1::pointwiseDivide(B2,ti,lambda); //B^2/Ti
            multigrid.project( lambda, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
            multi_invgamma2[u].set_chi( multi_chi[u]); //(B^2/Ti_0 - tau_i nabla_perp^2 +  0.25*tau_i^2 nabla_perp^2 Ti_0/B^2  nabla_perp^2)  
            }
            dg::blas1::transform( ti, chi, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //t_i_tilde            
            
            //RHS + "correction terms":
            dg::blas1::pointwiseDivide(chi,B2,chi); //chi=t_i_tilde/b^2    
            dg::blas2::gemv(lapperpM,chi,omega); //-lap T_i_tilde/B^2
            // chi= Pi_tilde + 2*a^3 tau*mcv^2        
            dg::blas1::axpby(1.0, src,   (p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*2.*p.mcv*p.mcv,one, chi); 
            // chi += a^2 tau lap ( T_i_tilde/B^2)
            dg::blas1::axpby(1.0, chi, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1],omega, chi); 
            dg::blas2::gemv(lapperpM,omega,target);//+ lap (lap T_i_tilde/B)
            // chi+= - a^2 tau^2*mcv^2 *0.5* lap^2 (T_i_tilde/B)
            dg::blas1::axpby(1.0, chi, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*p.tau[1]*p.mcv*p.mcv*0.5,target, chi);            
            dg::blas1::pointwiseDivide(omega,lambda,omega); //-Ti/B^2 lap T_i_tilde/B^2
            dg::blas2::gemv(lapperpM,omega,target);//+ lap (Ti/B^2 lap T_i_tilde/B)
            // chi+= - a^2 tau^2/4 lap (Ti/B^2 lap T_i_tilde/B)
            dg::blas1::axpby(1.0, chi, -(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)*p.tau[1]*p.tau[1]*0.25,target, chi);     

            number = multigrid.solve( multi_invgamma2, target, chi, p.eps_gamma); //=(p_i_tilde) = bar(Gamma)_dagger { P_i_tilde + a^2 tau_i lap T_i_tilde/B^2  - a^2 tau^2 /4 lap (Ti/B^2 lap T_i_tilde/B^2)   }
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_gamma);
            dg::blas1::pointwiseDot(target,lambda,target); //target = B^2/Ti target
            
            dg::blas1::axpby(-2.0,iota,1.0,target,target); //target+=  +2 nabla( P/B^2 nabla phi)
        }
        if (p.flrmode==0)
        {
            //solve polarisation for phi with Ti=Ni=ne
            dg::blas1::pointwiseDot( ti, binv, chi);        //chi = (T_i ) /B = n_e/B^2
            dg::blas1::pointwiseDot( chi, binv, chi);       //chi = (T_i ) /B^2 = n_e/B^2
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
            multi_pol[u].set_chi( multi_chi[u]); //set chi of polarisation: nabla_perp (chi nabla_perp )
            }
            
            multigrid.project( one, multi_chi);
            for( unsigned u=0; u<3; u++)
            {
            multi_invgamma1[u].set_chi( multi_chi[u]);               //(1 - 0.5*tau_i nabla_perp^2)
            }  
            dg::blas1::transform( ti, omega, dg::PLUS<>( -1.0*(p.bgprofamp + p.nprofileamp))); //chi = T_i_tilde = ne_tilde
            
            std::vector<unsigned> number = multigrid.solve( multi_invgamma1, chi,omega, p.eps_gamma);// chi = Gamma^-1 omega
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_gamma);

            dg::blas1::axpby( -1., omega, 1.,chi,chi);  //chi= Gamma1^dagger (Ti-(bgamp+profamp)) -(Ti-(bgamp+profamp))

            number = multigrid.solve( multi_pol, omega, chi, p.eps_pol); //(Gamma1^dagger -1) T_i = -nabla ( chi nabla phi)
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_pol);
            
            dg::blas1::pointwiseDot( ti, ti, chi); //Pi=ti^2
            dg::blas1::pointwiseDot( chi, binv, chi);
            dg::blas1::pointwiseDot( chi, binv, chi);       //t_i^2 /B^2
            lapperpM.set_chi(chi);
            dg::blas2::symv(lapperpM,omega,iota); //- nabla( P/B^2 nabla phi)
            lapperpM.set_chi(one);
            //omega=phi
            
            //solve gamma_1 Pi
            number = multigrid.solve( multi_invgamma1, target,src, p.eps_gamma); //=(p_i_tilde) = bar(Gamma)_dagger { P_i_tilde + a^2 tau_i lap T_i_tilde/B^2  - a^2 tau^2 /4 lap (Ti/B^2 lap T_i_tilde/B^2)}
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( p.eps_gamma);
            
            dg::blas1::axpby(-2.0,iota,1.0,target,target); //target+=  +2 nabla( P/B^2 nabla phi)
        }
    }
}

template<class G, class Matrix, class container>
void Explicit<G, Matrix, container>::operator()( double, const std::vector<container>& y, std::vector<container>& yp)
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


