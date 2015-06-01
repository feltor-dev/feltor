#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "parameters.h"
// #include "geometry_circ.h"
#include "dg/backend/average.cuh"

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
    template<class Grid2d>
    Rolkar( const Grid2d& g, eule::Parameters p):
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
    dg::Elliptic<Matrix, container, Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const Preconditioner& weights(){return LaplacianM_perp.weights();}
    const Preconditioner& precond(){return LaplacianM_perp.precond();}
  private:
    const eule::Parameters p;
    container temp;    
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

    template<class Grid2d>
    Feltor( const Grid2d& g, eule::Parameters p);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
    void initializene( const container& y, const container& helper, container& target);

    void operator()( std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    std::vector<double> energy_vector( ) {return evec;}
    double energy_diffusion( ){ return ediff_;}
    void energies( std::vector<container>& y);

  private:
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi(const container& helper, container& potential);
    container& compute_chii(const container& helper, container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation

    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!
    container chii; //dont use them as helper
    const container binv;
    const container one;
    const Preconditioner w2d, v2d;
    std::vector<container> phi; // =(phi,psi_i), (0,chi_i)
    std::vector<container> ype, logype; 

    //matrices and solvers
    dg::Poisson< Matrix, container> poisson; 

    dg::Elliptic< Matrix, container, Preconditioner > pol,lapperp; 
    dg::Helmholtz< Matrix, container, Preconditioner > invgammaDIR,invgammaNU;

    dg::Invert<container> invert_pol,invert_invgamma;
    const eule::Parameters p;

    double mass_, energy_, diff_, ediff_;
    std::vector<double> evec;
    //probe
};     

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, eule::Parameters p): 
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi), 
    binv( dg::evaluate( dg::LinearX( p.mcv, 1.), g) ),
    one( dg::evaluate( dg::one, g)),    
    w2d( dg::create::weights(g)), v2d( dg::create::inv_weights(g)), 
    phi( 2, chi),chii(dg::evaluate(dg::CONSTANT(0.0),g)),// (phi,psi), (chi_i)
    ype(4,chi), logype(ype), // y+(bgamp+profamp) , log(ype)
    poisson(g, g.bcx(), g.bcy(), g.bcx(), g.bcy()), //first N/U then phi BCC
    pol(    g, g.bcx(), g.bcy(), dg::not_normed,          dg::centered), 
    lapperp ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
    invgammaDIR( g,g.bcx(), g.bcy(),-0.5*p.tau[1]*p.mu[1],dg::centered),
    invgammaNU(  g,g.bcx(), g.bcy(),-0.5*p.tau[1]*p.mu[1],dg::centered),
    invert_pol(      omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    p(p),
    evec(3)
{
}

// template<class Matrix, class container, class P>
// container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
// {
//     dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-(bgamp+profamp)) 
//     dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]*(p.bgprofamp + p.nprofileamp))); //mu_i n_i
//     dg::blas1::pointwiseDot( chi, binv, chi);
//     dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
//     pol.set_chi( chi);
//     invert_invgamma(invgammaNU,chi,y[1]); //chi= Gamma (Ni-(bgamp+profamp))    
//     dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-(bgamp+profamp)) -(n_e-(bgamp+profamp))
//     //= Gamma n_i - n_e
//     unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi
//         if(  number == invert_pol.get_max())
//             throw dg::Fail( p.eps_pol);
//     return phi[0];
// }
template<class Matrix, class container, class P>
container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-(bgamp+profamp)) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]*(p.bgprofamp + p.nprofileamp))); //mu_i n_i
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
    pol.set_chi( chi);
    
    dg::blas1::pointwiseDivide(one,binv,lambda);
    dg::blas1::pointwiseDot(lambda,lambda,lambda);//B^2
    dg::blas1::transform( y[3], chi, dg::PLUS<>( (p.bgprofamp + p.nprofileamp))); //mu_i n_i
    dg::blas1::pointwiseDivide(lambda,chi,lambda); //B^2/T
    invgammaNU.set_chi(lambda);
    invert_invgamma(invgammaNU,chi,y[1]); //chi= Gamma (Ni-(bgamp+profamp))    
    dg::blas1::pointwiseDot(chi,lambda,chi);
    invgammaNU.set_chi(one);
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-(bgamp+profamp)) -(n_e-(bgamp+profamp))
    //= Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi
        if(  number == invert_pol.get_max())
            throw dg::Fail( p.eps_pol);
    return phi[0];
}
// template< class Matrix, class container, class P>
// container& Feltor<Matrix,container, P>::compute_psi( container& potential)
// {
//     invert_invgamma(invgammaDIR,chi,potential);                 //chi  = Gamma phi
//     poisson.variationRHS(potential, omega);
//     dg::blas1::pointwiseDot( binv, omega, omega);
//     dg::blas1::pointwiseDot( binv, omega, omega);
//     dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
//     return phi[1];    
// }
template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_psi(const container& ti,container& potential)
{
    dg::blas1::pointwiseDivide(one,binv,lambda);
    dg::blas1::pointwiseDot(lambda,lambda,lambda);//B^2
    dg::blas1::pointwiseDivide(lambda,ti,lambda); //B^2/T
    invgammaDIR.set_chi(lambda);
    dg::blas1::pointwiseDot(lambda,potential,lambda);
    invert_invgamma(invgammaDIR,chi,lambda);                 //chi  = Gamma phi
    invgammaDIR.set_chi(one);
    poisson.variationRHS(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}
template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_chii(const container& ti,container& potential)
{
    dg::blas1::pointwiseDivide(one,binv,lambda);
    dg::blas1::pointwiseDot(lambda,lambda,lambda);//B^2
    dg::blas1::pointwiseDivide(lambda,ti,lambda); //B^2/T
    invgammaDIR.set_chi(lambda);
    dg::blas1::pointwiseDot(lambda,potential,lambda);
    invert_invgamma(invgammaDIR,chi,lambda);                 //chi  = Gamma phi
    invgammaDIR.set_chi(one);
    poisson.variationRHS(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::axpby( 1., chi, -0.5, omega,chii);             //psi  Gamma phi - 0.5 u_E^2
    return chii;    
}
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::initializene( const container& src, const container& ti,container& target)
{   
    dg::blas1::pointwiseDivide(one,binv,lambda);
    dg::blas1::pointwiseDot(lambda,lambda,lambda);//B^2
    dg::blas1::pointwiseDivide(lambda,ti,lambda); //B^2/T
    
    invgammaNU.set_chi(lambda);
    invert_invgamma(invgammaNU,target,src); //=ne-1 = bar(Gamma)_dagger (ni-1)    
    dg::blas1::pointwiseDot(target,lambda,target);
    invgammaNU.set_chi(one);

}
// template<class Matrix, class container, class P>
// void Feltor<Matrix, container, P>::initializene( const container& src, container& target)
// {   
//     invert_invgamma(invgammaNU,target,src); //=ne-1 = Gamma_dagger (ni-1)    
// }
template<class M, class V, class P>
void Feltor<M, V, P>::energies( std::vector<V>& y)
{
    phi[0] = polarisation( y);
    double z[2]    = {-1.0,1.0};
    double S[2]    = {0.0, 0.0};
    double Dperp[2] = {0.0, 0.0};
    //transform compute n and logn and energies
    for(unsigned i=0; i<4; i++)
    {
        dg::blas1::transform( y[i], ype[i], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //ype = y +p.bgprofamp + p.nprofileamp
        dg::blas1::transform( ype[i], logype[i], dg::LN<value_type>()); //log(ype)
    }
    phi[1] = compute_psi(ype[3], phi[0]); //sets omega

    for(unsigned i=0; i<2; i++)
    {
        S[i]    = z[i]*p.tau[i]*dg::blas2::dot( ype[i+2], w2d, ype[i]); // N T
    }
    mass_ = dg::blas2::dot( one, w2d, y[0] ); //take real ion density which is electron density!!
    double Tperp = 0.5*p.mu[1]*dg::blas2::dot( ype[1], w2d, omega);   //= 0.5 mu_i N_i u_E^2
    energy_ = S[0] + S[1]  + Tperp; 
    evec[0] = S[0], evec[1] = S[1], evec[2] = Tperp;
   
}


template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
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
    //compute phi via polarisation
    double z[2]    = {-1.0,1.0};
    phi[0] = polarisation( y);
    //compute chii

    //transform compute n and logn and energies
    for(unsigned i=0; i<4; i++)
    {
        dg::blas1::transform( y[i], ype[i], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //ype = y +p.bgprofamp + p.nprofileamp
        dg::blas1::transform( ype[i], logype[i], dg::LN<value_type>()); //log(ype)
    }
    phi[1] = compute_psi(ype[3], phi[0]); //sets omega

    //ExB dynamics
    for(unsigned i=0; i<2; i++)
    {
        poisson( y[i], phi[i], yp[i]);  //[N-1,psi]_xy
        poisson( y[i+2], phi[i], yp[i+2]);  //[T-1,psi]_xy
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);  //dt N = 1/B [N-1,psi]_xy
        dg::blas1::pointwiseDot( yp[i+2], binv, yp[i+2]);  //dt T = 1/B [T-1,psi]_xy
    }
    //add 2nd order FLR terms to ExB dynamics 
    //[chi,Ni] and [chi,Ti] terms
    poisson( y[1], chii, omega);  //[Ni-1,chi_i]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //1/B[Ni-1,chii]_xy
    dg::blas1::axpby(1.,omega,1.0,yp[1]); 
    poisson( y[3], chii, omega);  //[T-1,chi_i]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //1/B [T-1,chii]_xy
    dg::blas1::axpby(2.,omega,1.0,yp[3]); 
    //[lnTi,Ni chii] term
    dg::blas1::pointwiseDot(ype[1],chii,lambda);
    poisson( logype[3], lambda, omega);  //[ln(Ti),Ni*chii]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //1/B [ln(Ti),Ni*chii]_xy
    dg::blas1::axpby(1.,omega,1.0,yp[1]); 
    //Ti chii [ln(chii)- ln(Ti),ln(Ni)] term
    poisson( logype[1],chii, omega);  //[ln(Ni),chii]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //1/B [ln(Ni),chii]_xy
    dg::blas1::pointwiseDot(omega, ype[3], omega); //Ti/B [ln(Ni),chii]_xy
    dg::blas1::axpby(1.,omega,1.0,yp[3]);   
    poisson( logype[1],y[3], omega);  //[ln(Ni),Ti-1]_xy
    dg::blas1::pointwiseDot(omega, binv, omega); //1/B [ln(Ni),Ti-1]_xy
    dg::blas1::pointwiseDot(omega, chii, omega); //chii/B [ln(Ni),chii]_xy
    dg::blas1::axpby(-1.,omega,1.0,yp[3]);   

    //curvature dynamics
    for(unsigned i=0; i<2; i++)
    {
        //n*K(psi) terms
        dg::blas2::gemv( poisson.dyrhs(), phi[i], lambda); //dy phi
        dg::blas1::pointwiseDot(lambda,ype[i],omega); // n dy phi
        dg::blas1::axpby(p.mcv,omega,1.0,yp[i]);   // dtN +=  mcv* N dy psi
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); // n dy phi
        dg::blas1::axpby(p.mcv,omega,1.0,yp[i+2]);   // dtT +=  mcv* T dy psi
        // K(T n) terms
        dg::blas2::gemv( poisson.dyrhs(), y[i], lambda); //dy (N-1)
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); // T dy (N-1)
        dg::blas1::axpby(p.tau[i]*p.mcv,omega,1.0,yp[i]); 
        dg::blas2::gemv( poisson.dyrhs(), y[i+2], lambda); //dy (T-1)
        dg::blas1::pointwiseDot(lambda,ype[i],omega); // N dy (T-1)
        dg::blas1::axpby(p.tau[i]*p.mcv,omega,1.0,yp[i]); 
        //T*K(T) terms
        dg::blas2::gemv( poisson.dyrhs(), y[i+2], lambda); //dy (T-1)
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); // T dy (T-1)
        dg::blas1::axpby(3.*p.tau[i]*p.mcv,omega,1.0,yp[i+2]); 
        //T^2*K(ln(T)) terms
        dg::blas2::gemv( poisson.dyrhs(), logype[i], lambda); //dy (ln(N))
        dg::blas1::pointwiseDot(lambda,ype[i+2],omega); // T dy (ln(N))
        dg::blas1::pointwiseDot(omega,ype[i+2],omega); // T^2 dy (ln(N))
        dg::blas1::axpby(p.tau[i]*p.mcv,omega,1.0,yp[i+2]); 
        
    }   
    //add FLR correction to curvature dynamics
    //Ni K(chii) and Ti K(chii) term
    dg::blas2::gemv( poisson.dyrhs(), chii, lambda); //dy chii
    dg::blas1::pointwiseDot(lambda,ype[1],omega); // Ni dy chii
    dg::blas1::axpby(p.mcv,omega,1.0,yp[1]);   // dtNi +=  mcv* Ni dy chii
    dg::blas1::pointwiseDot(lambda,ype[3],omega); // Ti dy chii
    dg::blas1::axpby(3.*p.mcv,omega,1.0,yp[3]);   // dtTi += 3.* mcv* Ti dy chii
    //Ni chii K(lnTi + lnNi) term
    dg::blas1::axpby(1.,logype[1],1.0,logype[3],omega); //ln(Ti)+ln(Ni)
    dg::blas2::gemv( poisson.dyrhs(), omega, lambda); //dy(ln(Ti)+ln(Ni))
    dg::blas1::pointwiseDot(lambda,ype[1],omega); // Ni dy(ln(Ti)+ln(Ni))
    dg::blas1::pointwiseDot(omega,chii,omega); // Ni  chii dy(ln(Ti)+ln(Ni))
    dg::blas1::axpby(p.mcv,omega,1.0,yp[1]);   // dtNi +=  mcv* Ni  chii dy(ln(Ti)+ln(Ni))
    //chii K(Ti) term
    dg::blas2::gemv( poisson.dyrhs(), y[3], lambda); //dy (Ti-1)
    dg::blas1::pointwiseDot(lambda,chii,omega); // chii dy (Ti-1)
    dg::blas1::axpby(p.mcv,omega,1.0,yp[3]);   // dtNi +=  mcv*  chii dy (Ti-1)
    //Ti chii K(lnNi)) term
    dg::blas2::gemv( poisson.dyrhs(), logype[3], lambda); //dy (ln(Ni))
    dg::blas1::pointwiseDot(lambda,chii,omega); // chii dy (Ti-1)
    dg::blas1::pointwiseDot(omega,ype[3],omega); // Ti chii dy (Ti-1)
    dg::blas1::axpby(p.mcv,omega,1.0,yp[3]);   // dtNi +=  mcv*  chii dy (Ti-1)
        
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

