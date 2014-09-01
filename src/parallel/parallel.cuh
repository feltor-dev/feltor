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
 * @brief Diffusive terms for ParallelFeltor solver
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
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::axpby( -p.nu_perp, y[i], 0., y[i]); //  nu_perp lapl_RZ (lapl_RZ (lnN,U)) 
        }
//         dg::blas1::scal( y[2],-1./p.mu[0]); //factor nu_e on U_e
        //add parallel resistivity
        for( unsigned i=0; i<2; i++)
            dg::blas1::transform( x[i], expy[i], dg::EXP<double>());
        dg::blas1::pointwiseDot( expy[0], x[2], omega); //N_e U_e 
        dg::blas1::pointwiseDot( expy[1], x[3], chi); //N_i U_i
        dg::blas1::axpby( -1., omega, 1., chi); //-N_e U_e + N_i U_i
        dg::blas1::pointwiseDivide( chi, expy[0], omega);//J_par/N_e
        dg::blas1::axpby( -p.c/p.mu[0]/p.eps_hat, omega, 1., y[2]);  // dtU_e =- C/hat(mu)_e J_par/N_e
        dg::blas1::axpby( -p.c/p.mu[1]/p.eps_hat,omega, 1., y[3]);    // dtU_e =- C/hat(mu)_i J_par/N_i   //n_e instead of n_i
        
        //cut contributions to boundary now with damping on all 4 quantities
        for( unsigned i=0; i<y.size(); i++){
            dg::blas1::pointwiseDot( dampgauss_, y[i], y[i]);
        }
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
struct ParallelFeltor
{
    //typedef std::vector<container> Vector;
    typedef typename dg::VectorTraits<container>::value_type value_type;
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    //typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    template<class Grid3d>
    ParallelFeltor( const Grid3d& g, eule::Parameters p,solovev::GeomParameters gp);

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

    container chi, omega;

    const container binv, curvR, curvZ, gradlnB;
    const container pupil, source, damping, one;
    const Preconditioner w3d, v3d;

    std::vector<container> phi, curvphi, dzphi, expy;
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
ParallelFeltor<Matrix, container, P>::ParallelFeltor( const Grid& g, eule::Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    pupil( dg::evaluate( solovev::Pupil( gp), g)),
    source( dg::evaluate(solovev::TanhSource(gp, p.amp_source), g)),
    damping( dg::evaluate( solovev::GaussianDamping(gp ), g)), 
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)), 
    phi( 2, chi), dzphi(phi), expy(phi),  
    dzy( 4, chi), 
    dz(solovev::Field(gp), g, gp.rk4eps, dg::DefaultLimiter()),
    arakawa( g), 
    pol(     g), 
    invgamma(g,-0.5*p.tau[1]*p.mu[1]),
    invert_pol( omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    p(p),
    gp(gp)
{ }
template< class Matrix, class container, class P>
container& ParallelFeltor<Matrix,container, P>::compute_vesqr( container& potential)
{
    arakawa.bracketS( potential, potential, chi);
    dg::blas1::pointwiseDot( binv, binv, omega);
    dg::blas1::pointwiseDot( chi, omega, omega);
    return omega;
}
template< class Matrix, class container, class P>
container& ParallelFeltor<Matrix,container, P>::compute_psi( container& potential)
{
    //without FLR
//     dg::blas1::axpby( 1., potential, -0.5, compute_vesqr( potential), phi[1]);
    //with FLR
    invert_invgamma(invgamma,chi,potential);
    dg::blas1::axpby( 1., chi, -0.5, compute_vesqr( potential),phi[1]);    
    return phi[1];
    
}
template<class Matrix, class container, class P>
void ParallelFeltor<Matrix, container, P>::initializene( const container& src, container& target)
{ 

    dg::blas1::transform( src,omega, dg::PLUS<double>(-1)); //n_i -1
    invert_invgamma(invgamma,target,omega); //=ne-1 = Gamma (ni-1)    
    dg::blas1::transform( target,target, dg::PLUS<double>(+1)); //n_i

}

//computes and modifies expy!!
template<class Matrix, class container, class P>
container& ParallelFeltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
    //compute chi and polarisation
    exp( y, expy, 2);
    dg::blas1::axpby( 1., expy[1], 0., chi); //\chi = a_i \mu_i n_i
    //correction
    dg::blas1::axpby( -p.mu[0], expy[0], 1., chi); //\chi = a_i \mu_i n_i -a_e \mu_e n_e
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi); //chi/= B^2

    //A = pol.create( chi);
    pol.set_chi( chi);
    dg::blas1::transform( expy[0], expy[0], dg::PLUS<double>(-1)); //n_e -1
    dg::blas1::transform( expy[1], omega,   dg::PLUS<double>(-1)); //n_i -1
    //with FLR
    unsigned number =  invert_invgamma(invgamma,chi,omega);    //chi= Gamma (Omega) = Gamma (ni-1)
/*    if( numberg == invert_invgamma.get_max())
        throw dg::Fail( p.eps_gamma);*/  
    dg::blas1::axpby( -1., expy[0], 1.,chi); //chi=  Gamma (n_i-1) - (n_e-1) = Gamma n_1 - n_e
    number = invert_pol( pol, phi[0], chi); //Gamma n_i -ne = -nabla chi nabla phi

    if( number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template<class Matrix, class container, class P>
void ParallelFeltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 4);
    assert( y.size() == yp.size());
    //compute phi via polarisation
    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    // the resistive dissipation without FLR      
    dg::blas1::pointwiseDot( expy[0], y[2], omega); //N_e U_e 
    dg::blas1::pointwiseDot( expy[1], y[3], chi); //N_i U_i
    dg::blas1::axpby( -1., omega, 1., chi); //-N_e U_e + N_i U_i                  //dt(lnN,U) = dt(lnN,U) + dz (dz (lnN,U))
    //double Dres = -p.c*dg::blas2::dot(chi, w3d, chi); //- C*J_parallel^2

    for( unsigned i=0; i<2; i++)
    {
        //compute parallel derivatives
        dz(y[i], dzy[i]);
        dz(phi[i], dzphi[i]);
        dz(y[i+2], dzy[2+i]);

        //parallel advection terms
        dg::blas1::pointwiseDot(y[i+2], dzy[i], omega); //Udz lnN 
        dg::blas1::axpby( -1., omega, 1., yp[i]);                            //dtlnN = dtlnN - Udz lnN
        dg::blas1::axpby( -1., dzy[2+i], 1., yp[i]);                         //dtlnN = dtlnN - dz U
        dg::blas1::pointwiseDot(y[i+2], gradlnB, omega);                     
        dg::blas1::axpby( 1., omega, 1., yp[i]);                            //dtlnN = dtlnN + U dz ln B
        dg::blas1::pointwiseDot(y[i+2], dzy[2+i], omega);                    
        dg::blas1::axpby( -1., omega, 1., yp[2+i]);                         //dtU = dtU - U dz U
        //parallel force terms
        dg::blas1::axpby( -p.tau[i]/p.mu[i]/p.eps_hat, dzy[i], 1., yp[2+i]); //dtU = dtU - tau/(hat(mu))*dz lnN
        dg::blas1::axpby( -1./p.mu[i]/p.eps_hat, dzphi[i], 1., yp[2+i]);     //dtU = dtU - 1/(hat(mu))*dz phi 
    }

    for( unsigned i=0; i<4; i++)
    {
        dz.dzz(y[i], omega); //dz (dz (N,U))
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i]);               
        //gradlnBcorrection
        dg::blas1::pointwiseDot(gradlnB,dzy[i], omega);    
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[i]);    
    }

    //add particle source to dtN
//     for( unsigned i=0; i<2; i++)
//     {
//         dg::blas1::pointwiseDivide( source, expy[i], omega); //source/N
//         dg::blas1::axpby( 1., omega, 1, yp[i]  );       //dtlnN = dtlnN + source/N
//     }

    for( unsigned i=0; i<4; i++) //damping and pupil on N and w
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]); 
}

//Exp
template<class Matrix, class container, class P>
void ParallelFeltor<Matrix, container, P>::exp( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        dg::blas1::transform( y[i], target[i], dg::EXP<value_type>());
}
//Log
template< class M, class container, class P>
void ParallelFeltor<M, container, P>::log( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        dg::blas1::transform( y[i], target[i], dg::LN<value_type>());
}

///@}

} //namespace eule
