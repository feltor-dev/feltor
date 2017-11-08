#pragma once 
#include <exception>

#include "dg/algorithm.h"
#include "dg/backend/typedefs.cuh"
#include "parameters.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif


namespace toefl
{

template<class Geometry, class Matrix, class container>
struct Implicit
{
    Implicit( const Geometry& g, double nu):
        nu_(nu), LaplacianM_perp( g, dg::normed, dg::centered){ }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
         * x[2] := N_i - 1 
         */
        for( unsigned i=0; i<x.size(); i++)
        {
            //dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            //dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            //dg::blas1::axpby( -nu_, y[i], 0., y[i]);
            dg::blas2::gemv( LaplacianM_perp, x[i], y[i]);
            dg::blas1::scal( y[i], -nu_);
        }
    }
    dg::Elliptic<Geometry, Matrix, container>& laplacianM() {return LaplacianM_perp;}
    const container& weights(){return LaplacianM_perp.weights();}
    const container& inv_weights(){return LaplacianM_perp.inv_weights();}
    const container& precond(){return LaplacianM_perp.precond();}

  private:
    double nu_;
    container temp;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perp;
};

template< class Geometry,  class Matrix, class container >
struct Explicit
{
    /**
     * @brief Construct a Explicit solver object
     *
     * @param g The grid on which to operate
     * @param p The parameters
     */
    Explicit( const Geometry& g, const Parameters& p );


    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}

    /**
     * @brief Return the normalized negative laplacian used by this object
     *
     * @return cusp matrix
     */
    dg::Elliptic<Geometry, Matrix, container>& laplacianM( ) { return laplaceM;}

    /**
     * @brief Return the Gamma operator used by this object
     *
     * @return Gamma operator
     */
    dg::Helmholtz<Geometry, Matrix, container >&  gamma() {return multi_gamma1[0];}

    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * y[0] = N_e - 1, 
     * y[1] = N_i - 1 || y[1] = Omega
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( const std::vector<container>& y, std::vector<container>& yp);

    /**
     * @brief Return the mass of the last field in operator() in a global computation
     *
     * @return int exp(y[0]) dA
     * @note undefined for a local computation
     */
    double mass( ) {return mass_;}
    /**
     * @brief Return the last integrated mass diffusion of operator() in a global computation
     *
     * @return int \nu \Delta (exp(y[0])-1)
     * @note undefined for a local computation
     */
    double mass_diffusion( ) {return diff_;}
    /**
     * @brief Return the energy of the last field in operator() in a global computation
     *
     * @return integrated total energy in {ne, ni}
     * @note undefined for a local computation
     */
    double energy( ) {return energy_;}
    /**
     * @brief Return the integrated energy diffusion of the last field in operator() in a global computation
     *
     * @return integrated total energy diffusion
     * @note undefined for a local computation
     */
    double energy_diffusion( ){ return ediff_;}

  private:
    //use chi and omega as helpers to compute square velocity in omega
    const container& compute_psi( const container& potential);
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;
    const container binv; //magnetic field

    std::vector<container> phi, dyphi, ype;
    std::vector<container> dyy, lny, lapy;
    container gamma_n;

    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, container> pol, laplaceM; //contains normalized laplacian
    std::vector<dg::Elliptic<Geometry, Matrix, container> > multi_pol;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > multi_gamma1;
    dg::ArakawaX< Geometry, Matrix, container> arakawa; 

    dg::Invert<container> invert_invgamma, invert_pol;
    dg::MultigridCG2d<Geometry, Matrix, container> multigrid;
    dg::Extrapolation<container> old_phi, old_psi, old_gammaN;
    std::vector<container> multi_chi;

    const container w2d, v2d, one;
    const double eps_pol, eps_gamma; 
    const double kappa, friction, nu, tau;
    const std::string equations;
    bool boussinesq;

    double mass_, energy_, diff_, ediff_;

};

template< class Geometry, class M, class container>
Explicit< Geometry, M, container>::Explicit( const Geometry& grid, const Parameters& p ): 
    chi( evaluate( dg::zero, grid)), omega(chi),
    binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)), 
    phi( 2, chi), dyphi( phi), ype(phi),
    dyy(2,chi), lny( dyy), lapy(dyy),
    gamma_n(chi),
    pol(     grid, dg::not_normed, dg::centered, p.jfactor), 
    laplaceM( grid, dg::normed, dg::centered),
    arakawa( grid), 
    invert_invgamma( omega, p.Nx*p.Ny*p.n*p.n, p.eps_gamma),
    invert_pol(      omega, p.Nx*p.Ny*p.n*p.n, p.eps_pol),
    multigrid( grid, 3),
    old_phi( 2, chi), old_psi( 2, chi), old_gammaN( 2, chi), 
    w2d( dg::create::volume(grid)), v2d( dg::create::inv_volume(grid)), one( dg::evaluate(dg::one, grid)),
    eps_pol(p.eps_pol), eps_gamma( p.eps_gamma), kappa(p.kappa), friction(p.friction), nu(p.nu), tau( p.tau), equations( p.equations), boussinesq(p.boussinesq)
{ 
    multi_chi= multigrid.project( chi);
    multi_pol.resize(3);
    multi_gamma1.resize(3);
    for( unsigned u=0; u<3; u++)
    {
        multi_pol[u].construct( multigrid.grids()[u].get(), dg::not_normed, dg::centered, p.jfactor);
        multi_gamma1[u].construct( multigrid.grids()[u].get(), -0.5*p.tau, dg::centered);
    }
}

template< class G, class M, class container>
const container& Explicit<G, M, container>::compute_psi( const container& potential)
{
    if(equations == "gravity_local") return potential;
    //in gyrofluid invert Gamma operator
    if( equations == "local" || equations == "global")
    {
        old_psi.extrapolate( phi[1]);
        std::vector<unsigned> number = multigrid.direct_solve( multi_gamma1, phi[1], potential, eps_gamma);
        old_psi.update( phi[1]);
        //unsigned number = invert_invgamma( gamma1, phi[1], potential);
        if(  number[0] == invert_invgamma.get_max())
            throw dg::Fail( eps_gamma);
    }
    //compute (nabla phi)^2
    arakawa.variation(potential, omega); 
    //compute psi
    if(equations == "global")
    {
        dg::blas1::pointwiseDot( binv, omega, omega);
        dg::blas1::pointwiseDot( binv, omega, omega);

        dg::blas1::axpby( 1., phi[1], -0.5, omega, phi[1]);   //psi  Gamma phi - 0.5 u_E^2
    }
    else if ( equations == "drift_global")
    {
        dg::blas1::pointwiseDot( binv, omega, omega);
        dg::blas1::pointwiseDot( binv, omega, omega);
        dg::blas1::axpby( 0.5, omega, 0., phi[1]);
    }
    else if( equations == "gravity_global" ) 
        dg::blas1::axpby( 0.5, omega, 0., phi[1]);
    return phi[1];    
}


//computes and modifies expy!!
template<class G, class M, class container>
const container& Explicit<G, M, container>::polarisation( const std::vector<container>& y)
{
    //compute chi 
    if(equations == "global" )
    {
        dg::blas1::transfer( y[1], chi);
        dg::blas1::plus( chi, 1.); 
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi = n_i
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
        if( !boussinesq) 
        {
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
                multi_pol[u].set_chi( multi_chi[u]);
            //pol.set_chi( chi);
        }
    }
    else if(equations == "gravity_global" )
    {
        dg::blas1::transfer( y[0], chi);
        dg::blas1::plus( chi, 1.); 
        if( !boussinesq) 
        {
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
                multi_pol[u].set_chi( multi_chi[u]);
            //pol.set_chi( chi);
        }
    }
    else if( equations == "drift_global" )
    {
        dg::blas1::transfer( y[0], chi);
        dg::blas1::plus( chi, 1.); 
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi = n_e
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
        if( !boussinesq) 
        {
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
                multi_pol[u].set_chi( multi_chi[u]);
            //pol.set_chi( chi);
        }
    }
    //compute polarisation
    if( equations == "local" || equations == "global")
    {
        old_gammaN.extrapolate( gamma_n);
        std::vector<unsigned> number = multigrid.direct_solve( multi_gamma1, gamma_n, y[1], eps_gamma);
        old_gammaN.update( gamma_n);
        //unsigned number = invert_invgamma( gamma1, gamma_n, y[1]);
        if(  number[0] == invert_invgamma.get_max())
            throw dg::Fail( eps_gamma);
        dg::blas1::axpby( -1., y[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
    }
    else 
        dg::blas1::axpby( -1. ,y[1], 0., omega);
    if( equations == "global" || equations == "gravity_global" || equations == "drift_global")
        if( boussinesq) 
            dg::blas1::pointwiseDivide( omega, chi, omega);
    //invert 

    //unsigned number = invert_pol( pol, phi[0], omega, v2d, chi);
    old_phi.extrapolate( phi[0]);
    std::vector<unsigned> number = multigrid.direct_solve( multi_pol, phi[0], omega, eps_pol);
    old_phi.update( phi[0]);
    if(  number[0] == invert_pol.get_max())
        throw dg::Fail( eps_pol);
    return phi[0];
}

template< class G, class M, class container>
void Explicit<G, M, container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    //y[0] = N_e - 1
    //y[1] = N_i - 1 || y[1] = Omega
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    for( unsigned i=0; i<y.size(); i++)
    {
        dg::blas1::transform( y[i], ype[i], dg::PLUS<double>(1.));
        dg::blas1::transform( ype[i], lny[i], dg::LN<double>()); 
        dg::blas2::symv( laplaceM, y[i], lapy[i]);
    }

    /////////////////////////update energetics, 2% of total time///////////////
    mass_ = dg::blas2::dot( one, w2d, y[0] ); //take real ion density which is electron density!!
    diff_ = nu*dg::blas2::dot( one, w2d, lapy[0]);
    if(equations == "global")
    {
        double Ue = dg::blas2::dot( lny[0], w2d, ype[0]);
        double Ui = tau*dg::blas2::dot( lny[1], w2d, ype[1]);
        double Uphi = 0.5*dg::blas2::dot( ype[1], w2d, omega); 
        energy_ = Ue + Ui + Uphi;

        double Ge = - dg::blas2::dot( one, w2d, lapy[0]) - dg::blas2::dot( lapy[0], w2d, lny[0]); // minus 
        double Gi = - tau*(dg::blas2::dot( one, w2d, lapy[1]) + dg::blas2::dot( lapy[1], w2d, lny[1])); // minus 
        double Gphi = -dg::blas2::dot( phi[0], w2d, lapy[0]);
        double Gpsi = -dg::blas2::dot( phi[1], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge + Gi - Gphi + Gpsi);
    }
    else if ( equations == "drift_global") 
    {
        double Se = dg::blas2::dot( lny[0], w2d, ype[0]);
        double Ephi = 0.5*dg::blas2::dot( ype[0], w2d, omega); 
        energy_ = Se + Ephi;

        double Ge = - dg::blas2::dot( one, w2d, lapy[0]) - dg::blas2::dot( lapy[0], w2d, lny[0]); // minus 
        double GeE = - dg::blas2::dot( phi[1], w2d, lapy[0]); 
        double Gpsi = -dg::blas2::dot( phi[0], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge - GeE - Gpsi);
    }
    else if(equations == "gravity_global" || equations == "gravity_local")
    {
        energy_ = 0.5*dg::blas2::dot( y[0], w2d, y[0]);
        double Ge = - dg::blas2::dot( y[0], w2d, lapy[0]);
        ediff_ = nu* Ge;
    }
    else
    {
        double Ue = 0.5*dg::blas2::dot( y[0], w2d, y[0]);
        double Ui = 0.5*tau*dg::blas2::dot( y[1], w2d, y[1]);
        double Uphi = 0.5*dg::blas2::dot( one, w2d, omega); 
        energy_ = Ue + Ui + Uphi;

        double Ge = - dg::blas2::dot( y[0], w2d, lapy[0]); // minus 
        double Gi = - tau*(dg::blas2::dot( y[1], w2d, lapy[1])); // minus 
        double Gphi = -dg::blas2::dot( phi[0], w2d, lapy[0]);
        double Gpsi = -dg::blas2::dot( phi[1], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge + Gi - Gphi + Gpsi);
    }
    ///////////////////////////////////////////////////////////////////////
    if( equations == "gravity_global")
    {
        arakawa(y[0], phi[0], yp[0]);
        arakawa(y[1], phi[0], yp[1]);
        arakawa(y[0], phi[1], omega);
        dg::blas1::axpbypgz( 1., omega, -friction, y[1], 1., yp[1]);
        dg::blas2::gemv( 1., arakawa.dy(), y[0], 1., yp[1]);
        return;
    }
    else if( equations == "gravity_local")
    {
        arakawa(y[0], phi[0], yp[0]);
        arakawa(y[1], phi[0], yp[1]);
        dg::blas2::gemv( arakawa.dy(), y[0], dyy[0]);
        dg::blas1::axpbypgz( -friction, y[1], -1., dyy[0], 1., yp[1]);
        return;
    }
    else if( equations == "drift_global")
    {
        arakawa(y[0], phi[0], yp[0]);
        arakawa(y[1], phi[0], yp[1]);
        arakawa(y[0], phi[1], omega);
        dg::blas1::pointwiseDot( binv, yp[0], yp[0]);
        dg::blas1::pointwiseDot( binv, yp[1], yp[1]);
        dg::blas1::pointwiseDot( 1., binv, omega, 1., yp[1]);

        dg::blas2::gemv( arakawa.dy(), phi[0], dyphi[0]);
        dg::blas2::gemv( arakawa.dy(), phi[1], dyphi[1]);
        //ExB compression
        dg::blas1::pointwiseDot( kappa, dyphi[0], ype[0], 1., yp[0]);
        dg::blas1::pointwiseDot( kappa, dyphi[0], y[1], kappa, dyphi[1], ype[0], 1.,  yp[1]);
        // diamagnetic compression
        dg::blas2::gemv( -kappa, arakawa.dy(), y[0], 1., yp[1]);
        return;
    }
    else
    {
        for( unsigned i=0; i<y.size(); i++)
        {
            arakawa( y[i], phi[i], yp[i]);
            if(equations == "global") dg::blas1::pointwiseDot( binv, yp[i], yp[i]);
        }
        double _tau[2] = {-1., tau};
        //compute derivatives and exb compression
        for( unsigned i=0; i<y.size(); i++)
        {
            dg::blas2::gemv( arakawa.dy(), y[i], dyy[i]);
            dg::blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);
            if(equations == "global") dg::blas1::pointwiseDot( dyphi[i], ype[i], dyphi[i]);
            dg::blas1::axpbypgz( kappa, dyphi[i], _tau[i]*kappa, dyy[i], 1., yp[i]);
        }
    }

    return;
}

}//namespace dg
