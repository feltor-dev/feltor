#pragma once

#include <exception>

#include "dg/algorithm.h"
#include "dg/backend/typedefs.cuh"
#include "parameters.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif


namespace dg
{

template<class Geometry, class Matrix, class container>
struct Diffusion
{
    Diffusion( const Geometry& g, double nu):
        nu_(nu), 
        temp( dg::evaluate(dg::zero, g)), 
        LaplacianM_perp( g, dg::normed, dg::centered){
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
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
    const container& precond(){return LaplacianM_perp.precond();}

  private:
    double nu_;
    const container w2d, v2d;
    container temp;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perp;
};

template< class Geometry,  class Matrix, class container >
struct ToeflR
{
    /**
     * @brief Construct a ToeflR solver object
     *
     * @param g The grid on which to operate
     * @param p The parameters
     */
    ToeflR( const Geometry& g, const Parameters& p );


    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const container& potential( ) const { return potential_;}

    /**
     * @brief Return the normalized negative laplacian used by this object
     *
     * @return cusp matrix
     */
    dg::Elliptic<Geometry, Matrix, container>& laplacianM( ) { return laplaceM;}


    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * y[0] = N_e - 1, 
     * y[1] = N_i - 1 || y[1] = Omega
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( std::vector<container>& y, std::vector<container>& yp);

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
    void compute_psi( const container& potential);
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;
    const container binv; //magnetic field

    std::vector<container> psi, dypsi, ype;
    std::vector<container> dyy, lny, lapy;
    container gamma_n, potential_;

    //matrices and solvers
    Elliptic<Geometry, Matrix, container> pol, laplaceM; //contains normalized laplacian
    Helmholtz<Geometry,  Matrix, container> gamma1;
    ArakawaX< Geometry, Matrix, container> arakawa; 

    dg::Invert<container> invert_pol, invert_invgamma;

    const container w2d, v2d, one;
    const double eps_pol, eps_gamma, kappa, nu; 
    double tau[2], mu[2], z[2];
    const std::string equations;
    bool boussinesq;

    double mass_, energy_, diff_, ediff_;
    const double debye_;

};

template< class Geometry, class M, class container>
ToeflR< Geometry, M, container>::ToeflR( const Geometry& grid, const Parameters& p ): 
    chi( evaluate( dg::zero, grid)), omega(chi),
    binv( evaluate( LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)), 
    gamma_n(chi), potential_(chi), psi( 2, chi), dypsi( psi), ype(psi),
    dyy(2,chi), lny( dyy), lapy(dyy), 
    pol(     grid, not_normed, dg::centered), 
    laplaceM( grid, normed, centered),
    gamma1(  grid, 0., dg::centered),
    arakawa( grid), 
    invert_pol(      omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    w2d( create::volume(grid)), v2d( create::inv_volume(grid)), one( dg::evaluate(dg::one, grid)),
    eps_pol(p.eps_pol), eps_gamma( p.eps_gamma), kappa(p.kappa), nu(p.nu), debye_( p.debye)
{
    tau[0] = p.tau[0], tau[1] = p.tau[1];
    mu[0] = p.mu[0], mu[1] = p.mu[1];
    z[0] = p.z[0], z[1] = p.z[1];
}

template< class G, class M, class container>
void ToeflR<G, M, container>::compute_psi( const container& phi)
{
    for( unsigned i=0; i<2; i++)
    {
        gamma1.alpha() = -0.5*mu[i]*tau[i];
        unsigned number = invert_invgamma( gamma1, psi[i], phi);
        if(  number == invert_invgamma.get_max())
            throw dg::Fail( eps_gamma);

        arakawa.variation(phi, omega); //needed also in local energy theorem
        dg::blas1::pointwiseDot( binv, omega, omega);
        dg::blas1::pointwiseDot( binv, omega, omega);
        dg::blas1::axpby( 1., psi[i], -0.5*mu[i], omega, psi[i]);   //psi  Gamma phi - 0.5 u_E^2
    }
}


template<class G, class M, class container>
const container& ToeflR<G, M, container>::polarisation( const std::vector<container>& y)
{
    //compute chi and polarisation
    dg::blas1::scal( chi, 0.);
    dg::blas1::plus( chi, debye_); 
    for( unsigned i=0; i<2; i++)
    {
        dg::blas1::transfer( y[i], omega);
        dg::blas1::plus( omega, 1.); 
        dg::blas1::scal( omega, z[i]*mu[i]);
        blas1::pointwiseDot( binv, omega, omega); 
        blas1::pointwiseDot( binv, omega, omega); //\omega *= binv^2
        dg::blas1::axpby( 1., omega, 1, chi);
    }
    pol.set_chi( chi);

    dg::blas1::scal( omega, 0.);
    for( unsigned i=0; i<2; i++)
    {
        gamma1.alpha() = -0.5*tau[i]*mu[i];
        unsigned number = invert_invgamma( gamma1, gamma_n, y[i]);
        if(  number == invert_invgamma.get_max())
            throw dg::Fail( eps_gamma);
        blas1::axpby( z[i], gamma_n, 1., omega); 
    }
    unsigned number = invert_pol( pol, potential_, omega);
    if(  number == invert_pol.get_max())
        throw dg::Fail( eps_pol);
    return potential_;
}

template< class G, class M, class container>
void ToeflR<G, M, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    //y[0] = N_e - 1
    //y[1] = N_i - 1 || y[1] = Omega
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    potential_ = polarisation( y);
    compute_psi(potential_);

    { //update energetics, 2% of total time
    for( unsigned i=0; i<y.size(); i++)
    {
        dg::blas1::transform( y[i], ype[i], dg::PLUS<double>(1.));
        dg::blas1::transform( ype[i], lny[i], dg::LN<double>()); 
        dg::blas2::symv( laplaceM, y[i], lapy[i]);
    }

        mass_ = blas2::dot( one, w2d, y[0] ) + blas2::dot( one, w2d, y[1]); //take real ion density which is electron density!!
        diff_ = nu*( blas2::dot( one, w2d, lapy[0]) + blas2::dot( one, w2d, lapy[1]));
        double Ue = z[0]*tau[0]*blas2::dot( lny[0], w2d, ype[0]);
        double Up = z[1]*tau[1]*blas2::dot( lny[1], w2d, ype[1]);
        double Uphi = 0.5*blas2::dot( ype[0], w2d, omega) + 0.5*blas2::dot( ype[1], w2d, omega); 
        arakawa.variation(potential_, omega); 
        double UE = debye_*dg::blas2::dot( one, w2d, omega);
        energy_ = Ue + Up + Uphi + UE;
        //std::cout << "Ue "<<Ue<< "Up "<<Up<< "Uphi "<<Uphi<< "UE "<<UE<<"\n";

        double Ge = - tau[0]*(blas2::dot( one, w2d, lapy[0]) + blas2::dot( lapy[0], w2d, lny[0])); // minus because of laplace
        double Gp = - tau[1]*(blas2::dot( one, w2d, lapy[1]) + blas2::dot( lapy[1], w2d, lny[1])); // minus because of laplace
        double Gpsie = -blas2::dot( psi[0], w2d, lapy[0]);
        double Gpsip = -blas2::dot( psi[1], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gp "<<Gp<<" gpsie "<<Gpsie<<" gpsip "<<Gpsip<<"\n";
        ediff_ = nu*( z[0]*Ge + z[1]*Gp + z[0]*Gpsie + z[1]*Gpsip);
    }


    for( unsigned i=0; i<y.size(); i++)
    {
        arakawa( y[i], psi[i], yp[i]);
        blas1::pointwiseDot( binv, yp[i], yp[i]);
        blas2::gemv( arakawa.dy(), y[i], dyy[i]);
        blas2::gemv( arakawa.dy(), psi[i], dypsi[i]);
        blas1::pointwiseDot( dypsi[i], ype[i], dypsi[i]);
        blas1::axpby( kappa, dypsi[i], 1., yp[i]);
        blas1::axpby( tau[i]*kappa, dyy[i], 1., yp[i]);
    }
    return;
}

}//namespace dg

