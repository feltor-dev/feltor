#ifndef _DG_TOEFLR_CUH
#define _DG_TOEFLR_CUH

#include <exception>

#include "dg/algorithm.h"
#include "dg/backend/typedefs.cuh"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif


//TODO es wäre besser, wenn ToeflR auch einen Zeitschritt berechnen würde 
// dann wäre die Rückgabe der Felder (Potential vs. Masse vs. exp( y)) konsistenter
// (nur das Objekt weiß welches Feld zu welchem Zeitschritt gehört)

namespace dg
{
template<class Geometry, class Matrix, class container>
struct Diffusion
{
    Diffusion( const Geometry& g, double nu, bool global):
        nu_(nu), global(global), 
        temp( dg::evaluate(dg::zero, g)), 
        LaplacianM_perp( g, dg::normed, dg::centered){
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
         * x[2] := N_i -1 
         */
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::axpby( -nu_, y[i], 0., y[i]);
        }
    }
    dg::Elliptic<Geometry, Matrix, container>& laplacianM() {return LaplacianM_perp;}
    const container& weights(){return LaplacianM_perp.weights();}
    const container& precond(){return LaplacianM_perp.precond();}

  private:
    double nu_;
    bool global;
    const container w2d, v2d;
    container temp;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perp;
};

template< class Matrix, class container >
struct ToeflR
{
    /**
     * @brief Construct a ToeflR solver object
     *
     * @param g The grid on which to operate
     * @param kappa The curvature
     * @param nu The artificial viscosity
     * @param tau The ion temperature
     * @param eps_pol stopping criterion for polarisation equation
     * @param eps_gamma stopping criterion for Gamma operator
     * @param global local or global computation
     */
    ToeflR( const CartesianGrid2d& g, double kappa, double nu, double tau, double eps_pol, double eps_gamma, int global);


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
    dg::Helmholtz<Geometry, Matrix, container >&  gamma() {return gamma1;}

    /**
     * @brief Compute the right-hand side of the toefl equations
     *
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
    const container& compute_psi( const container& potential);
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;
    const container binv; //magnetic field

    std::vector<container> phi, dyphi, ype;
    std::vector<container> dyy, lny, lapy;
    container gamma_n;

    //matrices and solvers
    Elliptic<Geometry, Matrix, container> pol, laplaceM; //contains normalized laplacian
    Helmholtz<Geometry,  Matrix, container> gamma1;
    ArakawaX< Geometry, Matrix, container> arakawa; 

    dg::Invert<container> invert_pol, invert_invgamma;

    const container w2d, v2d, one;
    const double eps_pol, eps_gamma; 
    const double kappa, nu, tau;
    const int global;

    double mass_, energy_, diff_, ediff_;

};

template< class Geometry, class M, class container>
ToeflR< Geometry, M, container>::ToeflR( const Geometry& grid, double kappa, double nu, double tau, double eps_pol, double eps_gamma, int global ): 
    chi( evaluate( dg::zero, grid)), omega(chi),
    binv( evaluate( LinearX( kappa, 1.), grid)), 
    phi( 2, chi), dyphi( phi), ype(phi),
    gamma_n(chi),
    dyy(2,chi), lny( dyy), lapy(dyy),
    gamma1(  grid, -0.5*tau, dg::centered),
    arakawa( grid), 
    pol(     grid, not_normed, dg::centered), 
    laplaceM( grid, normed, centered),
    invert_pol(      omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    w2d( create::volume(grid)), v2d( create::inv_volume(grid)), one( dg::evaluate(dg::one, grid)),
    eps_pol(eps_pol), eps_gamma( eps_gamma), kappa(kappa), nu(nu), tau( tau), global( global)
{
}

template< class G, class M, class container>
const container& ToeflR<G, M, container>::compute_psi( const container& potential)
{
    invert_invgamma( gamma1, phi[1], potential);

    arakawa.variation(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);

    dg::blas1::axpby( 1., phi[1], -0.5, omega, phi[1]);   //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}
}


//computes and modifies expy!!
template<class G, class M, class container>
const container& ToeflR<G, M, container>::polarisation( const std::vector<container>& y)
{
#ifdef DG_BENCHMARK
    Timer t; 
    t.tic();
#endif
    //compute chi and polarisation
    if( global) 
    {
        dg::blas1::transfer( y[1], chi);
        dg::blas1::plus( chi, 1.); 
        blas1::pointwiseDot( binv, chi, chi); //\chi = n_i
        blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
        if( global == 1)
            pol.set_chi( chi);
    }
    else
    {
        blas1::axpby( 1., y[1], 0., omega); //n_i = omega
    }
#ifdef DG_BENCHMARK
    t.toc();
    std::cout<< "Polarisation assembly took "<<t.diff()<<"s\n";
    t.tic();
#endif 
    invert_invgamma( gamma1, gamma_n, y[1]);
    if( number == pcg.get_max())
        throw Fail( eps_gamma);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for n_i \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
    t.tic();
#endif 
    if( global)
    {
        blas1::axpby( -1., y[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
        if( global == 2)
            divide( omega, expy[1], omega);
        blas2::symv( w2d, omega, omega);
    }
    else
    {
        blas1::axpby( -1, y[0], 1., gamma_n, chi); 
        gamma1.alpha() = -tau;
        blas2::symv( gamma1, chi, omega); //apply \Gamma_0^-1 ( gamma_n - n_e)
        gamma1.alpha() = -0.5*tau;
    }
    number = pcg( pol, phi[0], omega, v2d, eps_pol);
    if( number == pcg.get_max())
        throw Fail( eps_pol);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for phi \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_DEBUG

    return phi[0];
}

template< class M, class container>
void ToeflR<M, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    //update energetics, 2% of total time
    if( global)
    {
        exp( y, expy);
        mass_ = blas2::dot( one, w2d, expy[0] ); //take real ion density which is electron density!!
        double Ue = blas2::dot( y[0], w2d, expy[0]);
        double Ui = tau*blas2::dot( y[1], w2d, expy[1]);
        double Uphi = 0.5*blas2::dot( expy[1], w2d, omega); 
        energy_ = Ue + Ui + Uphi;

        for( unsigned i=0; i<y.size(); i++)
        {
            dg::blas1::transform( expy[i], expy[i], dg::PLUS<double>(-1));
            blas2::gemv( laplaceM, expy[i], lapy[i]); //Laplace wants Dir BC!!
        }
        diff_ = -nu*blas2::dot( one, w2d, lapy[0]);
        double Ge = - blas2::dot( one, w2d, lapy[0]) - blas2::dot( lapy[0], w2d, y[0]); // minus 
        double Gi = - tau*(blas2::dot( one, w2d, lapy[1]) + blas2::dot( lapy[1], w2d, y[1])); // minus 
        double Gphi = -blas2::dot( phi[0], w2d, lapy[0]);
        double Gpsi = -blas2::dot( phi[1], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge + Gi - Gphi + Gpsi);
    }

    for( unsigned i=0; i<y.size(); i++)
    {
        arakawa( y[i], phi[i], yp[i]);
        blas1::pointwiseDot( binv, yp[i], yp[i]);
    }

    //compute derivatives
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( arakawa.dx(), y[i], dxy[i]);
        blas2::gemv( arakawa.dy(), y[i], dyy[i]);
        blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);
    }
    // curvature terms
    blas1::axpby( kappa, dyphi[0], 1., yp[0]);
    blas1::axpby( kappa, dyphi[1], 1., yp[1]);
    blas1::axpby( -1.*kappa, dyy[0], 1., yp[0]);
    blas1::axpby( tau*kappa, dyy[1], 1., yp[1]);


}

}//namespace dg

#endif //_DG_TOEFLR_CUH
