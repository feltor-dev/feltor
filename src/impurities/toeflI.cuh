#pragma once

#include "dg/algorithm.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif


//TODO es wäre besser, wenn ToeflI auch einen Zeitschritt berechnen würde 
// dann wäre die Rückgabe der Felder (Potential vs. Masse vs. exp( y)) konsistenter
// (nur das Objekt weiß welches Feld zu welchem Zeitschritt gehört)

namespace dg
{

template<class Geometry, class Matrix, class container>
struct Diffusion
{
    Diffusion( const Geometry& g, imp::Parameters p):
        p(p),
        temp( dg::evaluate(dg::zero, g)),
        LaplacianM_perp ( g,g.bcx(),g.bcy(), dg::normed, dg::centered)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
           x[1] := N_i - 1
           x[2] := N_j - 1

        */
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<x.size(); i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu);  //  nu_perp lapl_RZ (lapl_RZ N) 
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

template< class Geometry, class Matrix, class container >
struct ToeflI
{
    typedef typename container::value_type value_type;

    /**
     * @brief Construct a ToeflI solver object
     *
     * @param g The grid on which to operate
     * @param p the parameters
     */
    ToeflI( const Geometry& g, imp::Parameters p);


    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}

    /**
     * @brief Return the Gamma operator used by this object
     *
     * @return Gamma operator
     */
    Helmholtz<Geometry, Matrix, container>& gamma() {return gamma1;}

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
     */
    double mass( ) {return mass_;}
    /**
     * @brief Return the last integrated mass diffusion of operator() in a global computation
     *
     * @return int \nu \Delta (exp(y[0])-1)
     */
    double mass_diffusion( ) {return diff_;}
    /**
     * @brief Return the energy of the last field in operator() in a global computation
     *
     * @return integrated total energy in {ne, ni}
     */
    double energy( ) {return energy_;}
    /**
     * @brief Return the integrated energy diffusion of the last field in operator() in a global computation
     *
     * @return integrated total energy diffusion
     */
    double energy_diffusion( ){ return ediff_;}

  private:
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    const container& compute_psi( container& potential, int idx);
    const container& polarization( const std::vector<container>& y);

    container chi, omega;
    const container binv; //magnetic field

    std::vector<container> phi, dyphi;
    std::vector<container> dyy, lapy;
    std::vector<container> gammp.an;

    //matrices and solvers
    Helmholtz< Geometry, Matrix, container > gamma1;
    ArakawaX< Geometry, Matrix, container> arakawa; 
    dg::Elliptic< Geometry, Matrix, container > pol; 
    dg::Invert<container> invert_pol, invert_invgamma;

    const container w2d, v2d, one;

    double mass_, energy_, diff_, ediff_;

    imp::Parameters p;


};

template< class Geometry, class Matrix, class container>
ToeflI< Geometry, Matrix, container>::ToeflI( const Grid2d<value_type>& grid, imp::Parameters p) :
    chi( grid.size(), 0.), omega(chi),  
    binv( evaluate( LinearX( kappa, 1.), grid)), 
    phi( 3, chi), dyphi( phi),
    gammp.an( 2, chi),
    dyy( 3, chi), lapy( dyy),
    gamma1(  grid, -0.5*p.taui),
    arakawa( grid), 
    pol(     grid, not_normed, centered), 
    invert_pol(      omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    w2d( create::weights(grid)), v2d( create::inv_weights(grid)), one( dg::evaluate(dg::one, grid)), p(p)
    { 
    }


//idx is impurity species one or two
template< class container>
const container& ToeflI<container>::compute_psi( container& potential, int idx)
{
    gamma1.alpha() = -0.5*p.tau[idx]*p.mu[idx];
    invert_invgamma( gamma1, phi[idx], potential);

    arakawa.variation(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);

    dg::blas1::axpby( 1., chi, -0.5*p.mu[idx], omega, phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[idx];    
}


template<class G, class Matrix, class container>
container& Feltor<G, Matrix, container>::polarisation( const std::vector<container>& y)
{ 
    //\chi = p.ai \p.mui n_i + p.as \p.mus n_s
    blas1::axpby( p.a[1]*p.mu[1], y[1], 0., chi); 
    blas1::axpby( p.a[2]*p.mu[2], y[2], 1., chi);
    dg::blas1::plus( chi, ( p.a[1]*p.mu[1]+p.a[2]*p.mu[2]); 
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\p.mui n_i ) /B^2
    pol.set_chi( chi);                              //set chi of polarisation: nablp.aperp (chi nablp.aperp )




    gamma1.alpha() = -0.5*p.tau[1]*p.mu[1];
    invert_invgamma( gamma1, gammp.an[0], y[1]);
    gamma1.alpha() = -0.5*p.tau[2]*p.mu[2];
    invert_invgamma( gamma1, gammp.an[1], y[2]);

    dg::blas1::axpby( 1., y[0], 0., chi);
    dg::blas1::axpby( -p.a[1], gammp.an[0], 1., chi);
    dg::blas1::axpby( -p.a[2], gammp.an[1], 1., chi);

    unsigned number = invert_pol( pol, phi[0], chi);//p.ajGamma n_j + p.aiGamma n_i -ne = -nabla chi nabla phi
    if(  number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    return phi[0];
}

template< class container>
void ToeflI< container>::operator()(std::vector<container>& y, std::vector<container>& yp)
{
    //y[0] = N_e - 1
    //y[1] = N_i - 1
    //y[2] = N_j - 1
    //
    assert( y.size() == 3);
    assert( y.size() == yp.size());

    phi[0] = polarization( y);
    phi[1] = compute_psi( phi[0], 1);
    phi[2] = compute_psi( phi[0], 2);
    dg::blas1::transform( y[i], ype[i], dg::PLUS<>(+1)); 

    //update energetics, 2% of total time
    /*
        exp( y, expy);
        mass_ = blas2::dot( one, w2d, expy[0] ); //take real ion density which is electron density!!
        double Ue = blas2::dot( y[0], w2d, expy[0]);
        double Ui = p.a[1]*p.tau[1]*blas2::dot( y[1], w2d, expy[1]);
        double Uz = p.a[2]*p.tau[2]*blas2::dot( y[2], w2d, expy[2]);
        double Uphii = 0.5*p.a[1]*p.mu[1]*blas2::dot( expy[1], w2d, omega); 
        double Uphiz = 0.5*p.a[2]*p.mu[2]*blas2::dot( expy[2], w2d, omega); 
        energy_ = Ue + Ui + Uphii + Uphiz;

        for( unsigned i=0; i<y.size(); i++)
        {
            thrust::transform( expy[i].begin(), expy[i].end(), expy[i].begin(), dg::PLUS<double>(-1));
            blas2::gemv( laplaceM, expy[i], lapy[i]); //Laplace wants Dir BC!!
        }
        diff_ = -nu*blas2::dot( one, w2d, lapy[0]);
        double Gi[3];
        Gi[0] = - blas2::dot( one, w2d, lapy[0]) - blas2::dot( lapy[0], w2d, y[0]); // minus 
        for( unsigned i=1; i<3; i++)
            Gi[i] = - p.tau[i]*(blas2::dot( one, w2d, lapy[i]) + blas2::dot( lapy[i], w2d, y[i])); // minus 
        double Gphi[3];
        for( unsigned i=0; i<3; i++)
            Gphi[i] = -blas2::dot( phi[i], w2d, lapy[i]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Gi[0] - Gphi[0] + p.a[1]*(Gi[1] + Gphi[1]) + p.a[2]*( Gi[2] + Gphi[2]));
        */

    for( unsigned i=0; i<y.size(); i++)
    {
        arakawa( y[i], phi[i], yp[i]);
        blas1::pointwiseDot( binv, yp[i], yp[i]);
    }

    //compute derivatives
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( arakawa.dy(), y[i], dyy[i]);
        blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);

        blas1::pointwiseDot( dyphi[i], ype[i], dyphi[i]);
        blas1::axpby( kappa, dyphi[i], 1., yp[i]);

        blas1::axpby( p.tau[i]*kappa, dyy[i], 1., yp[i]);
    }
}

}//namespace dg

