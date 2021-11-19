#pragma once

#include "dg/algorithm.h"
#include "parameters.h"

namespace dg
{
template<class Geometry, class Matrix, class container>
struct Diffusion
{
    Diffusion( const Geometry& g, imp::Parameters p):
        p(p),
        temp( dg::evaluate(dg::zero, g)),
        LaplacianM_perp ( g, g.bcx(), g.bcy(),  dg::centered)
    {
    }
    void operator()(double t, const std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
           x[1] := N_i - 1
           x[2] := N_j - 1

        */
        dg::blas1::axpby( 0., x, 0., y);
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
    const imp::Parameters p;
    container temp;
    dg::Elliptic<Geometry, Matrix, container> LaplacianM_perp;
};

template< class Geometry, class Matrix, class container >
struct ToeflI
{
    using value_type = dg::get_value_type<container>;

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
    void operator()(double t, const std::vector<container>& y, std::vector<container>& yp);

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
    const container& polarization( double t, const std::vector<container>& y);

private:
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    const container& compute_psi( const container& potential, int idx);
//    const container& polarization( const std::vector<container>& y);

    container chi, omega;
    const container binv; //magnetic field

    std::vector<container> phi, dyphi, ype;
    std::vector<container> dyy, lny, lapy;
    std::vector<container> gamma_n;

    //matrices and solvers
    Helmholtz< Geometry, Matrix, container > gamma1;
    ArakawaX< Geometry, Matrix, container> arakawa; 
    dg::Elliptic< Geometry, Matrix, container > pol, laplaceM; 
    dg::PCG<container> invert_pol, invert_invgamma;
    dg::Extrapolation<container> extra_pol, extra_invgamma;

    const container w2d;

    double mass_, energy_, diff_, ediff_;

    imp::Parameters p;
};

template< class Geometry, class Matrix, class container>
ToeflI< Geometry, Matrix, container>::ToeflI( const Geometry& grid, imp::Parameters p) :
    chi( evaluate( dg::zero, grid )), omega(chi),
    binv( evaluate( LinearX( p.kappa, 1.), grid)),
    phi( 3, chi), dyphi( phi), ype(phi),
    dyy( 3, chi), lny(dyy), lapy( dyy),
    gamma_n( 2, chi),
    gamma1(  grid, -0.5*p.tau[1]),
    arakawa( grid),
    pol(      grid, centered),
    laplaceM( grid, centered),
    invert_pol(      omega, omega.size()),
    invert_invgamma( omega, omega.size()),
    extra_pol( 2, omega),
    w2d( create::volume(grid)), p(p)
    {
    }


//idx is impurity species one or two
template< class G, class M, class container>
const container& ToeflI<G, M, container>::compute_psi( const container& potential, int idx)
{
    gamma1.alpha() = -0.5*p.tau[idx]*p.mu[idx];

    invert_invgamma.solve( gamma1, phi[idx], potential, gamma1.precond(),
            gamma1.weights(), p.eps_gamma);

    pol.variation(binv,potential, omega); // u_E^2

    dg::blas1::axpby( 1., phi[idx], -0.5*p.mu[idx], omega, phi[idx]);   //psi  Gamma phi - 0.5 u_E^2
    return phi[idx];
}


template<class G, class Matrix, class container>
const container& ToeflI<G, Matrix, container>::polarization( double t, const std::vector<container>& y)
{
    //\chi = p.ai \p.mui n_i + p.as \p.mus n_s
    blas1::axpby( p.a[1]*p.mu[1], y[1], 0., chi); 
    blas1::axpby( p.a[2]*p.mu[2], y[2], 1., chi);
    dg::blas1::plus( chi, p.a[1]*p.mu[1]+p.a[2]*p.mu[2]); 
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\p.mui n_i ) /B^2
    pol.set_chi( chi);                              //set chi of polarisation: nablp.aperp (chi nablp.aperp )

    gamma1.alpha() = -0.5*p.tau[1]*p.mu[1];
    invert_invgamma.solve( gamma1, gamma_n[0], y[1], gamma1.precond(),
            gamma1.weights(), p.eps_gamma);
    gamma1.alpha() = -0.5*p.tau[2]*p.mu[2];
    invert_invgamma.solve( gamma1, gamma_n[1], y[2], gamma1.precond(),
            gamma1.weights(), p.eps_gamma );

    dg::blas1::axpby( -1., y[0], 0., chi);
    dg::blas1::axpby( +p.a[1], gamma_n[0], 1., chi);
    dg::blas1::axpby( +p.a[2], gamma_n[1], 1., chi);

    extra_pol.extrapolate( t, phi[0]);
    unsigned number = invert_pol.solve( pol, phi[0], chi, pol.precond(),
            pol.weights(), p.eps_pol);
    //p.ajGamma n_j + p.aiGamma n_i -ne = -nabla chi nabla phi
    if(  number == invert_pol.get_max())
        throw dg::Fail( p.eps_pol);
    extra_pol.update( t, phi[0]);
    return phi[0];
}

template< class G, class M, class container>
void ToeflI< G, M, container>::operator()(double t, const std::vector<container>& y, std::vector<container>& yp)
{
    //y[0] = N_e - 1
    //y[1] = N_i - 1
    //y[2] = N_j - 1
    //
    assert( y.size() == 3);
    assert( y.size() == yp.size());

    phi[0] = polarization(t, y);
    phi[1] = compute_psi( phi[0], 1);
    phi[2] = compute_psi( phi[0], 2);
    for( unsigned i=0; i<y.size(); i++)
    {
        dg::blas1::transform( y[i], ype[i], dg::PLUS<>(+1)); 
        dg::blas1::transform( ype[i], lny[i], dg::LN<double>()); 
        dg::blas2::symv( laplaceM, y[i], chi);
        dg::blas2::symv( laplaceM, chi, lapy[i]);
    }

    //update energetics, 2% of total time
        mass_ = blas2::dot( 1., w2d, y[0] ); //take real ion density which is electron density!!
        double Se = blas2::dot( ype[0], w2d, lny[0]);
        double Si = p.a[1]*p.tau[1]*blas2::dot( ype[1], w2d, lny[1]);
        double Sz = p.a[2]*p.tau[2]*blas2::dot( ype[2], w2d, lny[2]);
        double Uphii = 0.5*p.a[1]*p.mu[1]*blas2::dot( ype[1], w2d, omega); 
        double Uphiz = 0.5*p.a[2]*p.mu[2]*blas2::dot( ype[2], w2d, omega); 
        energy_ = Se + Si + Sz + Uphii + Uphiz;

        diff_ = -p.nu*blas2::dot( 1., w2d, lapy[0]);
        double Gi[3];
        Gi[0] = - blas2::dot( 1., w2d, lapy[0]) - blas2::dot( lapy[0], w2d, lny[0]); // minus 
        for( unsigned i=1; i<3; i++)
            Gi[i] = - p.tau[i]*(blas2::dot( 1., w2d, lapy[i]) + blas2::dot( lapy[i], w2d, lny[i])); // minus 
        double Gphi[3];
        for( unsigned i=0; i<3; i++)
            Gphi[i] = -blas2::dot( phi[i], w2d, lapy[i]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = p.nu*( Gi[0] - Gphi[0] + p.a[1]*(Gi[1] + Gphi[1]) + p.a[2]*( Gi[2] + Gphi[2]));

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
        blas1::axpby( p.kappa, dyphi[i], 1., yp[i]);

        blas1::axpby( p.tau[i]*p.kappa, dyy[i], 1., yp[i]);
    }
}

}//namespace dg

