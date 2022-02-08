#pragma once
#include <exception>

#include "dg/algorithm.h"
#include "parameters.h"

namespace toefl
{

template< class Geometry,  class Matrix, class container >
struct Explicit
{
    Explicit( const Geometry& g, const Parameters& p );

    const std::array<container,2>& potential( ) const { return phi;}

    dg::Elliptic<Geometry, Matrix, container>& laplacianM( ) { return laplaceM;}

    dg::Helmholtz<Geometry, Matrix, container >&  gamma() {return multi_gamma1[0];}

    void operator()( double t, const std::array<container,2>& y, std::array<container,2>& yp);

  private:
    //use chi and omega as helpers to compute square velocity in omega
    const container& compute_psi( double t, const container& potential);
    const container& polarisation( double t, const std::array<container,2>& y);

    container chi, omega;
    const container m_binv; //magnetic field

    std::array<container,2> phi, dyphi, ype;
    std::array<container,2> dyy, lny, lapy;
    container gamma_n;

    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, container> pol, laplaceM; //contains normalized laplacian
    std::vector<dg::Elliptic<Geometry, Matrix, container> > multi_pol;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > multi_gamma1;
    dg::ArakawaX< Geometry, Matrix, container> arakawa;

    dg::MultigridCG2d<Geometry, Matrix, container> multigrid;
    dg::Extrapolation<container> old_phi, old_psi, old_gammaN;
    std::vector<container> multi_chi;

    const container w2d, one;
    Parameters m_p;

};

template< class Geometry, class M, class container>
Explicit< Geometry, M, container>::Explicit( const Geometry& grid, const Parameters& p ):
    chi( evaluate( dg::zero, grid)), omega(chi),
    m_binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)),
    phi( 2, chi), dyphi( phi), ype(phi),
    dyy(2,chi), lny( dyy), lapy(dyy),
    gamma_n(chi),
    pol(     grid,  dg::centered, p.jfactor),
    laplaceM( grid,  dg::centered),
    arakawa( grid),
    multigrid( grid, 3),
    old_phi( 2, chi), old_psi( 2, chi), old_gammaN( 2, chi),
{
    multi_chi= multigrid.project( chi);
    multi_pol.resize(3);
    multi_gamma1.resize(3);
    for( unsigned u=0; u<3; u++)
    {
        multi_pol[u].construct( multigrid.grid(u),  dg::centered, p.jfactor);
        multi_gamma1[u].construct( multigrid.grid(u), -0.5*p.tau, dg::centered);
    }
}

template< class G, class M, class container>
const container& Explicit<G, M, container>::compute_psi( double t, const container& potential)
{
    if(m_p.model == "gravity_local") return potential;
    //in gyrofluid invert Gamma operator
    if( m_p.model == "local" || m_p.model == "global")
    {
        if (tau == 0.) {
            dg::blas1::axpby( 1.,potential, 0.,phi[1]); //chi = N_i - 1
        }
        else {
            old_psi.extrapolate( t, phi[1]);
            std::vector<unsigned> number = multigrid.solve( multi_gamma1, phi[1], potential, eps_gamma);
            old_psi.update( t, phi[1]);
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( eps_gamma);
        }
    }
    //compute (nabla phi)^2
    pol.variation(potential, omega);
    //compute psi
    if(m_p.model == "global")
    {
        dg::blas1::pointwiseDot( -0.5, m_binv, m_binv, omega, 1., phi[1]);
    }
    else if ( m_p.model == "drift_global")
    {
        dg::blas1::pointwiseDot( 0.5, m_binv, m_binv, omega, 0., phi[1]);
    }
    else if( m_p.model == "gravity_global" )
        dg::blas1::axpby( 0.5, omega, 0., phi[1]);
    return phi[1];
}


//computes and modifies expy!!
template<class G, class M, class container>
const container& Explicit<G, M, container>::polarisation( double t, const std::array<container,2>& y)
{
    //compute chi
    if(m_p.model == "global" )
    {
        dg::assign( y[1], chi);
        dg::blas1::plus( chi, 1.);
        dg::blas1::pointwiseDot( m_binv, chi, chi); //\chi = n_i
        dg::blas1::pointwiseDot( m_binv, chi, chi); //\chi *= m_binv^2
        if( !boussinesq)
        {
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
                multi_pol[u].set_chi( multi_chi[u]);
            //pol.set_chi( chi);
        }
    }
    else if(m_p.model == "gravity_global" )
    {
        dg::assign( y[0], chi);
        dg::blas1::plus( chi, 1.);
        if( !boussinesq)
        {
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
                multi_pol[u].set_chi( multi_chi[u]);
            //pol.set_chi( chi);
        }
    }
    else if( m_p.model == "drift_global" )
    {
        dg::assign( y[0], chi);
        dg::blas1::plus( chi, 1.);
        dg::blas1::pointwiseDot( m_binv, chi, chi); //\chi = n_e
        dg::blas1::pointwiseDot( m_binv, chi, chi); //\chi *= m_binv^2
        if( !boussinesq)
        {
            multigrid.project( chi, multi_chi);
            for( unsigned u=0; u<3; u++)
                multi_pol[u].set_chi( multi_chi[u]);
            //pol.set_chi( chi);
        }
    }
    //compute polarisation
    if( m_p.model == "local" || m_p.model == "global")
    {
        if (tau == 0.) {
            dg::blas1::axpby( 1., y[1], 0.,gamma_n); //chi = N_i - 1
        }
        else {
            old_gammaN.extrapolate(t, gamma_n);
            std::vector<unsigned> number = multigrid.solve( multi_gamma1, gamma_n, y[1], eps_gamma);
            old_gammaN.update(t, gamma_n);
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( eps_gamma);
        }
        dg::blas1::axpby( -1., y[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
    }
    else
        dg::blas1::axpby( -1. ,y[1], 0., omega);
    if( m_p.model == "global" || m_p.model == "gravity_global" || m_p.model == "drift_global")
        if( boussinesq)
            dg::blas1::pointwiseDivide( omega, chi, omega);
    //invert

    old_phi.extrapolate(t, phi[0]);
    std::vector<unsigned> number = multigrid.solve( multi_pol, phi[0], omega, eps_pol);
    old_phi.update( t, phi[0]);
    if(  number[0] == multigrid.max_iter())
        throw dg::Fail( eps_pol);
    return phi[0];
}

template< class G, class M, class container>
void Explicit<G, M, container>::operator()( double t, const std::array<container,2>& y, std::array<container,2>& yp)
{
    //y[0] = N_e - 1
    //y[1] = N_i - 1 || y[1] = Omega

    phi[0] = polarisation( t, y);
    phi[1] = compute_psi( t, phi[0]);

    for( unsigned i=0; i<y.size(); i++)
    {
        dg::blas1::transform( y[i], m_ype[i], dg::PLUS<double>(1.));
        dg::blas2::symv( -1., m_laplaceM, y[i], 0., m_lapy[i]);
    }

    /////////////////////////update energetics, 2% of total time///////////////
    mass_ = dg::blas2::dot( one, w2d, y[0] ); //take real ion density which is electron density!!
    diff_ = nu*dg::blas2::dot( one, w2d, lapy[0]);
    if(m_p.model == "global")
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
    else if ( m_p.model == "drift_global")
    {
        double Se = dg::blas2::dot( lny[0], w2d, ype[0]);
        double Ephi = dg::blas2::dot( ype[0], w2d, phi[1]); //phi[1] equals 0.5*u_E^2
        energy_ = Se + Ephi;

        double Ge = - dg::blas2::dot( one, w2d, lapy[0]) - dg::blas2::dot( lapy[0], w2d, lny[0]); // minus
        double GeE = - dg::blas2::dot( phi[1], w2d, lapy[0]);
        double Gpsi = -dg::blas2::dot( phi[0], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge - GeE - Gpsi);
    }
    else if(m_p.model == "gravity_global" || m_p.model == "gravity_local")
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
    if( m_p.model == "gravity_global")
    {
        arakawa(y[0], phi[0], yp[0]);
        arakawa(y[1], phi[0], yp[1]);
        arakawa(y[0], phi[1], omega);
        dg::blas1::axpbypgz( 1., omega, -friction, y[1], 1., yp[1]);
        dg::blas2::gemv( 1., arakawa.dy(), y[0], 1., yp[1]);
        return;
    }
    else if( m_p.model == "gravity_local")
    {
        arakawa(y[0], phi[0], yp[0]);
        arakawa(y[1], phi[0], yp[1]);
        dg::blas2::gemv( arakawa.dy(), y[0], dyy[0]);
        dg::blas1::axpbypgz( -friction, y[1], -1., dyy[0], 1., yp[1]);
        return;
    }
    else if( m_p.model == "drift_global")
    {
        arakawa(y[0], phi[0], yp[0]);
        arakawa(y[1], phi[0], yp[1]);
        arakawa(y[0], phi[1], omega);
        dg::blas1::pointwiseDot( m_binv, yp[0], yp[0]);
        dg::blas1::pointwiseDot( m_binv, yp[1], yp[1]);
        dg::blas1::pointwiseDot( 1., m_binv, omega, 1., yp[1]);

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
        for( unsigned i=0; i<2; i++)
        {
            arakawa( y[i], phi[i], yp[i]);
            if(m_p.model == "global")
                dg::blas1::pointwiseDot( m_binv, yp[i], yp[i]);
        }
        double _tau[2] = {-1., tau};
        //compute derivatives and exb compression
        for( unsigned i=0; i<2; i++)
        {
            dg::blas2::gemv( arakawa.dy(), y[i], dyy[i]);
            dg::blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);
            if(m_p.model == "global")
                dg::blas1::pointwiseDot( dyphi[i], ype[i], dyphi[i]);
            dg::blas1::axpbypgz( kappa, dyphi[i], _tau[i]*kappa, dyy[i], 1., yp[i]);
        }
    }

    dg::blas1::axpby( nu, m_lapy, 1., yp);
    return;
}

}//namespace dg
