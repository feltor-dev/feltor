#pragma once
#include <exception>
#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/polarization.h"

namespace poet
{


template<class Geometry, class Matrix, class container>
struct Implicit
{
    Implicit( const Geometry& g, double nu):
        nu_(nu),     
        temp( dg::evaluate(dg::zero, g)),
        LaplacianM_perp( g, dg::normed, dg::centered){ }
    void operator()(double t, const std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - 1
         * x[2] := N_i - 1 
         */
        for( unsigned i=0; i<x.size(); i++)
        {
//             dg::blas2::gemv( LaplacianM_perp, x[i], temp);
//             dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
//             dg::blas1::scal( y[i], -nu_);
            dg::blas2::gemv( -nu_, LaplacianM_perp, x[i], 0., y[i]);
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

template< class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class container >
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
     * @brief compute \f$ yp = \Gamma_1 y \f$ (or \f$ yp = \sqrt{\Gamma_1} y \f$ )
     *
     * @return yp
     */
    void gamma1_y( const container& y, container& yp)
    {
        if (equations == "ff-O2") {
//             dg::blas1::scal(yp, 0.0);
//             sqrtinvert(yp, y); //TODO produces wrong solution - origin of bug?
            //via two step approach
            sqrtsolve(y, m_iota);
            std::vector<unsigned> number = multigrid.direct_solve( m_multi_g0, yp, m_iota, eps_gamma);
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( eps_gamma);
        }
        else {
            std::vector<unsigned> number = multigrid.direct_solve( m_multi_g1, yp, y, eps_gamma);
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( eps_gamma);
        }
    }
    /**
     * @brief compute \f$  \Gamma_1 yp = y \f$ (or \f$ \sqrt{\Gamma_1} yp =  y \f$ )
     *
     * @return yp
     */
    void gamma1inv_y( const container& y, container& yp)
    {
        if (equations == "ff-O2")
        {
            sqrtsolve(y, yp);
        }
        else
        {
            dg::blas2::symv( m_multi_g1[0], y, m_chi); //invG ne-1
            dg::blas2::symv( m_v2d, m_chi, yp);
        }
    }
    /**
     * @brief Invert \f$ -\nabla \cdot (1/B \nabla_\perp yp) = y \f$ where y equals the ExB vorticity
     *
     * @return yp
     */  
    void invLap_y( const container& y, container& yp)
    {
        multigrid.project( m_binv, m_multi_chi);
        for( unsigned u=0; u<3; u++) 
            m_multi_elliptic[u].set_chi( m_multi_chi[u]);
        
        std::vector<unsigned> number = multigrid.direct_solve( m_multi_elliptic, yp, y, eps_pol);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_pol[0]);
    }
    /**
     * @brief solve the LWL approximated polarization equation for Ni with ne and potential given for the FF models and exactly for the df models
     *
     * @param ne fluctuating electron density
     * @param potential electric potential
     * @param Ni fluctuating ion gy-density
     */  
    void solve_Ni_lwl(const container& ne, const container& potential, container& Ni)
    {
        //ff-Pol term
        if( equations == "ff-lwl" || equations == "ff-O2" || equations == "ff-O4")
        {
            dg::assign( ne, m_chi);
            dg::blas1::plus( m_chi, 1.);
            dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi = n_e
            dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi *= m_binv^2
            m_multi_elliptic[0].set_chi(m_chi);
        }
      
        //Compute elliptic/tensor elliptic term on phi
        dg::blas2::symv( m_multi_elliptic[0], potential, m_chi);
        dg::blas2::symv( m_v2d, m_chi, m_iota);
        
        //apply G0 for df-02
        if( equations == "df-O2")
        {
            std::vector<unsigned> number = multigrid.direct_solve( m_multi_g0, Ni, m_iota, eps_gamma);
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( eps_gamma);
        }
        else dg::blas1::copy(m_iota, Ni);
             
        
        if( equations == "df-O2" || equations == "df-lwl")
        {
            dg::blas1::axpby(1.0, ne, 1.0, Ni, m_iota);
            dg::blas2::symv( m_multi_g1[0], m_iota, m_chi); //invG ne-1
            dg::blas2::symv( m_v2d, m_chi, Ni);
        }
        else
        {
            dg::blas2::symv( m_multi_g1[0], ne, m_chi); //invG ne-1
            dg::blas2::symv( m_v2d, m_chi, m_iota);
            dg::blas1::axpby( 1.0, m_iota, 1.0, Ni);

        }
    }
    /**
     * @brief Compute the right-hand side of the poet equations
     *
     * y[0] = N_e - 1, 
     * y[1] = N_i - 1 
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( double t, const std::vector<container>& y, std::vector<container>& yp);

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
    //use chi and m_omega as helpers to compute square velocity in m_omega
    const container& compute_psi( double t, const container& potential);
    const container& polarisation( double t, const std::vector<container>& y);

    container m_chi, m_omega, m_iota, m_gamma_n, m_g_phi;
    const container m_binv; //magnetic field
    container m_n_old, m_gamma_n_old, m_phi_old, m_gamma_m_phi_old;
    std::vector<container> phi, dyphi, ype;
    std::vector<container> dyy, lny, lapy;
    
    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, container>  laplaceM; //contains normalized laplacian
    std::vector<dg::Elliptic<Geometry, Matrix, container> > m_multi_elliptic;
    std::vector<dg::TensorElliptic<Geometry, Matrix, container> > m_multi_tensorelliptic;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > m_multi_g1, m_multi_g0;
    
    dg::KrylovSqrtCauchySolve< Geometry, Matrix, DiaMatrix, CooMatrix, container, dg::DVec> sqrtsolve;
    dg::KrylovSqrtCauchyinvert<Geometry, Matrix, DiaMatrix, CooMatrix, container, dg::DVec> sqrtinvert;
    
    dg::Advection<Geometry, Matrix, container> m_adv;
    
    dg::MultigridCG2d<Geometry, Matrix, container> multigrid;
    dg::Extrapolation<container> m_phi_ex, m_gamma_phi_ex, m_psi_ex, m_gamma_n_ex;
    std::vector<container> m_multi_chi, m_multi_iota;
    
    Matrix m_centered[2];
 
    
    const container w2d,m_v2d, one;
    const std::vector<double> eps_pol;
    const double eps_gamma;
    const double kappa,  nu, tau;
    const std::string equations;

    double mass_, energy_, diff_, ediff_;


};

template< class Geometry, class M, class DM, class CM, class container>
Explicit< Geometry, M, DM, CM, container>::Explicit( const Geometry& grid, const Parameters& p ):
    m_chi( evaluate( dg::zero, grid)), m_omega(m_chi), m_iota(m_chi), m_gamma_n(m_chi), m_g_phi(m_chi),
    m_binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)),
    m_n_old(m_chi),    m_gamma_n_old(m_chi), m_phi_old(m_chi),    m_gamma_m_phi_old(m_chi),
    phi( 2, m_chi), dyphi( phi), ype(phi),
    dyy(2,m_chi), lny( dyy), lapy(dyy), 
    laplaceM( grid, dg::normed, dg::centered),
    multigrid( grid, 3),
    m_phi_ex( 2, m_chi),  m_gamma_phi_ex(2, m_chi), m_psi_ex( 2, m_chi), m_gamma_n_ex( 2, m_chi),
    w2d( dg::create::volume(grid)), m_v2d( dg::create::inv_weights(grid)), one( dg::evaluate(dg::one, grid)),
    eps_pol(p.eps_pol), eps_gamma( p.eps_gamma), kappa(p.kappa), nu(p.nu), tau( p.tau), equations( p.equations)
{
    m_multi_chi= multigrid.project( m_chi);
    m_multi_iota= multigrid.project( m_chi);
    m_multi_elliptic.resize(3);
    m_multi_tensorelliptic.resize(3);
    m_multi_g1.resize(3);
    m_multi_g0.resize(3);
    m_adv.construct(grid);
    m_centered[0] = dg::create::dx( grid, grid.bcx(), dg::centered);
    m_centered[1] = dg::create::dy( grid, grid.bcy(), dg::centered);
    for( unsigned u=0; u<3; u++)
    {
        m_multi_elliptic[u].construct( multigrid.grid(u), dg::not_normed, dg::centered, p.jfactor);
        m_multi_tensorelliptic[u].construct( multigrid.grid(u),  dg::not_normed, dg::centered, p.jfactor); //only centered implemented

        
        m_multi_g0[u].construct( multigrid.grid(u), -p.tau, dg::centered, p.jfactor);
        if( equations == "ff-O2" ) {
            m_multi_g1[u].construct( multigrid.grid(u), -p.tau, dg::centered, p.jfactor);
            sqrtsolve.construct( m_multi_g1[0], grid, m_chi,  p.eps_time, p.maxiter_sqrt, p.maxiter_cauchy,  p.eps_gamma);
            sqrtinvert.construct(m_multi_g1[0], grid, m_chi,  p.eps_time, p.maxiter_sqrt, p.maxiter_cauchy,  p.eps_gamma);
        }
        else {
            m_multi_g1[u].construct( multigrid.grid(u), -0.5*p.tau, dg::centered, p.jfactor);
        }               
    }

}

template< class G,  class M, class DM, class CM, class container>
const container& Explicit<G,  M, DM, CM, container>::compute_psi( double t, const container& potential)
{
    if( equations == "ff-O4" )
    {     
        dg::blas1::pointwiseDot(m_binv, m_binv, m_chi);
        m_multi_tensorelliptic[0].variation( m_g_phi, tau/2., m_chi, m_omega);
        dg::blas1::axpby( 1.,  m_g_phi, 1.0, m_omega,  phi[1]);
    }
    else if ( equations == "ff-O2") {
        //elliptic part
        m_multi_elliptic[0].variation(m_g_phi, phi[1]);   
        dg::blas1::pointwiseDot(1.0, m_binv, m_binv, phi[1], 0.0, m_omega); 
        dg::blas1::axpby( 1.,  m_g_phi, -0.5, m_omega,  phi[1]); 
    }
    else {
        if (tau == 0.) {
            dg::blas1::axpby( 1., potential, 0.,phi[1]); 
        }
        else {
            m_psi_ex.extrapolate( t, phi[1]);
            std::vector<unsigned> number = multigrid.direct_solve( m_multi_g1, phi[1], potential, eps_gamma);
            m_psi_ex.update( t, phi[1]);
            if(  number[0] == multigrid.max_iter())
                throw dg::Fail( eps_gamma);
        }
        //compute (nabla phi)^2
        m_multi_elliptic[0].variation(potential, m_omega); //m_omega used for ExB energy derivation
        //compute psi
        if(equations == "ff-lwl")
        {

            dg::blas1::pointwiseDot( -0.5, m_binv, m_binv, m_omega, 1., phi[1]);
        }
    }
    return phi[1];
}

//computes and modifies expy!!
template<class G,  class M, class DM, class CM, class container>
const container& Explicit<G,  M, DM, CM, container>::polarisation( double t, const std::vector<container>& y)
{
    //Compute chi and m_iota for global models
    if( equations == "ff-lwl" || equations == "ff-O2" || equations == "ff-O4")
    {
        dg::assign( y[1], m_chi);
        dg::blas1::plus( m_chi, 1.);
        dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi = n_i
        dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi *= m_binv^2
        multigrid.project( m_chi, m_multi_chi);
        if( equations == "ff-O4" )
        {
            dg::blas1::pointwiseDot(tau/4., m_chi,m_binv,m_binv,0., m_chi);
            multigrid.project( m_chi, m_multi_iota);
            for( unsigned u=0; u<3; u++)
            {
                m_multi_tensorelliptic[u].set_chi( m_multi_chi[u]);
                m_multi_tensorelliptic[u].set_iota( m_multi_iota[u]);
            }
        }
        else
        {
            for( unsigned u=0; u<3; u++) 
                m_multi_elliptic[u].set_chi( m_multi_chi[u]);
        }
    }
    
    //Compute rho
    if( equations == "ff-O4" )
    {
        dg::blas2::symv( m_multi_g1[0], y[0], m_chi); //invG ne-1
        dg::blas2::symv( m_v2d, m_chi, m_gamma_n);
    }
    else if( equations == "ff-O2" )
    {
        dg::blas1::axpby(1.0, y[0], -1.0, m_n_old);
        sqrtsolve(m_n_old, m_gamma_n); 
        dg::blas1::axpby( 1.0, m_gamma_n_old, 1.0, m_gamma_n); 
        dg::blas1::copy(y[0], m_n_old);
        dg::blas1::copy(m_gamma_n, m_gamma_n_old); 
//         sqrtsolve(y[0], m_gamma_n); //without old solution
    }
    else 
    {  
        m_gamma_n_ex.extrapolate(t, m_gamma_n);
        std::vector<unsigned> number = multigrid.direct_solve( m_multi_g1, m_gamma_n, y[1], eps_gamma);
        m_gamma_n_ex.update(t, m_gamma_n);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_gamma);
    }

    if( equations == "ff-O2" || equations == "ff-O4" )
        dg::blas1::axpby(1., y[1],  -1., m_gamma_n, m_omega);    
    else
    {
        dg::blas1::axpby( -1., y[0], 1., m_gamma_n, m_omega); 
        if (equations == "df-O2") 
        {            
            dg::blas2::symv(m_multi_g0[0], m_omega, m_chi);
            dg::blas2::symv(m_v2d, m_chi, m_omega);
        }
    }

    //solve polarization equation for phi
    if(equations == "ff-O4" )
    {
        m_gamma_phi_ex.extrapolate(t, m_g_phi);
        std::vector<unsigned> number = multigrid.direct_solve( m_multi_tensorelliptic, m_g_phi, m_omega, eps_pol);
        m_gamma_phi_ex.update( t, m_g_phi);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_pol[0]);

        dg::blas2::symv(m_multi_g1[0], m_g_phi, phi[0]); 
        dg::blas1::pointwiseDot( m_v2d, phi[0], phi[0]);
    }
    else if( equations == "ff-O2" )
    {
        m_gamma_phi_ex.extrapolate(t, m_g_phi);
        std::vector<unsigned> number = multigrid.direct_solve( m_multi_elliptic, m_g_phi, m_omega, eps_pol);
        m_gamma_phi_ex.update( t, m_g_phi);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_pol[0]);

        dg::blas1::axpby(1.0, m_g_phi, -1.0, m_gamma_m_phi_old); //        
        sqrtsolve(m_gamma_m_phi_old, phi[0]); 
        dg::blas1::axpby( 1.0, m_phi_old, 1.0, phi[0]); //       
        dg::blas1::copy(m_g_phi, m_gamma_m_phi_old);//     
        dg::blas1::copy(phi[0], m_phi_old); 
//         sqrtsolve(m_g_phi, phi[0]);    //without old solution
    }
    else
    {
        m_phi_ex.extrapolate(t, phi[0]);
        std::vector<unsigned> number = multigrid.direct_solve( m_multi_elliptic, phi[0], m_omega, eps_pol);
        m_phi_ex.update( t, phi[0]);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_pol[0]);
    }

    return phi[0];
}

template< class G,  class M, class DM, class CM, class container>
void Explicit<G,  M, DM, CM, container>::operator()( double t, const std::vector<container>& y, std::vector<container>& yp)
{
    //y[0] = N_e - 1
    //y[1] = N_i - 1 
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi[0] = polarisation( t, y);
    phi[1] = compute_psi( t, phi[0]);

    for( unsigned i=0; i<y.size(); i++)
    {
        dg::blas1::transform( y[i], ype[i], dg::PLUS<double>(1.));
        dg::blas1::transform( ype[i], lny[i], dg::LN<double>());
        dg::blas2::symv( laplaceM, y[i], lapy[i]);
    }

    /////////////////////////update energetics, 2% of total time///////////////
    mass_ = dg::blas2::dot( one, w2d, y[0] ); //take real ion density which is electron density!!
    diff_ = nu*dg::blas2::dot( one, w2d, lapy[0]);
    if(equations == "ff-lwl" || equations == "ff-O4"  || equations == "ff-O2" )
    {
        double Ue = dg::blas2::dot( lny[0], w2d, ype[0]);
        double Ui = tau*dg::blas2::dot( lny[1], w2d, ype[1]);
        double Uphi = 0.5*dg::blas2::dot( ype[1], w2d, m_omega);
        energy_ = Ue + Ui + Uphi;

        double Ge = - dg::blas2::dot( one, w2d, lapy[0]) - dg::blas2::dot( lapy[0], w2d, lny[0]); // minus
        double Gi = - tau*(dg::blas2::dot( one, w2d, lapy[1]) + dg::blas2::dot( lapy[1], w2d, lny[1])); // minus
        double Gphi = -dg::blas2::dot( phi[0], w2d, lapy[0]);
        double Gpsi = -dg::blas2::dot( phi[1], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge + Gi - Gphi + Gpsi);
    }
    else
    {
        double Ue = 0.5*dg::blas2::dot( y[0], w2d, y[0]);
        double Ui = 0.5*tau*dg::blas2::dot( y[1], w2d, y[1]);
        double Uphi = 0.5*dg::blas2::dot( one, w2d, m_omega);
        energy_ = Ue + Ui + Uphi;

        double Ge = - dg::blas2::dot( y[0], w2d, lapy[0]); // minus
        double Gi = - tau*(dg::blas2::dot( y[1], w2d, lapy[1])); // minus
        double Gphi = -dg::blas2::dot( phi[0], w2d, lapy[0]);
        double Gpsi = -dg::blas2::dot( phi[1], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge + Gi - Gphi + Gpsi);
    }
    ///////////////////////////////////////////////////////////////////////
    double _tau[2] = {-1., tau};

    for( unsigned i=0; i<y.size(); i++) 
    {
        
        //  - v_x dx n
        dg::blas2::symv( -1., m_centered[1], phi[i], 0., m_chi); //v_x
        //  - v_y dy n
        dg::blas2::symv( 1., m_centered[0], phi[i], 0., m_iota); //v_y
        m_adv.upwind( -1., m_chi, m_iota, y[i], 0., yp[i]);  
        
        if(equations == "ff-lwl" || equations == "ff-O4" || equations == "ff-O2")
        {
            dg::blas1::pointwiseDot( m_binv, yp[i], yp[i]);
        }
        dg::blas2::symv( m_centered[1], y[i], dyy[i]);
        dg::blas2::symv( m_centered[1], phi[i], dyphi[i]);
        if(equations == "ff-lwl" || equations == "ff-O4"|| equations == "ff-O2") {
            dg::blas1::pointwiseDot( dyphi[i], ype[i], dyphi[i]);
        }
        dg::blas1::axpbypgz( kappa, dyphi[i], _tau[i]*kappa, dyy[i], 1., yp[i]);
            
    }
    return;
}

}//namespace dg
