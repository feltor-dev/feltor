#pragma once
#include <exception>
#include "dg/algorithm.h"
#include "dg/matrixsqrt.h"
#include "parameters.h"
namespace poet
{

template< class Geometry, class Matrix, class container >
struct Poet
{
    /**
     * @brief Construct a Poet solver object
     *
     * @param g The grid on which to operate
     * @param p The parameters
     */
    Poet( const Geometry& g, const Parameters& p );

    const container& potential( int i) const { return m_phi[i];}
    const container& density(   int i) const { return m_ype[i];}
    const Geometry& grid() const {return m_multigrid.grid(0);}
    void compute_lapM ( double alpha, const container& in, double beta, container& result)
    {
        dg::blas2::symv( alpha, m_lapMperp, in, beta, result);
    }
    void compute_diff( double alpha, const container& nme, double beta, container& result)
    {
        if( m_p.nu != 0)
        {
            dg::blas2::gemv( m_lapMperp, nme, m_iota);
            dg::blas2::gemv( -alpha*m_p.nu, m_lapMperp, m_iota, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
    }
    /**
     * @brief compute \f$ yp = \Gamma_1 y \f$ (or \f$ yp = \sqrt{\Gamma_1} y \f$ )
     *
     * @return yp
     */
    void gamma1_y( const container& y, container& yp)
    {
        if (m_p.equations == "ff-O2") {
//             via two step approach
            m_sqrtsolve(y, m_iota);
            std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_g0, yp, m_iota, m_p.eps_gamma);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma);
        }
        else {
            std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_g1, yp, y, m_p.eps_gamma);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma);
        }
    }
    /**
     * @brief compute \f$  \Gamma_1 yp = y \f$ (or \f$ \sqrt{\Gamma_1} yp =  y \f$ )
     *
     * @return yp
     */
    void gamma1inv_y( const container& y, container& yp)
    {
        if (m_p.equations == "ff-O2")
        {
            m_sqrtsolve(y, yp);
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
        m_multigrid.project( m_binv, m_multi_chi);
        for( unsigned u=0; u<3; u++) 
            m_multi_elliptic[u].set_chi( m_multi_chi[u]);
        
        std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_elliptic, yp, y, m_p.eps_pol);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);
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
        if( m_p.equations == "ff-lwl" || m_p.equations == "ff-O2" || m_p.equations == "ff-O4")
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
        if( m_p.equations == "df-O2")
        {
            std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_g0, Ni, m_iota, m_p.eps_gamma);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma);
        }
        else dg::blas1::copy(m_iota, Ni);
             
        
        if( m_p.equations == "df-O2" || m_p.equations == "df-lwl")
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
     * @brief Compute the right-hand side of the poet m_p.equations
     *
     * y[0] = N_e - 1, 
     * y[1] = N_i - 1 
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( double t, const std::array<container,2>& y, std::array<container,2>& yp);

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
    const container& polarisation( double t, const std::array<container,2>& y);

    container m_chi, m_omega, m_iota, m_gamma_n, m_gamma_phi;
    const container m_binv; //magnetic field
    container m_n_old, m_gamma_n_old, m_phi_old, m_gamma_phi_old;
    std::array<container,2> m_phi, m_dyphi, m_ype;
    std::array<container,2> m_dyy, m_lnype, m_lapy;
    
    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, container>  m_lapMperp; //contains normalized laplacian
    std::vector<dg::Elliptic<Geometry, Matrix, container> > m_multi_elliptic;
    std::vector<dg::TensorElliptic<Geometry, Matrix, container> > m_multi_tensorelliptic;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > m_multi_g1, m_multi_g0;
    
    dg::KrylovSqrtCauchySolve< Geometry, Matrix, container> m_sqrtsolve;
    
    dg::Advection<Geometry, Matrix, container> m_adv;
    
    dg::MultigridCG2d<Geometry, Matrix, container> m_multigrid;
    dg::Extrapolation<container> m_phi_ex, m_gamma_phi_ex, m_psi_ex, m_gamma_n_ex;
    std::vector<container> m_multi_chi, m_multi_iota;
    
    Matrix m_centered[2];
    
    const container m_w2d, m_v2d, m_one;

    double mass_, energy_, diff_, ediff_;
    const poet::Parameters m_p;
};

template< class Geometry, class M, class container>
Poet< Geometry, M,  container>::Poet( const Geometry& grid, const Parameters& p ):
    m_chi( evaluate( dg::zero, grid)), m_omega(m_chi), m_iota(m_chi), m_gamma_n(m_chi), m_gamma_phi(m_chi),
    m_binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)),
    m_n_old(m_chi),    m_gamma_n_old(m_chi), m_phi_old(m_chi),    m_gamma_phi_old(m_chi),
    m_lapMperp( grid, dg::normed, dg::centered),
    m_multigrid( grid, 3),
    m_phi_ex( 2, m_chi),  m_gamma_phi_ex(2, m_chi), m_psi_ex( 2, m_chi), m_gamma_n_ex( 2, m_chi),
    m_w2d( dg::create::volume(grid)), m_v2d( dg::create::inv_weights(grid)), m_one( dg::evaluate(dg::one, grid)),
    m_p(p)
{
    m_phi[0] = m_phi[1] = m_dyphi[0] = m_dyphi[1] = m_ype[0] = m_ype[1]  = m_chi; 
    m_dyy[0] = m_dyy[1] = m_lnype[0] = m_lnype[1] = m_lapy[0] = m_lapy[1] = m_chi;
    m_multi_chi= m_multigrid.project( m_chi);
    m_multi_iota= m_multigrid.project( m_chi);
    m_multi_elliptic.resize(3);
    m_multi_tensorelliptic.resize(3);
    m_multi_g1.resize(3);
    m_multi_g0.resize(3);
    m_adv.construct(grid);
    m_centered[0] = dg::create::dx( grid, grid.bcx(), dg::centered);
    m_centered[1] = dg::create::dy( grid, grid.bcy(), dg::centered);
    for( unsigned u=0; u<3; u++)
    {
        m_multi_elliptic[u].construct(       m_multigrid.grid(u), dg::not_normed, dg::centered, p.jfactor);
        m_multi_tensorelliptic[u].construct( m_multigrid.grid(u), dg::not_normed, dg::centered, p.jfactor); //only centered implemented
        
        m_multi_g0[u].construct( m_multigrid.grid(u), -p.tau[1], dg::centered, p.jfactor);
        if( m_p.equations == "ff-O2" ) {
            m_multi_g1[u].construct( m_multigrid.grid(u), -p.tau[1], dg::centered, p.jfactor);
            m_sqrtsolve.construct( m_multi_g1[0], grid, m_chi,  p.eps_cauchy, p.maxiter_sqrt, p.maxiter_cauchy,  p.eps_gamma);
        }
        else {
            m_multi_g1[u].construct( m_multigrid.grid(u), -0.5*p.tau[1], dg::centered, p.jfactor);
        }               
    }

}

template< class G,  class M, class container>
const container& Poet<G,  M,  container>::compute_psi( double t, const container& potential)
{
    if( m_p.equations == "ff-O4" )
    {     
        dg::blas1::pointwiseDot(m_binv, m_binv, m_chi);
        m_multi_tensorelliptic[0].variation( m_gamma_phi, m_p.tau[1]/2., m_chi, m_omega);
        dg::blas1::axpby( 1.,  m_gamma_phi, 1.0, m_omega,  m_phi[1]);
    }
    else if ( m_p.equations == "ff-O2") {
        //elliptic part
        m_multi_elliptic[0].variation(m_gamma_phi, m_phi[1]);   
        dg::blas1::pointwiseDot(1.0, m_binv, m_binv, m_phi[1], 0.0, m_omega); 
        dg::blas1::axpby( 1.,  m_gamma_phi, -0.5, m_omega,  m_phi[1]); 
    }
    else {
        if (m_p.tau[1] == 0.) {
            dg::blas1::axpby( 1., potential, 0.,m_phi[1]); 
        }
        else {
            m_psi_ex.extrapolate( t, m_phi[1]);
            std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_g1, m_phi[1], potential, m_p.eps_gamma);
            m_psi_ex.update( t, m_phi[1]);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma);
        }
        //compute (nabla phi)^2
        m_multi_elliptic[0].variation(potential, m_omega); //m_omega used for ExB energy derivation
        //compute psi
        if(m_p.equations == "ff-lwl")
        {

            dg::blas1::pointwiseDot( -0.5, m_binv, m_binv, m_omega, 1., m_phi[1]);
        }
    }
    return m_phi[1];
}

template<class G,  class M,  class container>
const container& Poet<G,  M, container>::polarisation( double t, const std::array<container,2>& y)
{
    //Compute chi and m_iota for global models
    if( m_p.equations == "ff-lwl" || m_p.equations == "ff-O2" || m_p.equations == "ff-O4")
    {
        dg::assign( y[1], m_chi);
        dg::blas1::plus( m_chi, 1.);
        dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi = n_i
        dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi *= m_binv^2
        m_multigrid.project( m_chi, m_multi_chi);
        if( m_p.equations == "ff-O4" )
        {
            dg::blas1::pointwiseDot(m_p.tau[1]/4., m_chi,m_binv,m_binv,0., m_chi);
            m_multigrid.project( m_chi, m_multi_iota);
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
    if( m_p.equations == "ff-O4" )
    {
        dg::blas2::symv( m_multi_g1[0], y[0], m_chi); //invG ne-1
        dg::blas2::symv( m_v2d, m_chi, m_gamma_n);
    }
    else if( m_p.equations == "ff-O2" )
    {
        dg::blas1::axpby(1.0, y[0], -1.0, m_n_old);
        m_sqrtsolve(m_n_old, m_gamma_n); 
        dg::blas1::axpby( 1.0, m_gamma_n_old, 1.0, m_gamma_n); 
        dg::blas1::copy(y[0], m_n_old);
        dg::blas1::copy(m_gamma_n, m_gamma_n_old); 
//         m_sqrtsolve(y[0], m_gamma_n); //without old solution
    }
    else 
    {  
        m_gamma_n_ex.extrapolate(t, m_gamma_n);
        std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_g1, m_gamma_n, y[1], m_p.eps_gamma);
        m_gamma_n_ex.update(t, m_gamma_n);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
    }

    if( m_p.equations == "ff-O2" || m_p.equations == "ff-O4" )
        dg::blas1::axpby(1., y[1],  -1., m_gamma_n, m_omega);    
    else
    {
        dg::blas1::axpby( -1., y[0], 1., m_gamma_n, m_omega); 
        if (m_p.equations == "df-O2") 
        {            
            dg::blas2::symv(m_multi_g0[0], m_omega, m_chi);
            dg::blas2::symv(m_v2d, m_chi, m_omega);
        }
    }

    //solve polarization equation for phi
    if(m_p.equations == "ff-O4" )
    {
        m_gamma_phi_ex.extrapolate(t, m_gamma_phi);
        std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_tensorelliptic, m_gamma_phi, m_omega, m_p.eps_pol);
        m_gamma_phi_ex.update( t, m_gamma_phi);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);

        dg::blas2::symv(m_multi_g1[0], m_gamma_phi, m_phi[0]); 
        dg::blas1::pointwiseDot( m_v2d, m_phi[0], m_phi[0]);
    }
    else if( m_p.equations == "ff-O2" )
    {
        m_gamma_phi_ex.extrapolate(t, m_gamma_phi);
        std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_elliptic, m_gamma_phi, m_omega, m_p.eps_pol);
        m_gamma_phi_ex.update( t, m_gamma_phi);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);

        dg::blas1::axpby(1.0, m_gamma_phi, -1.0, m_gamma_phi_old); //        
        m_sqrtsolve(m_gamma_phi_old, m_phi[0]); 
        dg::blas1::axpby( 1.0, m_phi_old, 1.0, m_phi[0]); //       
        dg::blas1::copy(m_gamma_phi, m_gamma_phi_old);//     
        dg::blas1::copy(m_phi[0], m_phi_old); 
//         m_sqrtsolve(m_gamma_phi, m_phi[0]);    //without old solution
    }
    else
    {
        m_phi_ex.extrapolate(t, m_phi[0]);
        std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_elliptic, m_phi[0], m_omega, m_p.eps_pol);
        m_phi_ex.update( t, m_phi[0]);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);
    }

    return m_phi[0];
}

template< class G,  class M,  class container>
void Poet<G,  M,  container>::operator()( double t, const std::array<container,2>& y, std::array<container,2>& yp)
{
    //y[0] = N_e - 1 or delta N_e
    //y[1] = N_i - 1 or delta N_i
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    m_phi[0] = polarisation( t, y);
    m_phi[1] = compute_psi( t, m_phi[0]);

    for( unsigned i=0; i<y.size(); i++)
    {
        dg::blas1::transform( y[i], m_ype[i], dg::PLUS<double>(1.));
        dg::blas1::transform( m_ype[i], m_lnype[i], dg::LN<double>());
        dg::blas2::symv( m_lapMperp, y[i], m_lapy[i]);
    }

    /////////////////////////update energetics, 2% of total time///////////////
    mass_ = dg::blas2::dot( m_one, m_w2d, y[0] ); //take real ion density which is electron density!!
    diff_ = m_p.nu*dg::blas2::dot( m_one, m_w2d, m_lapy[0]);
    if(m_p.equations == "ff-lwl" || m_p.equations == "ff-O4"  || m_p.equations == "ff-O2" )
    {
        double Ue = dg::blas2::dot( m_lnype[0], m_w2d, m_ype[0]);
        double Ui = m_p.tau[1]*dg::blas2::dot( m_lnype[1], m_w2d, m_ype[1]);
        double Uphi = 0.5*dg::blas2::dot( m_ype[1], m_w2d, m_omega);
        energy_ = Ue + Ui + Uphi;

        double Ge = - dg::blas2::dot( m_one, m_w2d, m_lapy[0]) - dg::blas2::dot( m_lapy[0], m_w2d, m_lnype[0]); // minus
        double Gi = - m_p.tau[1]*(dg::blas2::dot( m_one, m_w2d, m_lapy[1]) + dg::blas2::dot( m_lapy[1], m_w2d, m_lnype[1])); // minus
        double Gphi = -dg::blas2::dot( m_phi[0], m_w2d, m_lapy[0]);
        double Gpsi = -dg::blas2::dot( m_phi[1], m_w2d, m_lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = m_p.nu*( Ge + Gi - Gphi + Gpsi);
    }
    else
    {
        double Ue = 0.5*dg::blas2::dot( y[0], m_w2d, y[0]);
        double Ui = 0.5*m_p.tau[1]*dg::blas2::dot( y[1], m_w2d, y[1]);
        double Uphi = 0.5*dg::blas2::dot( m_one, m_w2d, m_omega);
        energy_ = Ue + Ui + Uphi;

        double Ge = - dg::blas2::dot( y[0], m_w2d, m_lapy[0]); // minus
        double Gi = - m_p.tau[1]*(dg::blas2::dot( y[1], m_w2d, m_lapy[1])); // minus
        double Gphi = -dg::blas2::dot( m_phi[0], m_w2d, m_lapy[0]);
        double Gpsi = -dg::blas2::dot( m_phi[1], m_w2d, m_lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = m_p.nu*( Ge + Gi - Gphi + Gpsi);
    }
    ///////////////////////////////////////////////////////////////////////

    for( unsigned i=0; i<y.size(); i++) 
    {
        //ExB drift  - v_y dy n - v_x dx n
        dg::blas2::symv( -1., m_centered[1], m_phi[i], 0., m_chi); //v_x
        dg::blas2::symv( 1., m_centered[0], m_phi[i], 0., m_iota); //v_y
        m_adv.upwind( -1., m_chi, m_iota, y[i], 0., yp[i]);   
        if(m_p.equations == "ff-lwl" || m_p.equations == "ff-O4" || m_p.equations == "ff-O2") {
            dg::blas1::pointwiseDot( m_binv, yp[i], yp[i]);
        }
        //GradB drift and ExB comprssion
        dg::blas2::symv( m_centered[1], y[i], m_dyy[i]);
        dg::blas2::symv( m_centered[1], m_phi[i], m_dyphi[i]);
        if(m_p.equations == "ff-lwl" || m_p.equations == "ff-O4" || m_p.equations == "ff-O2") {
            dg::blas1::pointwiseDot( m_dyphi[i], m_ype[i], m_dyphi[i]);
        }
        dg::blas1::axpbypgz( m_p.kappa, m_dyphi[i], m_p.tau[i]*m_p.kappa, m_dyy[i], 1., yp[i]);
        
        // add diffusion
        compute_diff( 1., y[i], 1., yp[i]);            
    }

    return;
}

}//namespace poet
