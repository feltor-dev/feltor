#pragma once
#include <exception>
#include "dg/algorithm.h"
#include "dg/matrix/matrix.h"
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
    const container& potential( int i) const { return m_psi[i];}
    const container& density(   int i) const { return m_ype[i];}
    const container& psi2() const {return m_psi2;}
    const container& gradn(int i) const { return m_gradn[i]; }
    const container& gradphi(int i) const { return m_gradphi[i]; }
    const Geometry& grid() const {return m_multigrid.grid(0);}
    const container& volume() const {return m_volume;}
    void compute_vorticity ( double alpha, const container& in, double beta, container& result)
    {
        m_lapMperp.set_chi(m_binv);
        dg::blas2::symv( -1.0*alpha, m_lapMperp, in, beta, result);
        m_lapMperp.set_chi(1.);
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
        std::vector<unsigned> number = m_multigrid.solve( m_multi_g1, yp, y, m_p.eps_gamma1);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma1);
    }
    /**
     * @brief compute \f$  \Gamma_1 yp = y \f$ (or \f$ \sqrt{\Gamma_1} yp =  y \f$ )
     *
     * @return yp
     */
    void gamma1inv_y( const container& y, container& yp)
    {
        dg::blas2::symv( m_multi_g1[0], y, yp); //invG ne-1
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
        
        std::vector<unsigned> number = m_multigrid.solve( m_multi_elliptic, yp, y, m_p.eps_pol);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);
    }
    /**
     * @brief solve the LWL approximated polarization equation for Ni with ne and potential given for the FF models (not finished TODO) and exactly for the df models
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
        dg::blas2::symv( m_multi_elliptic[0], potential, m_iota);
        
        //apply G0 for df-02
        if( m_p.equations == "df-O2")
        {
            std::vector<unsigned> number = m_multigrid.solve( m_multi_g0, Ni, m_iota, m_p.eps_gamma0);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma0);
        }
        else dg::blas1::copy(m_iota, Ni);
        
        if( m_p.equations == "df-O2" || m_p.equations == "df-lwl")
        {
            dg::blas1::axpby(1.0, ne, 1.0, Ni, m_iota);
            dg::blas2::symv( m_multi_g1[0], m_iota, Ni); //invG ne-1
        }
        else
        {
            dg::blas2::symv( m_multi_g1[0], ne, m_iota); //invG ne-1
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

  private:
    //use chi and m_omega as helpers to compute square velocity in m_omega
    const container& compute_psi( double t, const container& potential);
    const container& polarisation( double t, const std::array<container,2>& y);

    container m_chi, m_omega, m_iota, m_gamma_n, m_psi1, m_psi2, m_rho_m1, m_phi_m1, m_gamma0sqrtinv_rho_m1, m_gamma0sqrt_phi_m1;
    const container m_binv; //magnetic field
    std::array<container,2> m_psi, m_ype, m_gradn, m_gradphi;
    
    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, container>  m_lapMperp; //contains normalized laplacian
    std::vector<dg::Elliptic<Geometry, Matrix, container> > m_multi_elliptic;
    std::vector<dg::mat::TensorElliptic<Geometry, Matrix, container> > m_multi_tensorelliptic;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > m_multi_g1, m_multi_g0;
    
    dg::mat::MatrixSqrt<container> m_sqrt;
    
    dg::Advection<Geometry, Matrix, container> m_adv;
    
    dg::MultigridCG2d<Geometry, Matrix, container> m_multigrid;
    dg::Extrapolation<container> m_phi_ex, m_psi1_ex, m_gamma_n_ex, m_gamma0sqrt_phi_ex, m_rho_ex, m_gamma0sqrtinv_rho_ex;
    std::vector<container> m_multi_chi, m_multi_iota;
    
    Matrix m_centered[2];
    
    const container m_volume;

    const poet::Parameters m_p;
};

template< class Geometry, class M, class container>
Poet< Geometry, M,  container>::Poet( const Geometry& grid, const Parameters& p ):
    m_chi( evaluate( dg::zero, grid)), m_omega(m_chi), m_iota(m_chi), m_gamma_n(m_chi), m_psi1(m_chi), m_psi2(m_chi), m_rho_m1(m_chi), m_phi_m1(m_chi), m_gamma0sqrtinv_rho_m1(m_chi), m_gamma0sqrt_phi_m1(m_chi),
    m_binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)),
    m_lapMperp( grid, dg::centered),
    m_multigrid( grid, 3),
    m_phi_ex( 2, m_chi),  m_psi1_ex(2, m_chi),  m_gamma_n_ex( 2, m_chi), m_gamma0sqrt_phi_ex( 2, m_chi), m_rho_ex(2, m_chi), 
    m_gamma0sqrtinv_rho_ex(2, m_chi),
    m_volume( dg::create::volume(grid)),
    m_p(p)
{
    m_psi[0] = m_psi[1] = m_ype[0] = m_ype[1]  = m_gradn[0] = m_gradn[1] = m_gradphi[0] = m_gradphi[1] = m_chi; 
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
        m_multi_elliptic[u].construct(       m_multigrid.grid(u), dg::centered, p.jfactor);
        m_multi_tensorelliptic[u].construct( m_multigrid.grid(u), dg::centered, p.jfactor);       
        m_multi_g0[u].construct( m_multigrid.grid(u), -p.tau[1], dg::centered, p.jfactor);
        m_multi_g1[u].construct( m_multigrid.grid(u), -0.5*p.tau[1], dg::centered, p.jfactor);     
    }
    m_sqrt.construct( m_multi_g0[0], +1, m_volume, p.eps_gamma0, p.maxiter_sqrt, p.maxiter_cauchy);
}

template< class G,  class M, class container>
const container& Poet<G,  M,  container>::compute_psi( double t, const container& potential)
{
    if( m_p.equations == "ff-O4" ) {     
        dg::blas1::pointwiseDot(m_binv, m_binv, m_chi);
        m_multi_tensorelliptic[0].variation( m_psi1, m_p.tau[1]/2., m_chi, m_psi2);
        dg::blas1::axpby( 1.0, m_psi1, 1.0, m_psi2, m_psi[1]);
    }
    else { //"df-lwl" || "df-O2" || "ff-lwl" || "ff-O2"
        if (m_p.tau[1] == 0.) {
            dg::blas1::axpby( 1., potential, 0., m_psi1); 
        }
        else {
            m_psi1_ex.extrapolate( t, m_psi1);
            std::vector<unsigned> number = m_multigrid.solve( m_multi_g1, m_psi1, potential, m_p.eps_gamma1);
            m_psi1_ex.update( t, m_psi1);
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_gamma1);
        }

        if ( m_p.equations == "ff-O2") {
            m_multi_elliptic[0].variation(-0.5, m_binv, m_psi1, 0.0, m_psi2);
            dg::blas1::axpby( 1.0, m_psi1, 1.0, m_psi2, m_psi[1]);
        }
        else if  (m_p.equations == "ff-lwl") {
            m_multi_elliptic[0].variation(-0.5, m_binv, potential, 0.0, m_psi2);
            dg::blas1::axpby( 1.0, m_psi1, 1.0, m_psi2, m_psi[1]);
        }
        else { //df-O2" || "df-lwl"            
            m_multi_elliptic[0].variation(potential, m_psi2);
            dg::blas1::scal(m_psi2, -0.5);
            if (m_p.equations == "df-O2") {
                dg::blas2::symv(m_multi_g0[0], m_psi2, m_chi);
                dg::blas1::copy( m_chi, m_psi2);
            }
            dg::blas1::axpby( 1.0, m_psi1, 0.0, m_psi[1]);
        }
    }
    return m_psi[1];
}

template<class G,  class M,  class container>
const container& Poet<G,  M, container>::polarisation( double t, const std::array<container,2>& y)
{
    //Compute chi and m_iota for ff models
    if( m_p.equations == "ff-lwl" || m_p.equations == "ff-O2" || m_p.equations == "ff-O4") {
        dg::assign( y[1], m_chi);
        dg::blas1::plus( m_chi, 1.);
        dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi = n_i
        dg::blas1::pointwiseDot( m_binv, m_chi, m_chi); //\chi *= m_binv^2
        m_multigrid.project( m_chi, m_multi_chi);
        if( m_p.equations == "ff-O4" ) {
            dg::blas1::pointwiseDot(m_p.tau[1]/4., m_chi,m_binv,m_binv,0., m_chi);
            m_multigrid.project( m_chi, m_multi_iota);
            for( unsigned u=0; u<3; u++) {
                m_multi_tensorelliptic[u].set_chi( m_multi_chi[u]);
                m_multi_tensorelliptic[u].set_iota( m_multi_iota[u]);
            }
        }
        else { //"ff-lwl" || "ff-O2" 
            for( unsigned u=0; u<3; u++) 
                m_multi_elliptic[u].set_chi( m_multi_chi[u]);
        }
    }
    
    //Compute rho
    if( m_p.equations == "ff-O4" ) {
        dg::blas2::symv( m_multi_g1[0], y[0], m_gamma_n); //invG ne-1
    }
    else { //"ff-lwl" || "df-lwl" || "df-O2"  || "ff-O2"
        m_gamma_n_ex.extrapolate(t, m_gamma_n);
        std::vector<unsigned> number = m_multigrid.solve( m_multi_g1, m_gamma_n, y[1], m_p.eps_gamma1);
        m_gamma_n_ex.update(t, m_gamma_n);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma1);
    }

//     if( m_p.equations == "ff-O2" || m_p.equations == "ff-O4" )
    if(  m_p.equations == "ff-O4" )
        dg::blas1::axpby(1., y[1],  -1., m_gamma_n, m_omega);    
    else { //"ff-lwl" || "df-lwl" || "df-O2" ||  "ff-O2"
        dg::blas1::axpby( 1., m_gamma_n, -1., y[0], m_omega); 
        if (m_p.equations == "df-O2") {            
            dg::blas2::symv(m_multi_g0[0], m_omega, m_chi);
            dg::blas1::copy( m_chi, m_omega);
        }
        else if (m_p.equations == "ff-O2") {
            dg::blas1::axpby(-1.0, m_gamma0sqrtinv_rho_m1, 1.0, m_omega, m_chi);
            dg::blas1::copy(m_omega, m_gamma0sqrtinv_rho_m1);
            dg::apply( m_sqrt, m_chi, m_omega);
            dg::blas1::axpby( 1.0, m_rho_m1, 1.0, m_omega); 
            dg::blas1::copy(m_omega, m_rho_m1);
//             m_sqrt(m_omega, m_chi); //without using linearity
//             dg::blas1::copy(m_chi, m_omega);            
        }
    }

    //solve polarization equation for phi
    if(m_p.equations == "ff-O4" ) {
        m_psi1_ex.extrapolate(t, m_psi1);
        std::vector<unsigned> number = m_multigrid.solve( m_multi_tensorelliptic, m_psi1, m_omega, m_p.eps_pol);
        m_psi1_ex.update( t, m_psi1);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);

        dg::blas2::symv(m_multi_g1[0], m_psi1, m_psi[0]); 
    }
    else if( m_p.equations == "ff-O2" ) {
        m_gamma0sqrt_phi_ex.extrapolate(t, m_iota);
        std::vector<unsigned> number = m_multigrid.solve( m_multi_elliptic, m_iota, m_omega, m_p.eps_pol);
        m_gamma0sqrt_phi_ex.update( t, m_iota); 
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);
        
        dg::blas1::axpby(1.0, m_iota, -1.0, m_gamma0sqrt_phi_m1, m_chi); 
        dg::blas1::copy(m_iota, m_gamma0sqrt_phi_m1);
        dg::apply( m_sqrt, m_chi, m_psi[0]); 
        dg::blas1::axpby( 1.0, m_phi_m1, 1.0, m_psi[0]); 
        dg::blas1::copy(m_psi[0], m_phi_m1);
//         m_sqrt(m_iota, m_psi[0]);    //without using linearity
    }
    else { // "ff-lwl" || "df-lwl" || "df-O2"
        m_phi_ex.extrapolate(t, m_psi[0]);
        std::vector<unsigned> number = m_multigrid.solve( m_multi_elliptic, m_psi[0], m_omega, m_p.eps_pol);
        m_phi_ex.update( t, m_psi[0]);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_pol[0]);
    }

    return m_psi[0];
}

template< class G,  class M,  class container>
void Poet<G,  M,  container>::operator()( double t, const std::array<container,2>& y, std::array<container,2>& yp)
{
    //y[0] = N_e - 1 or delta N_e
    //y[1] = N_i - 1 or delta N_i
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    m_psi[0] = polarisation( t, y);
    m_psi[1] = compute_psi( t, m_psi[0]);


    for( unsigned i=0; i<y.size(); i++) 
    {
        dg::blas1::transform( y[i], m_ype[i], dg::PLUS<double>(1.));

        //ExB drift  - v_y dy n - v_x dx n
        dg::blas2::symv( -1., m_centered[1], m_psi[i], 0., m_chi); //v_x
        dg::blas2::symv(  1., m_centered[0], m_psi[i], 0., m_iota); //v_y
        if (i==0)
        {
            dg::blas1::copy(m_iota, m_gradphi[0]);
            dg::blas1::copy(m_chi, m_gradphi[1]);
	    dg::blas1::scal(m_gradphi[1], -1.0);
        }

        m_adv.upwind( -1., m_chi, m_iota, y[i], 0., yp[i]);   
        if(m_p.equations == "ff-lwl" || m_p.equations == "ff-O4" || m_p.equations == "ff-O2") {
            dg::blas1::pointwiseDot( m_binv, yp[i], yp[i]);
        }
        //GradB drift and ExB comprssion
        dg::blas2::symv( m_centered[1], y[i], m_iota);
        if (i==0)
        {
            dg::blas2::symv( m_centered[0], y[i], m_gradn[0]); //dx n
            dg::blas1::copy(m_iota, m_gradn[1]);
        }
        dg::blas2::symv( m_centered[1], m_psi[i], m_omega);
        if(m_p.equations == "ff-lwl" || m_p.equations == "ff-O4" || m_p.equations == "ff-O2") {
            dg::blas1::pointwiseDot( m_omega, m_ype[i], m_omega);
        }
        dg::blas1::axpbypgz( m_p.kappa, m_omega, m_p.tau[i]*m_p.kappa, m_iota, 1., yp[i]);
        
        // add diffusion
        compute_diff( 1., y[i], 1., yp[i]);            
    }

    return;
}

}//namespace poet
