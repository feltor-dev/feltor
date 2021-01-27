#pragma once
#include <exception>
#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/matrixsqrt.h"
namespace toefl
{

struct Upwind{
DG_DEVICE
    void operator()( double& result, double fw, double bw, double v)
    {
        if( v > 0)
            result -= bw; // yp = - Div( F)
        else
            result -= fw;
    }
};
    
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

template< class Geometry, class IMatrix, class Matrix, class container >
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

    void gamma1_y( const container& y, container& yp)
    {
        if (equations == "arbpolO2")
        {
            krylovsqrtcauchysolve(y, yp, 10); 
        }
        else
        {
            dg::blas2::symv( multi_gamma1[0], y ,chi); 
            dg::blas2::symv( v2d, chi, yp);
        }
    }
    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * y[0] = N_e - 1, 
     * y[1] = N_i - 1 || y[1] = Omega
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
    //use chi and omega as helpers to compute square velocity in omega
    const container& compute_psi( double t, const container& potential);
    const container& polarisation( double t, const std::vector<container>& y);

    container chi, omega, iota, m_fine_phi[2], m_fine_v[2], m_fine_y[2], m_fine_yp[2], m_fine_ype[2], m_fine_chi, m_fine_omega, m_fine_iota;
    const container binv; //magnetic field
    container m_fine_binv;
    std::vector<container> phi, dyphi, ype;
    std::vector<container> dyy, lny, lapy;
    std::vector<container> m_v;
    
    container gamma_n, gamma_phi;

    
    //matrices and solvers
    dg::Elliptic<Geometry, Matrix, container> pol, laplaceM; //contains normalized laplacian
    std::vector<dg::Elliptic<Geometry, Matrix, container> > multi_pol;
    std::vector<dg::ArbPol<Geometry, Matrix, container> > multi_arbpol;
    std::vector<dg::Helmholtz<Geometry,  Matrix, container> > multi_gamma1;
    dg::Helmholtz<Geometry,  Matrix, container> m_fine_gamma1;
    KrylovSqrtCauchySolve<Geometry, Matrix, cusp::dia_matrix<int, dg::get_value_type<container>, cusp::device_memory>, cusp::coo_matrix<int, dg::get_value_type<container>, cusp::device_memory>, container> krylovsqrtcauchysolve, krylovsqrtcauchysolve_fine;
    dg::ArakawaX< Geometry, Matrix, container> arakawa, arakawa_fine;

    dg::MultigridCG2d<Geometry, Matrix, container> multigrid;
    dg::Extrapolation<container> old_phi, old_gamma_phi, old_psi, old_gammaN;
    std::vector<container> multi_chi, multi_iota;

    Matrix m_forward[2], m_backward[2], m_centered[2], m_forward_f[2], m_backward_f[2], m_centered_f[2];
    IMatrix m_inter, m_project;
  
    
    const container w2d,v2d, one;
    const std::vector<double> eps_pol;
    const double eps_gamma;
    const double kappa, friction, nu, tau;
    const std::string equations, m_advection, m_multiplication;
    bool boussinesq;

    double mass_, energy_, diff_, ediff_;

};

template< class Geometry,  class IM, class M, class container>
Explicit< Geometry, IM, M, container>::Explicit( const Geometry& grid, const Parameters& p ):
    chi( evaluate( dg::zero, grid)), omega(chi), iota(chi),
    binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)),
    phi( 2, chi), dyphi( phi), ype(phi),
    dyy(2,chi), lny( dyy), lapy(dyy), m_v(2,chi),
    gamma_n(chi), gamma_phi(chi),
    pol(     grid, dg::not_normed, dg::centered, p.jfactor),
    laplaceM( grid, dg::normed, dg::centered),
    multigrid( grid, 3),
    old_phi( 2, chi),  old_gamma_phi(2, chi), old_psi( 2, chi), old_gammaN( 2, chi),
    w2d( dg::create::volume(grid)), v2d( dg::create::inv_weights(grid)), one( dg::evaluate(dg::one, grid)),
    eps_pol(p.eps_pol), eps_gamma( p.eps_gamma), kappa(p.kappa), friction(p.friction), nu(p.nu), tau( p.tau), equations( p.equations),  m_advection(p.advection), m_multiplication(p.multiplication), boussinesq(p.boussinesq)
{
    multi_chi= multigrid.project( chi);
    multi_iota= multigrid.project( chi);
    multi_pol.resize(3);
    multi_arbpol.resize(3);
    multi_gamma1.resize(3);
    for( unsigned u=0; u<3; u++)
    {
        multi_pol[u].construct( multigrid.grid(u), dg::not_normed, dg::centered, p.jfactor);
        multi_arbpol[u].construct( multigrid.grid(u),  dg::centered, p.jfactor);
        if( equations == "arbpolO2" ) {
            multi_gamma1[u].construct( multigrid.grid(u), -p.tau, dg::centered, p.jfactor);
        }
        else {
            multi_gamma1[u].construct( multigrid.grid(u), -0.5*p.tau, dg::centered, p.jfactor);
        }    
            
    }
    krylovsqrtcauchysolve.construct(multi_gamma1[0], multigrid.grid(0), chi,  p.eps_time, 500, p.eps_gamma);
    
    if (m_multiplication == "projection")
    {
        Geometry fine_grid = grid;
        fine_grid.set( 2*grid.n()-1, grid.Nx(), grid.Ny());
        m_inter = dg::create::interpolation( fine_grid, grid);
        m_project = dg::create::projection( grid, fine_grid);
        m_centered[0] = dg::create::dx( fine_grid, grid.bcx(), dg::centered);
        m_centered[1] = dg::create::dy( fine_grid, grid.bcy(), dg::centered);
        m_forward[0] = dg::create::dx( fine_grid, dg::inverse( grid.bcx()), dg::forward);
        m_forward[1] = dg::create::dy( fine_grid, dg::inverse( grid.bcy()), dg::forward);
        m_backward[0] = dg::create::dx( fine_grid, dg::inverse( grid.bcx()), dg::backward);
        m_backward[1] = dg::create::dy( fine_grid, dg::inverse( grid.bcy()), dg::backward);
        for (unsigned i=0;i<2;i++)
        {
            m_fine_phi[i] = dg::evaluate( dg::zero, fine_grid);
            m_fine_y[i]   = dg::evaluate( dg::zero, fine_grid);
            m_fine_yp[i]  = dg::evaluate( dg::zero, fine_grid);
            m_fine_ype[i] = dg::evaluate( dg::zero, fine_grid);
            m_fine_v[i]   = dg::evaluate( dg::zero, fine_grid);
        }

        m_fine_chi = dg::evaluate( dg::zero, fine_grid);
        m_fine_omega = dg::evaluate( dg::zero, fine_grid);
        m_fine_iota = dg::evaluate( dg::zero, fine_grid);
        arakawa_fine.construct( fine_grid);
        arakawa.construct(grid);

        m_fine_binv = dg::evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), fine_grid);
        
        if( equations == "arbpolO2" ) {
            m_fine_gamma1.construct( fine_grid, -p.tau, dg::centered, p.jfactor);
        }
        else {
            m_fine_gamma1.construct( fine_grid, -0.5*p.tau, dg::centered, p.jfactor);
        }  
        krylovsqrtcauchysolve_fine.construct(m_fine_gamma1, fine_grid, m_fine_chi,  p.eps_time, 500, p.eps_gamma);

    }
    else
    {
        m_centered[0] = dg::create::dx( grid, grid.bcx(), dg::centered);
        m_centered[1] = dg::create::dy( grid, grid.bcy(), dg::centered);
        m_forward[0] = dg::create::dx( grid, dg::inverse( grid.bcx()), dg::forward);
        m_forward[1] = dg::create::dy( grid, dg::inverse( grid.bcy()), dg::forward);
        m_backward[0] = dg::create::dx( grid, dg::inverse( grid.bcx()), dg::backward);
        m_backward[1] = dg::create::dy( grid, dg::inverse( grid.bcy()), dg::backward);
        
        m_centered_f[0] = dg::create::dx( grid, dg::NEU, dg::centered);
        m_centered_f[1] = dg::create::dy( grid, dg::NEU, dg::centered);
        m_forward_f[0] = dg::create::dx( grid, dg::inverse( dg::NEU), dg::forward);
        m_forward_f[1] = dg::create::dy( grid, dg::inverse( dg::NEU), dg::forward);
        m_backward_f[0] = dg::create::dx( grid, dg::inverse( dg::NEU), dg::backward);
        m_backward_f[1] = dg::create::dy( grid, dg::inverse( dg::NEU), dg::backward);
        arakawa.construct(grid);
    }

}

template< class G, class IM, class M, class container>
const container& Explicit<G, IM, M, container>::compute_psi( double t, const container& potential)
{
    if( equations == "arbpolO4" )
    {        
        //tensor part
        dg::blas2::gemv( arakawa.dx(), gamma_phi, phi[1]); //R_x Gamma phi          
        dg::blas2::gemv( arakawa.dy(), phi[1],chi); //R_y R_y Gamma phi   
        dg::blas1::pointwiseDot(chi,binv,chi); // 1/B R_y R_x Gamma phi
        dg::blas1::pointwiseDot(tau, chi,chi, 0.0, omega); //omega = tau/B^2 (R_y R_x)^2 Gamma phi)^2
        
        dg::blas2::gemv( arakawa.dx(), phi[1], chi); //R_x R_x Gamma phi
        dg::blas1::pointwiseDot(chi,binv,chi); //1/B R_x R_x Gamma phi   
        dg::blas1::pointwiseDot(tau/2., chi,chi, 1.0, omega); //omega+= tau/2/B^2 (R_x R_x Gamma phi)^2
        
        dg::blas2::gemv( arakawa.dy(), gamma_phi, phi[1]); //R_y Gamma phi                
        dg::blas2::gemv( arakawa.dy(), phi[1], chi); //R_y R_y Gamma phi  
        dg::blas1::pointwiseDot(chi,binv,chi); //1/B R_y R_y Gamma phi
        dg::blas1::pointwiseDot(tau/2., chi,chi, 1.0, omega); //omega+= tau/2/B^2 (R_y R_y Gamma phi)^2
        
        //laplacian part
        dg::blas2::symv(laplaceM,gamma_phi,chi); 
        dg::blas1::pointwiseDot(chi,binv,chi);
        dg::blas1::pointwiseDot(tau/4., chi,chi, -1., omega); //omega-= tau/4/B^2 (lap Gamma phi)^2
        
        //elliptic part
        arakawa.variation(gamma_phi, phi[1]);   // (grad gamma phi)^2
        dg::blas1::pointwiseDot(1.0, binv, binv, phi[1], 1.0, omega); //omega +=  1/B^2 (grad gamma phi)^2 <=> - 2\psi_2
        dg::blas1::axpby( 1.,  gamma_phi, -0.5, omega,  phi[1]);
    }
    else if ( equations == "arbpolO2") {
        //elliptic part
        arakawa.variation(gamma_phi, phi[1]);   // (grad gamma phi)^2
        dg::blas1::pointwiseDot(1.0, binv, binv, phi[1], 0.0, omega); //omega =  1/B^2 (grad gamma phi)^2 <=> - 2\psi_2
        dg::blas1::axpby( 1.,  gamma_phi, -0.5, omega,  phi[1]); // omega = psi_1+ psi_2
    }
    else {
        if(equations == "gravity_local") return potential;
        //in gyrofluid invert Gamma operator
        if( equations == "local" || equations == "global")
        {
            if (tau == 0.) {
                dg::blas1::axpby( 1.,potential, 0.,phi[1]); 
            }
            else {
                old_psi.extrapolate( t, phi[1]);
                std::vector<unsigned> number = multigrid.direct_solve( multi_gamma1, phi[1], potential, eps_gamma);
                old_psi.update( t, phi[1]);
                if(  number[0] == multigrid.max_iter())
                    throw dg::Fail( eps_gamma);
            }
        }
        //compute (nabla phi)^2
        arakawa.variation(potential, omega);
        //compute psi
        if(equations == "global")
        {

            dg::blas1::pointwiseDot( -0.5, binv, binv, omega, 1., phi[1]);
        }
        else if ( equations == "drift_global")
        {
            dg::blas1::pointwiseDot( 0.5, binv, binv, omega, 0., phi[1]);
        }
        else if( equations == "gravity_global" )
            dg::blas1::axpby( 0.5, omega, 0., phi[1]);
    }

    return phi[1];
}


//computes and modifies expy!!
template<class G, class IM, class M, class container>
const container& Explicit<G, IM, M, container>::polarisation( double t, const std::vector<container>& y)
{
    if( equations == "arbpolO4" )
    {
        dg::blas1::transfer( y[1], chi);
        dg::blas1::plus( chi, 1.);
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi = n_i
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2

        multigrid.project( chi, multi_chi);
        dg::blas1::pointwiseDot(tau/4., chi,binv,binv,0., chi);
        multigrid.project( chi, multi_iota);
        for( unsigned u=0; u<3; u++)
        {
            multi_arbpol[u].set_chi( multi_chi[u]);
            multi_arbpol[u].set_iota( multi_iota[u]);
        }
        

        dg::blas2::symv( multi_gamma1[0],y[0],chi); //invG ne-1
        dg::blas2::symv( v2d, chi, gamma_n);

        dg::blas1::axpby(1., y[1],  -1., gamma_n, omega);      

                
        old_gamma_phi.extrapolate(t, gamma_phi);
        std::vector<unsigned> number = multigrid.direct_solve( multi_arbpol, gamma_phi, omega, eps_pol);
        old_gamma_phi.update( t, gamma_phi);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_pol[0]);

        if (m_multiplication == "pointwise")
        {

            dg::blas2::symv(multi_gamma1[0], gamma_phi, phi[0]); //invG gamma_phi
            dg::blas1::pointwiseDot( v2d, phi[0], phi[0]);

        }
        else
        {
            dg::blas2::symv(m_inter, gamma_phi, m_fine_iota); //invG gamma_phi
            dg::blas2::symv(m_fine_gamma1, m_fine_iota, m_fine_chi);
//             dg::blas2::symv(m_fine_gamma1.inv_weights(), m_fine_chi, m_fine_iota);
            dg::blas2::symv(m_project, m_fine_iota,phi[0]);
        }

        

    }
    else if( equations == "arbpolO2" )
    {
        dg::blas1::transfer( y[1], chi);
        dg::blas1::plus( chi, 1.);
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi = n_i
        dg::blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
        multigrid.project( chi, multi_chi);
        for( unsigned u=0; u<3; u++) multi_pol[u].set_chi( multi_chi[u]);
        
        //Compute G^{-1} n_e via LanczosSqrtODE solve
        krylovsqrtcauchysolve(y[0], gamma_n, 10); 

        dg::blas1::axpby(1.,y[1],  -1., gamma_n, omega);  //- G^{-1} n_e + N_i    
        
//         dg::blas1::pointwiseDot(w2d,omega,omega);
        
        //Solve G^{-1} n_e - N_i = nabla. (chi nabla gamma_phi) for gamma_phi         
        old_gamma_phi.extrapolate(t, gamma_phi);
        std::vector<unsigned> number = multigrid.direct_solve( multi_pol, gamma_phi, omega, eps_pol);
        old_gamma_phi.update( t, gamma_phi);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_pol[0]);

        //Compute \phi = G^{-1} \gamma_phi via LanczosSqrtODE solve
        if (m_multiplication == "pointwise")
        {

            krylovsqrtcauchysolve(gamma_phi, phi[0], 10);    

        }
        else
        {
            dg::blas2::symv(m_inter, gamma_phi, m_fine_iota); //invG gamma_phi
            krylovsqrtcauchysolve_fine(m_fine_iota, m_fine_chi, 10);    
//             dg::blas2::symv(m_fine_gamma1.inv_weights(), m_fine_chi, m_fine_iota);
            dg::blas2::symv(m_project, m_fine_iota,phi[0]);
        }

    }
    else {
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
            if (tau == 0.) {
                dg::blas1::axpby( 1., y[1], 0.,gamma_n); //chi = N_i - 1
            }
            else {
                old_gammaN.extrapolate(t, gamma_n);
                std::vector<unsigned> number = multigrid.direct_solve( multi_gamma1, gamma_n, y[1], eps_gamma);
                old_gammaN.update(t, gamma_n);
                if(  number[0] == multigrid.max_iter())
                    throw dg::Fail( eps_gamma);
            }
            dg::blas1::axpby( -1., y[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
        }
        else
            dg::blas1::axpby( -1. ,y[1], 0., omega);
        if( equations == "global" || equations == "gravity_global" || equations == "drift_global")
            if( boussinesq)
                dg::blas1::pointwiseDivide( omega, chi, omega);
        //invert

        old_phi.extrapolate(t, phi[0]);
        std::vector<unsigned> number = multigrid.direct_solve( multi_pol, phi[0], omega,eps_pol);
        old_phi.update( t, phi[0]);
        if(  number[0] == multigrid.max_iter())
            throw dg::Fail( eps_pol[0]);
    }
    return phi[0];
}

template< class G, class IM, class M, class container>
void Explicit<G, IM, M, container>::operator()( double t, const std::vector<container>& y, std::vector<container>& yp)
{
    //y[0] = N_e - 1
    //y[1] = N_i - 1 || y[1] = Omega
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
    if(equations == "arbpolO4"  || equations == "arbpolO2")
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
    else if(equations == "global"  )
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
        double Ephi = dg::blas2::dot( ype[0], w2d, phi[1]); //phi[1] equals 0.5*u_E^2
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
        double _tau[2] = {-1., tau};

        for( unsigned i=0; i<y.size(); i++) 
        {
            if (m_multiplication == "pointwise")
            {
                if (m_advection == "arakawa")
                {
                    arakawa( y[i], phi[i], yp[i]);
                    if(equations == "global" || equations == "arbpolO4" || equations == "arbpolO2")
                    {
                    dg::blas1::pointwiseDot( binv, yp[i], yp[i]);
                    }
                    dg::blas2::gemv( arakawa.dy(), y[i], dyy[i]);
                    dg::blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2") {
                    dg::blas1::pointwiseDot( dyphi[i], ype[i], dyphi[i]);
                    }
                    dg::blas1::axpbypgz( kappa, dyphi[i], _tau[i]*kappa, dyy[i], 1., yp[i]);
                }
                if (m_advection == "upwind")
                {
                    dg::blas1::copy( 0., yp[i]);                    
                //dx ( nv_x)
                    dg::blas2::symv( -1., m_centered[1], phi[i], 0., m_v[i]); // - dy psi
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(binv,m_v[i],m_v[i]);   //v_x= - B^{1} dy psi
                    }                    
                    dg::blas1::pointwiseDot( ype[i], m_v[i], chi); //f_x = N v_x
                    dg::blas2::symv( m_forward[0], chi, omega);
                    dg::blas2::symv( m_backward[0], chi, iota);
                    dg::blas1::subroutine( toefl::Upwind(), yp[i], omega, iota, m_v[i]);
    //              //dy ( nv_y)
                    dg::blas2::symv( 1., m_centered[0], phi[i], 0., m_v[i]); //v_y =  dx psi
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(binv, m_v[i], m_v[i]);         //v_y = B^{1} dx psi
                    }
                    //add gradB drift
                    dg::blas1::plus(m_v[i],-kappa*_tau[i]);


                    dg::blas1::pointwiseDot( ype[i], m_v[i], chi); //f_y = N v_y
                    dg::blas2::symv( m_forward[1], chi, omega);
                    dg::blas2::symv( m_backward[1], chi, iota);
                    dg::blas1::subroutine( toefl::Upwind(), yp[i], omega, iota, m_v[i]);
                }
                if (m_advection == "centered")
                {
                    //centered scheme
                    //dx ( nv_x)
                    dg::blas2::symv( -1., m_centered[1], phi[i], 0., m_v[i]); //v_x
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(binv, m_v[i], m_v[i]);         
                    }        
                    
                    dg::blas1::pointwiseDot( ype[i], m_v[i], chi); //f_x
                    dg::blas2::symv( -1., m_centered_f[0], chi, 0., yp[i]);
                    //dy ( nv_y)
                    dg::blas2::symv(  1., m_centered[0], phi[i], 0., m_v[i]); //v_y
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(binv, m_v[i], m_v[i]);         
                    }        
                    dg::blas1::plus(m_v[i],-kappa*_tau[i]);
                    dg::blas1::pointwiseDot( ype[i], m_v[i], chi); //f_y
                    dg::blas2::symv( -1., m_centered_f[1], chi, 1., yp[i]);
                }
            }
            else
            {
                dg::blas2::symv( m_inter, y[i], m_fine_y[i]);
                dg::blas2::symv( m_inter, phi[i], m_fine_phi[i]);
                dg::blas1::transform( m_fine_y[i], m_fine_ype[i], dg::PLUS<double>(1.));

                if( m_advection == "arakawa")
                {
                    arakawa_fine( m_fine_y[i], m_fine_phi[i], m_fine_yp[i]); //A(y,psi)-> yp
                    if(equations == "global" || equations == "arbpolO4" || equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot( m_fine_binv, m_fine_yp[i], m_fine_yp[i]);
                    }
                    dg::blas2::gemv( m_centered[1], m_fine_y[i], m_fine_chi);
                    dg::blas2::gemv( m_centered[1], m_fine_phi[i], m_fine_iota);
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2") {
                        dg::blas1::transform( m_fine_y[i], m_fine_ype[i], dg::PLUS<double>(1.));
                        dg::blas1::pointwiseDot( m_fine_iota, m_fine_ype[i], m_fine_iota);
                    }
                    dg::blas1::axpbypgz( kappa, m_fine_iota, _tau[i]*kappa, m_fine_chi, 1., m_fine_yp[i]);
                }
                if (m_advection == "upwind")
                {
                    dg::blas1::copy( 0., m_fine_yp[i]);                    
                //dx ( nv_x)
                    dg::blas2::symv( -1., m_centered[1], m_fine_phi[i], 0., m_fine_v[i]); // - dy psi
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(m_fine_binv,m_fine_v[i],m_fine_v[i]);   //v_x= - B^{1} dy psi
                    }                    
                    dg::blas1::pointwiseDot( m_fine_ype[i], m_fine_v[i], m_fine_chi); //f_x = N v_x
                    dg::blas2::symv( m_forward[0], m_fine_chi, m_fine_omega);
                    dg::blas2::symv( m_backward[0], m_fine_chi, m_fine_iota);
                    dg::blas1::subroutine( toefl::Upwind(), m_fine_yp[i], m_fine_omega, m_fine_iota, m_fine_v[i]);
    //              //dy ( nv_y)
                    dg::blas2::symv( 1., m_centered[0], m_fine_phi[i], 0., m_fine_v[i]); //v_y =  dx psi
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(binv, m_fine_v[i], m_fine_v[i]);         //v_y = B^{1} dx psi
                    }
                    //add gradB drift
                    dg::blas1::plus(m_fine_v[i],-kappa*_tau[i]);


                    dg::blas1::pointwiseDot( m_fine_ype[i], m_fine_v[i], m_fine_chi); //f_y = N v_y
                    dg::blas2::symv( m_forward[1], m_fine_chi, m_fine_omega);
                    dg::blas2::symv( m_backward[1], m_fine_chi, m_fine_iota);
                    dg::blas1::subroutine( toefl::Upwind(), m_fine_yp[i], m_fine_omega, m_fine_iota, m_fine_v[i]);
                }
                if (m_advection == "centered")
                {
                    //centered scheme
                    //dx ( nv_x)
                    dg::blas2::symv( -1., m_centered[1], m_fine_phi[i], 0., m_fine_v[i]); //v_x
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(m_fine_binv, m_fine_v[i], m_fine_v[i]);         
                    }        
                    
                    dg::blas1::pointwiseDot( m_fine_ype[i], m_fine_v[i], m_fine_chi); //f_x
                    dg::blas2::symv( -1., m_centered[0], m_fine_chi, 0., m_fine_yp[i]);
                    //dy ( nv_y)
                    dg::blas2::symv(  1., m_centered[0], m_fine_phi[i], 0., m_fine_v[i]); //v_y
                    if(equations == "global" || equations == "arbpolO4"|| equations == "arbpolO2")
                    {
                        dg::blas1::pointwiseDot(m_fine_binv, m_fine_v[i], m_fine_v[i]);         
                    }        
                    dg::blas1::plus(m_fine_v[i],-kappa*_tau[i]);
                    dg::blas1::pointwiseDot( m_fine_ype[i], m_fine_v[i], m_fine_chi); //f_y
                    dg::blas2::symv( -1., m_centered[1], m_fine_chi, 1., m_fine_yp[i]);
                }
                dg::blas2::symv( m_project, m_fine_yp[i], yp[i]);

            }
        }
    }

    //If you want to test an explicit timestepper:
    //for( unsigned i=0; i<y.size(); i++)
    //    dg::blas2::gemv( -nu, laplaceM, y[i], 1., yp[i]);
    return;
}

}//namespace dg
