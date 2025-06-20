#pragma once

#include "dg/algorithm.h"
#include "parameters.h"

namespace asela
{

template< class Geometry, class Matrix, class Container >
struct Asela
{
    Asela( const Geometry& g, asela::Parameters p);

    void operator()( double time,  const std::array<std::array<Container,2>,2>& y, std::array<std::array<Container,2>,2>& yp);

    /// ///////////////////DIAGNOSTICS///////////////////////////////
    const Container& potential( int i) const { return m_phi[i];}
    const Container& aparallel( int i) const { return m_apar[i];}
    const Container& density(   int i) const { return m_n[i];}
    const Container& velocity( int i) const { return m_u[i];}
    const std::array<Container,2>& gradN( int i) const { return m_dFN[i];}
    const std::array<Container,2>& gradU( int i) const { return m_dFU[i];}
    const std::array<Container,2>& gradP( int i) const { return m_dP[i];}
    const std::array<Container,2>& gradA( int i) const { return m_dA[i];}
    const dg::Elliptic<Geometry, Matrix, Container>& laplacianM() const{return m_lapMperp;}
    const Geometry& grid() const {return m_multigrid.grid(0);}
    /// ////////////////DIAGNOSTICS END//////////////////////////////

    void compute_psi( double time);
    void compute_phi( double time, const std::array<Container,2>& n);
    void compute_apar( double time, const std::array<Container,2>& n, std::array<Container,2>& u);
    void compute_perp( double time, const std::array<std::array<Container,2>,2>& y, std::array<std::array<Container,2>,2>& yp);
    void compute_diff( double alpha, const Container& nme, double beta, Container& result)
    {
        if( m_p.nu_perp != 0)
        {
            dg::blas2::gemv( m_lapMperp, nme, m_temp0);
            dg::blas2::gemv( -alpha*m_p.nu_perp, m_lapMperp, m_temp0, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
    }
    void compute_lapM( double alpha, const Container& in, double beta, Container& result)
    {
        dg::blas2::symv( alpha, m_lapMperp, in, beta, result);
    }

  private:
    //Containers
    Container m_temp0, m_temp1, m_temp2;
    std::array<Container,2> m_phi, m_apar, m_n, m_u;
    std::array<std::array<Container,2>,2> m_dFN, m_dBN, m_dFU, m_dBU, m_dP, m_dA;
    std::vector<Container> m_multi_chi;
    //matrices and solvers
    Matrix m_dxF, m_dxB, m_dxC;
    Matrix m_dyF, m_dyB, m_dyC;
    dg::ArakawaX< Geometry, Matrix, Container > m_arakawa;
    dg::Elliptic< Geometry, Matrix, Container > m_lapMperp;
    std::vector<dg::Elliptic<Geometry, Matrix, Container>> m_multi_pol;
    std::vector<dg::Helmholtz<Geometry, Matrix, Container>> m_multi_maxwell, m_multi_invgamma;

    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    dg::Extrapolation<Container> m_old_phi, m_old_psi, m_old_gammaN, m_old_gammaNW, m_old_apar, m_old_gammaApar;

    const asela::Parameters m_p;
};
///@}

template<class Grid, class Matrix, class Container>
Asela<Grid, Matrix, Container>::Asela( const Grid& g, Parameters p):
    //////////the derivative and arakawa operators /////////////////////////
    m_dxF ( dg::create::dx( g, dg::forward)),
    m_dxB ( dg::create::dx( g, dg::backward)),
    m_dxC ( dg::create::dx( g, dg::centered)),
    m_dyF ( dg::create::dy( g, dg::forward)),
    m_dyB ( dg::create::dy( g, dg::backward)),
    m_dyC ( dg::create::dy( g, dg::centered)),
    m_arakawa(g, g.bcx(), g.bcy()),
    //////////the elliptic and Helmholtz operators//////////////////////////
    m_lapMperp ( g,  dg::str2direction(p.direction_diff)),
    m_multigrid( g, 3),
    m_old_phi( 2, dg::evaluate( dg::zero, g)),
    m_old_psi( 2, dg::evaluate( dg::zero, g)),
    m_old_gammaN( 2, dg::evaluate( dg::zero, g)),
    m_old_gammaNW( 2, dg::evaluate( dg::zero, g)),
    m_old_apar( 2, dg::evaluate( dg::zero, g)),
    m_old_gammaApar( 2, dg::evaluate( dg::zero, g)),
    m_p(p)
{
    ////////////////////////////init temporaries///////////////////
    dg::assign( dg::evaluate( dg::zero, g), m_temp0);
    m_temp1 = m_temp2 = m_temp0;
    m_phi[0] = m_phi[1] = m_temp0;
    m_apar = m_n = m_u = m_phi;
    m_dA[0] = m_dA[1] = m_phi;
    m_dFN = m_dBN = m_dFU = m_dBU = m_dP = m_dA;

    //////////////////////////////init elliptic and helmholtz operators////////////
    m_multi_chi = m_multigrid.project( m_temp0);
    m_multi_pol.resize(3);
    for( unsigned u=0; u<3; u++)
    {
        m_multi_pol[u].construct(      m_multigrid.grid(u),  dg::str2direction(m_p.direction_ell), m_p.jfactor);
        m_multi_maxwell.push_back(  {-1., {m_multigrid.grid(u), dg::str2direction(m_p.direction_ell)}});
        m_multi_invgamma.push_back( {-0.5*m_p.tau[1]*m_p.mu[1], {m_multigrid.grid(u), dg::str2direction(m_p.direction_ell)}});
    }
}

template<class Geometry, class Matrix, class Container>
void Asela<Geometry, Matrix, Container>::compute_phi( double time, const std::array<Container,2>& nme)
{
    dg::blas1::axpby( m_p.mu[1], nme[1], m_p.mu[1], 1., m_temp0); //chi =  \mu_i n_i

    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<3; u++)
        m_multi_pol[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    if (m_p.tau[1] == 0.) {
        //compute N_i - n_e
        dg::blas1::axpby( 1., nme[1], -1., nme[0], m_temp0);
    }
    else
    {
        //compute Gamma N_i - n_e
        m_old_gammaN.extrapolate( time, m_temp0);
        std::vector<unsigned> numberG = m_multigrid.solve(
            m_multi_invgamma, m_temp0, nme[1], m_p.eps_gamma);
        m_old_gammaN.update( time, m_temp0);
        if(  numberG[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
        dg::blas1::axpby( -1., nme[0], 1., m_temp0, m_temp0);
    }
    //----------Invert polarisation----------------------------//
    m_old_phi.extrapolate( time, m_phi[0]);
    std::vector<unsigned> number = m_multigrid.solve(
        m_multi_pol, m_phi[0], m_temp0, m_p.eps_pol);
    m_old_phi.update( time, m_phi[0]);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol[0]);
}
template<class Geometry, class Matrix, class Container>
void Asela<Geometry, Matrix, Container>::compute_psi(
    double time)
{
    //-----------Solve for Gamma Phi---------------------------//
    if (m_p.tau[1] == 0.) {
        dg::blas1::copy( m_phi[0], m_phi[1]);
    } else {
        m_old_psi.extrapolate( time, m_phi[1]);
        std::vector<unsigned> number = m_multigrid.solve(
            m_multi_invgamma, m_phi[1], m_phi[0], m_p.eps_gamma);
        m_old_psi.update( time, m_phi[1]);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_p.eps_gamma);
    }
    //-------Compute Psi and derivatives
    dg::blas2::symv( m_dxC, m_phi[0], m_dP[0][0]);
    dg::blas2::symv( m_dyC, m_phi[0], m_dP[0][1]);
    dg::blas1::pointwiseDot( -0.5, m_dP[0][0], m_dP[0][0], -0.5, m_dP[0][1], m_dP[0][1], 1., m_phi[1]);
    dg::blas2::symv( m_dxC, m_phi[1], m_dP[1][0]);
    dg::blas2::symv( m_dyC, m_phi[1], m_dP[1][1]);
}
template<class Geometry, class Matrix, class Container>
void Asela<Geometry, Matrix, Container>::compute_apar(
    double time, const std::array<Container,2>& n, std::array<Container,2>& u)
{
    //on input
    //n[0] = n_e, u[0]:= w_e
    //n[1] = N_i, u[1]:= W_i
    //----------Compute and set chi----------------------------//
    dg::blas1::axpby(  m_p.beta/m_p.mu[1], n[1],
                      -m_p.beta/m_p.mu[0], n[0], m_temp0);
    m_multigrid.project( m_temp0, m_multi_chi);
    for( unsigned u=0; u<m_p.stages; u++)
        m_multi_maxwell[u].set_chi( m_multi_chi[u]);

    //----------Compute right hand side------------------------//
    dg::blas1::pointwiseDot(  m_p.beta, n[1], u[1],
                             -m_p.beta, n[0], u[0],
                              0., m_temp0);
    //----------Invert Induction Eq----------------------------//
    m_old_apar.extrapolate( time, m_apar[0]);
    std::vector<unsigned> number = m_multigrid.solve(
        m_multi_maxwell, m_apar[0], m_temp0, m_p.eps_pol[0]);
    m_old_apar.update( time, m_apar[0]);
    if(  number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_p.eps_pol[0]);
    //----------Compute Derivatives----------------------------//
    //For now we do not apply Gamma on A
    dg::blas1::copy( m_apar[0], m_apar[1]);
    dg::blas2::symv( m_dxC, m_apar[0], m_dA[0][0]);
    dg::blas2::symv( m_dyC, m_apar[0], m_dA[0][1]);
    dg::blas2::symv( m_dxC, m_apar[1], m_dA[1][0]);
    dg::blas2::symv( m_dyC, m_apar[1], m_dA[1][1]);

    //----------Compute Velocities-----------------------------//
    dg::blas1::axpby( 1., u[0], -1./m_p.mu[0], m_apar[0], u[0]);
    dg::blas1::axpby( 1., u[1], -1./m_p.mu[1], m_apar[1], u[1]);
}

template<class G, class M, class Container>
void Asela<G, M, Container>::compute_perp( double, const std::array<std::array<Container,2>,2>& y, std::array<std::array<Container,2>,2>& yp)
{
    if( m_p.advection == "arakawa")
    {
        //Compute using Arakawa brackets
        for( unsigned i=0; i<2; i++)
        {
            //ExB dynamics
            m_arakawa( y[0][i], m_phi[i], yp[0][i]);                 //[N,phi]_RZ
            m_arakawa( y[1][i], m_phi[i], yp[1][i]);                 //[w,phi]_RZ

            // Density equation
            dg::blas1::pointwiseDot( 1., m_n[i], m_u[i], 0., m_temp0);
            m_arakawa( 1., m_apar[i], m_temp0,   1., yp[0][i]); // [Apar, UN]_RZ

            // Velocity Equation
            dg::blas1::transform( m_n[i], m_temp0, dg::LN<double>());
            m_arakawa( m_p.tau[i]/m_p.mu[i], m_apar[i], m_temp0, 1., yp[1][i]);  // + tau/mu [Apar,logN]_RZ
            dg::blas1::pointwiseDot( 1., m_u[i], m_u[i], 0., m_temp0);
            m_arakawa( 0.5, m_apar[i], m_temp0,   1., yp[1][i]);                       // +0.5[Apar,U^2]_RZ
        }
    }
    else if ( m_p.advection == "upwind")
    {
        for( unsigned i=0; i<2; i++)
        {
            //First compute forward and backward derivatives for upwind scheme
            dg::blas2::symv( m_dxF, y[0][i], m_dFN[i][0]);
            dg::blas2::symv( m_dyF, y[0][i], m_dFN[i][1]);
            dg::blas2::symv( m_dxB, y[0][i], m_dBN[i][0]);
            dg::blas2::symv( m_dyB, y[0][i], m_dBN[i][1]);
            dg::blas2::symv( m_dxF, m_u[i], m_dFU[i][0]);
            dg::blas2::symv( m_dyF, m_u[i], m_dFU[i][1]);
            dg::blas2::symv( m_dxB, m_u[i], m_dBU[i][0]);
            dg::blas2::symv( m_dyB, m_u[i], m_dBU[i][1]);
            double mu = m_p.mu[i], tau = m_p.tau[i], beta = m_p.beta;
            dg::blas1::subroutine( [mu, tau, beta] DG_DEVICE (
                    double N, double d0FN, double d1FN,
                              double d0BN, double d1BN,
                    double U, double d0FU, double d1FU,
                              double d0BU, double d1BU,
                              double d0P, double d1P,
                              double d0A, double d1A,
                    double& dtN, double& dtU
                )
                {
                    dtN = dtU = 0;
                    // upwind scheme
                    double v0 = -d1P;
                    double v1 =  d0P;
                    if( beta != 0)
                    {
                        v0 +=   U * d1A;
                        v1 += - U * d0A;
                        //Q: doesn't U in U b_perp create a nonlinearity
                        //in velocity equation that may create shocks?
                        //A: since advection is in perp direction it should not give
                        //shocks. LeVeque argues that for smooth solutions the
                        //upwind discretization should be fine but is wrong for shocks
                    }
                    dtN += ( v0 > 0 ) ? -v0*d0BN : -v0*d0FN;
                    dtN += ( v1 > 0 ) ? -v1*d1BN : -v1*d1FN;
                    dtU += ( v0 > 0 ) ? -v0*d0BU : -v0*d0FU;
                    dtU += ( v1 > 0 ) ? -v1*d1BU : -v1*d1FU;

                    if( beta != 0)
                    {
                        double UA = ( (d0FU+d0BU)*d1A-(d1FU+d1BU)*d0A) / 2.;
                        dtN +=  -N * UA ; // N div U bperp
                        double NA = ( (d0FN+d0BN)*d1A-(d1FN+d1BN)*d0A) / 2.;
                        double PA = ( d0P*d1A-d1P*d0A);
                        dtU +=  -1./mu *  PA
                                -tau/mu *  NA / N;
                    }
                },
                //species depdendent
                m_n[i], m_dFN[i][0], m_dFN[i][1],
                        m_dBN[i][0], m_dBN[i][1],
                m_u[i], m_dFU[i][0], m_dFU[i][1],
                        m_dBU[i][0], m_dBU[i][1],
                        m_dP[i][0], m_dP[i][1],
                        m_dA[i][0], m_dA[i][1],
                //magnetic parameters
                yp[0][i], yp[1][i]
            );
        }
    }
    else
        throw dg::Error(dg::Message(_ping_)<<"Error: Unrecognized advection option: '"<<m_p.advection<<"'! Exit now!");
}

template<class Geometry, class Matrix, class Container>
void Asela<Geometry, Matrix, Container>::operator()( double time,  const std::array<std::array<Container,2>,2>& y, std::array<std::array<Container,2>,2>& yp)
{
    /* y[0][0] := n_e - 1
       y[0][1] := N_i - 1
       y[1][0] := w_e = U_e + Apar_e / mu_e
       y[1][1] := W_i = U_i + Apar_i / mu_i
    */

    dg::Timer t;
    t.tic();

    compute_phi( time, y[0]); //computes phi[0] and Gamma n_i

    compute_psi( time ); //compues phi[1]

    //transform n-1 to n
    dg::blas1::transform( y[0], m_n, dg::PLUS<>(+1.)); //n = y+1

    // Compute Apar and m_U if necessary --- reads and updates m_u
    dg::blas1::copy( y[1], m_u);
    if( m_p.beta != 0)
        compute_apar( time, m_n, m_u);

    //parallel dynamics (uses m_n and m_u)
    compute_perp( time, y, yp);

    // Add diffusion
    for( unsigned i=0; i<2; i++)
    {
        compute_diff( 1., y[0][i], 1., yp[0][i]);
        if( m_p.viscosity == "velocity-viscosity")
            compute_diff(  1., m_u[i], 1., yp[1][i]);
        else if ( m_p.viscosity == "canonical-viscosity")
            compute_diff(  1., y[1][i], 1., yp[1][i]);
        else
            throw dg::Error(dg::Message(_ping_)<<"Error: Unrecognized viscosity type: '"<<m_p.viscosity<<"'! Exit now!");
    }

    t.toc();
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif
    std::cout << "One rhs took "<<t.diff()<<"s\n";
}

} //namespace asela
