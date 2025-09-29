#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "parameters.h"

namespace thermal
{


template< class Geometry, class IMatrix, class Matrix, class Container >
struct PerpDynamics
{
    PerpDynamics( const Geometry&, thermal::Parameters,
        dg::geo::TokamakMagneticField, dg::file::WrappedJsonValue);

    void update_derivatives(
        const Container& apar,
        Container& dxapar, Container& dyapar,
        const std::array<std::vector<Container>,6>& y,
        std::map<std::string, std::vector<Container>>& q
    );
    void update_STderivatives(
        const Container& aparST,
        Container& dxaparST, Container& dyaparST,
        const std::array<std::vector<Container>,6>& y,
        std::map<std::string, std::vector<Container>>& q
    );
    // dtN += ... , dtPperp += ... , dtPpara += ...
    void add_densities_advection(
        unsigned s,
        const Container& apar,
        const Container& dxapar,
        const Container& dyapar,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp
    ) const;

    // dtW += ... , dtQperp += ... , dtQpara += ...
    void add_velocities_advection(
        unsigned s,
        const Container& aparST,
        const Container& dxaparST,
        const Container& dyaparST,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp
    ) const;
    void add_densities_diffusion(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp)
    {
        std::vector<std::string> in = {"N", "Tperp", "Tpara"};
        for( unsigned u=0; u<3; u++)
            compute_perp_laplace( -m_p.nu_perp[u], q.at(in[u])[s], m_p.diff_order,
                m_temp0, m_temp1, 1., yp[u][s]);
        // Ppara interfaces with U
        compute_perp_laplace( m_p.nu_perp[3], q.at("U")[s], m_p.diff_order-1,
            m_temp0, m_temp1, 0., m_temp0);
        dg::blas2::symv( m_dxC, m_temp0, m_temp1);
        dg::blas2::symv( m_dxC, q.at("U")[s], m_temp2);
        dg::blas1::pointwiseDot( 2.*m_p.mu[s], m_temp1, m_temp2, 1., yp[2][s]);

        dg::blas2::symv( m_dyC, m_temp0, m_temp1);
        dg::blas2::symv( m_dyC, q.at("U")[s], m_temp2);
        dg::blas1::pointwiseDot( 2.*m_p.mu[s], m_temp1, m_temp2, 1., yp[2][s]);
    }
    void add_velocities_diffusion(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp)
    {
        compute_perp_laplace( -m_p.nu_perp[3], q.at("ST U")[s], m_p.diff_order,
            m_temp0, m_temp1, 0., m_temp0);
        dg::blas1::pointwiseDivide( 1., m_temp0, q.at("ST N")[s], 1., yp[3][s]);
        compute_perp_laplace( -m_p.nu_perp[4], q.at("ST Uperp")[s], m_p.diff_order,
            m_temp0, m_temp1, 1., yp[4][s]);
        compute_perp_laplace( -m_p.nu_perp[5], q.at("ST Upara")[s], m_p.diff_order,
            m_temp0, m_temp1, 1., yp[5][s]);

        // U interfaces with N
        compute_perp_laplace( -m_p.nu_perp[0], q.at("ST N")[s], m_p.diff_order-1,
            m_temp0, m_temp1, 0., m_temp0);
        // - v_x dx U
        if( m_p.diff_dir == dg::centered)
            dg::blas2::symv( m_dxC, m_temp0, m_temp1);
        else if( m_p.diff_dir == dg::forward)
            dg::blas2::symv( m_dxF, m_temp0, m_temp1);
        else
            dg::blas2::symv( m_dxB, m_temp0, m_temp1);
        dg::blas1::pointwiseDivide( m_temp1, q.at("ST N")[s], m_temp1);
        dg::blas2::symv( m_dxB, q.at("ST U")[s], m_temp2);
        dg::blas2::symv( m_dxF, q.at("ST U")[s], m_temp3);
        dg::blas1::evaluate( yp[3][s], dg::minus_equals(), dg::UpwindProduct(),
                    m_temp1, m_temp2, m_temp3);

        // - v_y dy U
        if( m_p.diff_dir == dg::centered)
            dg::blas2::symv( m_dyC, m_temp0, m_temp1);
        else if( m_p.diff_dir == dg::forward)
            dg::blas2::symv( m_dyF, m_temp0, m_temp1);
        else
            dg::blas2::symv( m_dyB, m_temp0, m_temp1);
        dg::blas1::pointwiseDivide( m_temp1, q.at("ST N")[s], m_temp1);
        dg::blas2::symv( m_dyB, q.at("ST U")[s], m_temp2);
        dg::blas2::symv( m_dyF, q.at("ST U")[s], m_temp3);
        dg::blas1::evaluate( yp[3][s], dg::minus_equals(), dg::UpwindProduct(),
                    m_temp1, m_temp2, m_temp3);

    }
    // y = alpha*(-Delta)^order X + beta * y
    void compute_perp_laplace( double alpha, const Container& in, unsigned order,
            Container& temp0, Container& temp1, double beta, Container& result ) const
    {
        // if beta == 0 result is allowed to alias temp0 or temp1
        if( alpha > 0)
        {
            dg::blas1::copy( in, temp0);
            for( unsigned s=0; s<order; s++)
            {
                using std::swap;
                swap( temp0, temp1);
                dg::blas2::symv( 1., m_lapperp, temp1, 0., temp0);
            }
            dg::blas1::axpby( alpha, temp0, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
    }
    const std::array<Container, 2> & curvNabla () const {
        return m_curvNabla;
    }
    const std::array<Container, 2> & curvKappa () const {
        return m_curvKappa;
    }
    const Container& divCurvKappa() const {
        return m_divCurvKappa;
    }
    const Container& bhatgB( ) const { return m_b_2; } // \pm 1/B
    const Container& divb( ) const { return m_divb; }
    const Container& bphi( ) const { return m_bphi; }
    // Compute divergence using centered derivatives
    // note that no matter how divergence is computed you always loose one order
    // unless the polarisation term or the Laplacian of N,U is computed
    // then the correct direction must be chosen
    // prefactor cannot alias result!!
    // Div ( f v)
    template<class Container2>
    void centered_div( const Container2& prefactor,
            const std::array<Container, 2>& contra_vec,
            Container& temp0, Container& result)
    {
        dg::blas1::pointwiseDot( 1., prefactor, m_detg, contra_vec[0], 0., temp0);
        dg::blas2::symv( m_dxC, temp0, result);
        dg::blas1::pointwiseDot( 1., prefactor, m_detg, contra_vec[1], 0., temp0);
        dg::blas2::symv( 1., m_dyC, temp0, 1., result);
        dg::blas1::pointwiseDivide( 1., result, m_detg, 0., result);
    }
    void centered_v_dot_nabla( const std::array<Container, 2>& contra_vec,
            const Container& f, Container& temp1, Container& result)
    {
        dg::blas2::symv( m_dxC, f, temp1);
        dg::blas1::pointwiseDot( contra_vec[0], temp1, result);
        dg::blas2::symv( m_dyC, f, temp1);
        dg::blas1::pointwiseDot( 1., contra_vec[1], temp1, 1., result);
    }
    private:
    //these should be considered const
    std::array<Container,2> m_curvNabla, m_curvKappa, m_gradLnB;
    Container m_divCurvKappa, m_b_2, m_divb, m_bphi, m_detg; //m_b_2 = bphi(covariant)/detg/B \approx pm 1/B
    // store covariant bphi = +- R for momentum conservation

    Matrix m_dxF, m_dxB, m_dxC, m_dx_P, m_dx_A;
    Matrix m_dyF, m_dyB, m_dyC, m_dy_P, m_dy_A;
    dg::Elliptic2d< Geometry, Matrix, Container> m_lapperp;

    Container m_temp0, m_temp1, m_temp2, m_temp3;

    const thermal::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
};

template<class Grid, class IMatrix, class Matrix, class Container>
PerpDynamics<Grid, IMatrix, Matrix, Container>::PerpDynamics( const Grid& g,
    thermal::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue js
    ):
    m_dxF( dg::create::dx( g, p.bcx, dg::forward) ),
    m_dxB( dg::create::dx( g, p.bcx, dg::backward) ),
    m_dxC( dg::create::dx( g, p.bcx, dg::centered) ),
    m_dx_P(  dg::create::dx( g, p.bcxP, p.pol_dir) ),
    m_dx_A(  dg::create::dx( g, p.bcxA, p.pol_dir) ),
    m_dyF( dg::create::dy( g, p.bcy, dg::forward) ),
    m_dyB( dg::create::dy( g, p.bcy, dg::backward) ),
    m_dyC( dg::create::dy( g, p.bcx, dg::centered) ),
    m_dy_P(  dg::create::dy( g, p.bcyP, p.pol_dir) ),
    m_dy_A(  dg::create::dy( g, p.bcyA, p.pol_dir) ),
    m_p(p), m_js(js)
{
    //--------------------------Construct-------------------------//
    //due to the various approximations bhat and mag not always correspond
    dg::geo::CylindricalVectorLvl0 curvNabla, curvKappa;
    bool reversed_field = false;
    if( mag.ipol()( g.x0(), g.y0()) < 0)
        reversed_field = true;
    if( p.curvmode == "true" )
        throw std::runtime_error( "curvmode : true is not possible in thermal code!");
    else if( p.curvmode == "low beta")
    {
        if( reversed_field)
            curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag, -1);
        else
            curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag, +1);
        dg::assign( dg::evaluate(dg::zero, g), m_divCurvKappa);
    }
    else if( p.curvmode == "toroidal")
    {
        if( reversed_field)
        {
            curvNabla = dg::geo::createCurvatureNablaB(mag, -1);
            curvKappa = dg::geo::createCurvatureKappa(mag, -1);
            dg::assign(  dg::pullback(dg::geo::DivCurvatureKappa(mag, -1), g),
                m_divCurvKappa);
        }
        else
        {
            curvNabla = dg::geo::createCurvatureNablaB(mag, +1);
            curvKappa = dg::geo::createCurvatureKappa(mag, +1);
            dg::assign(  dg::pullback(dg::geo::DivCurvatureKappa(mag, +1), g),
                m_divCurvKappa);
        }
    }
    else
        throw std::runtime_error( "Warning! curvmode value '"+p.curvmode+"' not recognized!! I don't know what to do! I exit!\n");
    dg::pushForward(curvNabla.x(), curvNabla.y(), curvNabla.z(),
        m_curvNabla[0], m_curvNabla[1], m_temp0, g);
    dg::pushForward(curvKappa.x(), curvKappa.y(), curvKappa.z(),
        m_curvKappa[0], m_curvKappa[1], m_temp0, g);
    dg::SparseTensor<Container> metric = g.metric();
    m_detg = dg::tensor::volume( metric);
    m_b_2 = m_temp1 = m_temp2 = m_temp3 = m_temp0;
    dg::assign(  dg::pullback(dg::geo::Divb(mag), g), m_divb);
    dg::assign(  dg::pullback(dg::geo::InvB(mag), g), m_b_2);
    // in PerpDynamics we take EhatPhi = 1,1,+/- 1/R, which leads to ephi_varphi / sqrt(g) / B = \pm 1/R
    dg::blas1::axpby( reversed_field ? -1. : +1., m_b_2, 0., m_b_2);
    m_bphi = dg::evaluate( dg::cooX3d, g); // R
    if( reversed_field)
        dg::blas1::scal( m_bphi, -1.);
    // Grad Ln B covariant components
    m_gradLnB[0] = dg::pullback( dg::geo::BR(mag), g);
    m_gradLnB[1] = dg::pullback( dg::geo::BZ(mag), g);
    dg::assign(  dg::pullback(dg::geo::InvB(mag), g), m_temp0);
    dg::blas1::pointwiseDot( m_gradLnB[0], m_temp0, m_gradLnB[0]);
    dg::blas1::pointwiseDot( m_gradLnB[1], m_temp0, m_gradLnB[1]);

    // Diffusion operators
    m_lapperp.construct ( g, p.bcx, p.bcy, p.diff_dir);
}

template<class Grid, class IMatrix, class Matrix, class Container>
void PerpDynamics<Grid, IMatrix, Matrix, Container>::update_derivatives(
    const Container& apar,
    Container& dxapar, Container& dyapar,
    const std::array<std::vector<Container>,6>& y,
    std::map<std::string, std::vector<Container>>& q
)
{
    dg::blas2::symv( m_dx_A, apar, dxapar);
    dg::blas2::symv( m_dy_A, apar, dyapar);
    for( unsigned s=0; s<m_p.num_species; s++)
    {
        dg::blas2::symv( m_dxF, y[0][s], q.at("dxF N")[s]);
        dg::blas2::symv( m_dxF, y[1][s], q.at("dxF Pperp")[s]);
        dg::blas2::symv( m_dxF, y[2][s], q.at("dxF Ppara")[s]);

        dg::blas2::symv( m_dxB, y[0][s], q.at("dxB N")[s]);
        dg::blas2::symv( m_dxB, y[1][s], q.at("dxB Pperp")[s]);
        dg::blas2::symv( m_dxB, y[2][s], q.at("dxB Ppara")[s]);

        dg::blas2::symv( m_dyF, y[0][s], q.at("dyF N")[s]);
        dg::blas2::symv( m_dyF, y[1][s], q.at("dyF Pperp")[s]);
        dg::blas2::symv( m_dyF, y[2][s], q.at("dyF Ppara")[s]);

        dg::blas2::symv( m_dyB, y[0][s], q.at("dyB N")[s]);
        dg::blas2::symv( m_dyB, y[1][s], q.at("dyB Pperp")[s]);
        dg::blas2::symv( m_dyB, y[2][s], q.at("dyB Ppara")[s]);

        dg::blas2::symv( m_dxC, q.at("Tperp")[s], q.at("dx Tperp")[s]);
        dg::blas2::symv( m_dxC, q.at("Tpara")[s], q.at("dx Tpara")[s]);

        dg::blas2::symv( m_dyC, q.at("Tperp")[s], q.at("dy Tperp")[s]);
        dg::blas2::symv( m_dyC, q.at("Tpara")[s], q.at("dy Tpara")[s]);

        dg::blas2::symv( m_dx_P, q.at("Psi0")[s], q.at("dx Psi0")[s]);
        dg::blas2::symv( m_dx_P, q.at("Psi1")[s], q.at("dx Psi1")[s]);
        dg::blas2::symv( m_dx_P, q.at("Psi2")[s], q.at("dx Psi2")[s]);

        dg::blas2::symv( m_dy_P, q.at("Psi0")[s], q.at("dy Psi0")[s]);
        dg::blas2::symv( m_dy_P, q.at("Psi1")[s], q.at("dy Psi1")[s]);
        dg::blas2::symv( m_dy_P, q.at("Psi2")[s], q.at("dy Psi2")[s]);

        dg::blas2::symv( m_dxC, q.at("U")[s],     q.at("dx U")[s] );
        dg::blas2::symv( m_dxC, q.at("Uperp")[s], q.at("dx Uperp")[s] );
        dg::blas2::symv( m_dxC, q.at("Upara")[s], q.at("dx Upara")[s] );

        dg::blas2::symv( m_dyC, q.at("U")[s],     q.at("dy U")[s] );
        dg::blas2::symv( m_dyC, q.at("Uperp")[s], q.at("dy Uperp")[s] );
        dg::blas2::symv( m_dyC, q.at("Upara")[s], q.at("dy Upara")[s] );
    }
}
template<class Grid, class IMatrix, class Matrix, class Container>
void PerpDynamics<Grid, IMatrix, Matrix, Container>::update_STderivatives(
    const Container& aparST,
    Container& dxaparST, Container& dyaparST,
    const std::array<std::vector<Container>,6>& y,
    std::map<std::string, std::vector<Container>>& q
)
{
    dg::blas2::symv( m_dx_A, aparST, dxaparST);
    dg::blas2::symv( m_dy_A, aparST, dyaparST);
    for( unsigned s=0; s<m_p.num_species; s++)
    {
        dg::blas2::symv( m_dxC, q.at("ST N")[s],     q.at("ST dx N")[s]);
        dg::blas2::symv( m_dxC, q.at("ST Tperp")[s], q.at("ST dx Tperp")[s]);
        dg::blas2::symv( m_dxC, q.at("ST Tpara")[s], q.at("ST dx Tpara")[s]);
        dg::blas2::symv( m_dxC, q.at("ST Psi0")[s],  q.at("ST dx Psi0")[s]);
        dg::blas2::symv( m_dxC, q.at("ST Psi1")[s],  q.at("ST dx Psi1")[s]);
        dg::blas2::symv( m_dxC, q.at("ST Psi2")[s],  q.at("ST dx Psi2")[s]);
        dg::blas2::symv( m_dxC, q.at("ST Psi3")[s],  q.at("ST dx Psi3")[s]);

        dg::blas2::symv( m_dyC, q.at("ST N")[s],     q.at("ST dy N")[s]);
        dg::blas2::symv( m_dyC, q.at("ST Tperp")[s], q.at("ST dy Tperp")[s]);
        dg::blas2::symv( m_dyC, q.at("ST Tpara")[s], q.at("ST dy Tpara")[s]);
        dg::blas2::symv( m_dyC, q.at("ST Psi0")[s],  q.at("ST dy Psi0")[s]);
        dg::blas2::symv( m_dyC, q.at("ST Psi1")[s],  q.at("ST dy Psi1")[s]);
        dg::blas2::symv( m_dyC, q.at("ST Psi2")[s],  q.at("ST dy Psi2")[s]);
        dg::blas2::symv( m_dyC, q.at("ST Psi3")[s],  q.at("ST dy Psi3")[s]);

        dg::blas2::symv( m_dxF, q.at("ST U")[s], q.at("ST dxF U")[s]);
        dg::blas2::symv( m_dxF, y[4][s],         q.at("ST dxF Qperp")[s]);
        dg::blas2::symv( m_dxF, y[5][s],         q.at("ST dxF Qpara")[s]);

        dg::blas2::symv( m_dxB, q.at("ST U")[s], q.at("ST dxB U")[s]);
        dg::blas2::symv( m_dxB, y[4][s],         q.at("ST dxB Qperp")[s]);
        dg::blas2::symv( m_dxB, y[5][s],         q.at("ST dxB Qpara")[s]);

        dg::blas2::symv( m_dyF, q.at("ST U")[s], q.at("ST dyF U")[s]);
        dg::blas2::symv( m_dyF, y[4][s],         q.at("ST dyF Qperp")[s]);
        dg::blas2::symv( m_dyF, y[5][s],         q.at("ST dyF Qpara")[s]);

        dg::blas2::symv( m_dyB, q.at("ST U")[s], q.at("ST dyB U")[s]);
        dg::blas2::symv( m_dyB, y[4][s],         q.at("ST dyB Qperp")[s]);
        dg::blas2::symv( m_dyB, y[5][s],         q.at("ST dyB Qpara")[s]);
    }
}

template<class Grid, class IMatrix, class Matrix, class Container>
void PerpDynamics<Grid, IMatrix, Matrix, Container>::add_densities_advection(
    unsigned s,
    const Container& apar, const Container& dxapar, const Container& dyapar,
    const std::map<std::string, std::vector<Container>>& q,
    std::array<std::vector<Container>,6>& yp
) const
{
    /////////////////////////////////////////////////////////////////
    double mu = m_p.mu[s], z = m_p.z[s], beta = m_p.beta;
    dg::blas1::subroutine( [mu, z, beta] DG_DEVICE (
            double N,
            double dxFN,     double dyFN,
            double dxBN,     double dyBN,
            double dxFPperp, double dyFPperp,
            double dxBPperp, double dyBPperp,
            double dxFPpara, double dyFPpara,
            double dxBPpara, double dyBPpara,
            double A,     double dxA,     double dyA,
            double Tperp, double dxTperp, double dyTperp,
            double Tpara, double dxTpara, double dyTpara,
            double U,     double dxU,     double dyU,
            double Uperp, double dxUperp, double dyUperp,
            double Upara, double dxUpara, double dyUpara,
                       double dxG0, double dyG0,
            double G1, double dxG1, double dyG1,
            double G2, double dxG2, double dyG2,
            double curvNablaX, double curvNablaY,
            double curvKappaX, double curvKappaY,
            double gradLnBX, double gradLnBY,
            double divCurvKappa, double b_2, double divb,
            double& dtN, double& dtPperp, double& dtPpara
        )
    {
        double E0X = dxG0 - G1*(dxTperp/Tperp - gradLnBX);
        double E0Y = dyG0 - G1*(dyTperp/Tperp - gradLnBY);
        double E1X = dxG1 - (G2-G1)*(dxTperp/Tperp - gradLnBX);
        double E1Y = dyG1 - (G2-G1)*(dyTperp/Tperp - gradLnBY);
        double bpX = 0., bpY = 0., divbp = 0.;
        if( beta != 0)
        {
            bpX = A * curvKappaX + (   dyA*b_2);
            bpY = A * curvKappaY + ( - dxA*b_2);
            divbp = A*divCurvKappa - curvNablaX*dxA - curvNablaY*dyA;
        }
        double vX = U * bpX + ( - b_2*E0Y) + Tperp/z *curvNablaX + (Tpara + mu*U*U)/z*curvKappaX;
        double vY = U * bpY + (   b_2*E0X) + Tperp/z *curvNablaY + (Tpara + mu*U*U)/z*curvKappaY;
        dtN += ( vX > 0 ) ? -vX*dxBN : -vX*dxFN;
        dtN += ( vY > 0 ) ? -vY*dyBN : -vY*dyFN;
        // Pperp
        vX = (U+Uperp)*bpX + ( - b_2*(E0Y+E1Y)) + 2*Tperp/z *curvNablaX
            + (Tpara + 2*mu*Uperp*U + mu*U*U)/z*curvKappaX;
        vY = (U+Uperp)*bpY + (   b_2*(E0X+E1X)) + 2*Tperp/z *curvNablaY
            + (Tpara + 2*mu*Uperp*U + mu*U*U)/z*curvKappaY;
        dtPperp += ( vX > 0 ) ? -vX*dxBPperp : -vX*dxFPperp;
        dtPperp += ( vY > 0 ) ? -vY*dyBPperp : -vY*dyFPperp;
        // Ppara
        vX = (U+Upara)*bpX + ( - b_2*E0Y) + Tperp/z *curvNablaX
            + (3*Tpara + 2*mu*Upara*U + mu*U*U)/z*curvKappaX;
        vY = (U+Upara)*bpY + (   b_2*E0X) + Tperp/z *curvNablaY
            + (3*Tpara + 2*mu*Upara*U + mu*U*U)/z*curvKappaY;
        dtPpara += ( vX > 0 ) ? -vX*dxBPpara : -vX*dxFPpara;
        dtPpara += ( vY > 0 ) ? -vY*dyBPpara : -vY*dyFPpara;

        dtN     -= N*( U*divbp + bpX*dxU + bpY*dyU );
        dtPperp -= N*Tperp*( (U+Uperp)*divbp + bpX*(dxU+dxUperp) + bpY*(dyU+dyUperp) );
        dtPpara -= N*Tpara*( (U+Upara)*divbp + bpX*(dxU+dxUpara) + bpY*(dyU+dyUpara) );

        dtN     -= N/z*(Tpara + mu*U*U - Tperp)*divCurvKappa;
        dtPperp -= N*Tperp/z*(Tpara + 2*mu*U*Uperp +  mu*U*U - 2*Tperp)*divCurvKappa;
        dtPpara -= N*Tpara/z*(3*Tpara + 2*mu*U*Upara +  mu*U*U - Tperp)*divCurvKappa;

        dtN     -= N/z*( curvKappaX*(dxTpara + 2*mu*U*dxU) 
                       + curvKappaY*(dyTpara + 2*mu*U*dyU));
        dtPperp -= N*Tperp/z*( curvKappaX*(dxTpara + 2*mu*U*dxUperp + 2*mu*(U+Uperp)*dxU)
                             + curvKappaY*(dyTpara + 2*mu*U*dyUperp + 2*mu*(U+Uperp)*dyU));
        dtPpara -= N*Tpara/z*( curvKappaX*(3*dxTpara + 2*mu*U*dxUpara + 2*mu*(U+Upara)*dxU)
                             + curvKappaY*(3*dyTpara + 2*mu*U*dyUpara + 2*mu*(U+Upara)*dyU));

        dtN     -= N/z*          (curvNablaX*dxTperp + curvNablaY*dyTperp);
        dtPperp -= N*Tperp/z* 2.*(curvNablaX*dxTperp + curvNablaY*dyTperp);
        dtPpara -= N*Tpara/z*    (curvNablaX*dxTperp + curvNablaY*dyTperp);

        double divuE0 =  (curvNablaX+curvKappaX)*E0X
                        +(curvNablaY+curvKappaY)*E0Y
                        +b_2* ( dxG1*(dyTperp/Tperp - gradLnBY)
                               -dyG1*(dxTperp/Tperp - gradLnBX));
        double divuE1 =  (curvNablaX+curvKappaX)*E1X
                        +(curvNablaY+curvKappaY)*E1Y
                        +b_2* ( (dxG2-dxG1)*(dyTperp/Tperp - gradLnBY)
                               -(dyG2-dyG1)*(dxTperp/Tperp - gradLnBX));
        dtN     -= N*divuE0;
        dtPperp -= N*Tperp*(divuE0 + divuE1);
        dtPpara -= N*Tpara*divuE0;


        // F1
        dtPperp -=  N*Tperp*( (divb + divbp)*(Uperp+U)
                   + (Tpara + 2*mu*U*Uperp + mu*U*U)*divCurvKappa/z
                   + curvNablaX*(E0X+E1X) + curvNablaY*(E0Y+E1Y) );
        // F2
        dtPpara += 2*N*Tperp*( (divb + divbp)*Uperp + (Tpara + mu*U*Uperp)*divCurvKappa/z);
        dtPpara -= 2*N*( z*Uperp*(bpX*E1X + bpY*E1Y)
                         + Tpara*(curvKappaX*E0X+curvKappaY*E0Y)
                         + mu*U*Uperp*(curvKappaX*E1X + curvKappaY*E1Y));
        dtPpara -=2*N*(bpX*Tpara
                    + mu/z*(Tpara*Upara + 2*U*Tpara)*curvKappaX
                    + mu/z*Tperp*Uperp*curvNablaX
                    -mu*Uperp*b_2*E1Y)*dxU;
        dtPpara -=2*N*(bpY*Tpara
                    + mu/z*(Tpara*Upara + 2*U*Tpara)*curvKappaY
                    + mu/z*Tperp*Uperp*curvNablaY
                    +mu*Uperp*b_2*E1X)*dyU;
    },
        q.at("N")[s],
        q.at("dxF N")[s],     q.at("dyF N")[s],
        q.at("dxB N")[s],     q.at("dyB N")[s],
        q.at("dxF Pperp")[s], q.at("dyF Pperp")[s],
        q.at("dxB Pperp")[s], q.at("dyB Pperp")[s],
        q.at("dxF Ppara")[s], q.at("dyF Ppara")[s],
        q.at("dxB Ppara")[s], q.at("dyB Ppara")[s],
        apar, dxapar, dyapar,
        q.at("Tperp")[s], q.at("dx Tperp")[s], q.at("dy Tperp")[s],
        q.at("Tpara")[s], q.at("dx Tpara")[s], q.at("dy Tpara")[s],
        q.at("U")[s],     q.at("dx U")[s],     q.at("dy U")[s],
        q.at("Uperp")[s], q.at("dx Uperp")[s], q.at("dy Uperp")[s],
        q.at("Upara")[s], q.at("dx Upara")[s], q.at("dy Upara")[s],
                         q.at("dx Psi0")[s], q.at("dy Psi0")[s],
        q.at("Psi1")[s], q.at("dx Psi1")[s], q.at("dy Psi1")[s],
        q.at("Psi2")[s], q.at("dx Psi2")[s], q.at("dy Psi2")[s],
        m_curvNabla[0], m_curvNabla[1],
        m_curvKappa[0], m_curvKappa[1],
        m_gradLnB[0], m_gradLnB[1],
        m_divCurvKappa, m_b_2, m_divb,
        yp[0][s], yp[1][s], yp[2][s]
    );
}

template<class Grid, class IMatrix, class Matrix, class Container>
void PerpDynamics<Grid, IMatrix, Matrix, Container>::add_velocities_advection(
    unsigned s,
    const Container& aparST,
    const Container& dxaparST,
    const Container& dyaparST,
    const std::map<std::string, std::vector<Container>>& q,
    std::array<std::vector<Container>,6>& yp
) const
{
    double mu = m_p.mu[s], z = m_p.z[s], beta = m_p.beta;
    dg::blas1::subroutine( [mu, z, beta] DG_DEVICE (
            double U,
            double dxFU,     double dyFU,
            double dxBU,     double dyBU,
            double dxFQperp, double dyFQperp,
            double dxBQperp, double dyBQperp,
            double dxFQpara, double dyFQpara,
            double dxBQpara, double dyBQpara,
            double N, double dxN, double dyN,
            double Tperp, double dxTperp, double dyTperp,
            double Tpara, double dxTpara, double dyTpara,
            double A, double dxA, double dyA,
            double Uperp, double Upara,
                       double dxG0, double dyG0,
            double G1, double dxG1, double dyG1,
            double G2, double dxG2, double dyG2,
            double G3, double dxG3, double dyG3,
            double curvNablaX, double curvNablaY,
            double curvKappaX, double curvKappaY,
            double gradLnBX, double gradLnBY,
            double divCurvKappa, double b_2, double divb,
            double& dtU, double& dtQperp, double& dtQpara
        )
    {
        double E0X = dxG0 - G1*(dxTperp/Tperp - gradLnBX);
        double E0Y = dyG0 - G1*(dyTperp/Tperp - gradLnBY);
        double E1X = dxG1 - (G2-G1)*(dxTperp/Tperp - gradLnBX);
        double E1Y = dyG1 - (G2-G1)*(dyTperp/Tperp - gradLnBY);
        double E2X = dxG2 - (G3-2*G2)*(dxTperp/Tperp - gradLnBX);
        double E2Y = dyG2 - (G3-2*G2)*(dyTperp/Tperp - gradLnBY);
        double bpX = 0., bpY = 0., divbp = 0.;
        if( beta != 0)
        {
            bpX = A * curvKappaX + (   dyA*b_2);
            bpY = A * curvKappaY + ( - dxA*b_2);
            divbp = A*divCurvKappa - curvNablaX*dxA - curvNablaY*dyA;
        }
        // U
        double vX = U * bpX + ( - b_2*E0Y) + Tperp/z *curvNablaX + (3*Tpara + mu*U*U)/z*curvKappaX;
        double vY = U * bpY + (   b_2*E0X) + Tperp/z *curvNablaY + (3*Tpara + mu*U*U)/z*curvKappaY;
        dtU += ( vX > 0 ) ? -vX*dxBU : -vX*dxFU;
        dtU += ( vY > 0 ) ? -vY*dyBU : -vY*dyFU;
        // Qperp
        vX = U * bpX + ( - b_2*(E0Y+E2Y)) + 3*Tperp/z *curvNablaX + (3*Tpara + mu*U*U)/z*curvKappaX;
        vY = U * bpY + (   b_2*(E0X+E2X)) + 3*Tperp/z *curvNablaY + (3*Tpara + mu*U*U)/z*curvKappaY;
        dtQperp += ( vX > 0 ) ? -vX*dxBQperp : -vX*dxFQperp;
        dtQperp += ( vY > 0 ) ? -vY*dyBQperp : -vY*dyFQperp;
        // Qpara
        vX = U * bpX + ( - b_2*E0Y) + Tperp/z *curvNablaX + (7*Tpara + mu*U*U)/z*curvKappaX;
        vY = U * bpY + (   b_2*E0X) + Tperp/z *curvNablaY + (7*Tpara + mu*U*U)/z*curvKappaY;
        dtQpara += ( vX > 0 ) ? -vX*dxBQpara : -vX*dxFQpara;
        dtQpara += ( vY > 0 ) ? -vY*dyBQpara : -vY*dyFQpara;

        dtU -= 2/z*( U*Tpara*divCurvKappa + U * (curvKappaX*dxTpara + curvKappaY*dyTpara)
                        + U*Tpara/N *(curvKappaX*dxN + curvKappaY*dyN));

        double divuE1 =  (curvNablaX+curvKappaX)*E1X
                        +(curvNablaY+curvKappaY)*E1Y
                        +b_2* ( (dxG2-dxG1)*(dyTperp/Tperp - gradLnBY)
                               -(dyG2-dyG1)*(dxTperp/Tperp - gradLnBX));
        dtU -=    Tpara/mu*(divb+divbp)
                + 1./mu*(bpX*dxTpara + bpY*dyTpara)
                + Tpara/mu/N*(bpX*dxN + bpY*dyN)
                + Tpara*Upara/z*divCurvKappa
                + 1./z/N*( curvKappaX * dxFQpara + curvKappaY * dyFQpara)
                - Tperp/z*Uperp*divCurvKappa
                + 1./z/N*( curvNablaX * dxFQperp + curvNablaY * dyFQperp)
                + Uperp*divuE1
                + 1./N/Tperp*b_2*(E1X*dyFQperp - E1Y*dxFQperp)
                - Uperp/Tperp*b_2*(E1X*dyTperp - E1Y*dxTperp);
	// F_U
        dtU += Tperp/mu*(divb+divbp)
                + Tperp/z*(Uperp+U)*divCurvKappa
                - z/mu*(bpX*E0X+bpY*E0Y)
                - U*(    curvKappaX*E0X + curvKappaY*E0Y)
                - Uperp*(curvKappaX*E1X + curvKappaY*E1Y);

        double divuE0 =  (curvNablaX+curvKappaX)*E0X
                        +(curvNablaY+curvKappaY)*E0Y
                        +b_2* ( dxG1*(dyTperp/Tperp - gradLnBY)
                               -dyG1*(dxTperp/Tperp - gradLnBX));
        double divuE2 =  (curvNablaX+curvKappaX)*E2X
                        +(curvNablaY+curvKappaY)*E2Y
                        +b_2* ( (dxG3-2*dxG2)*(dyTperp/Tperp - gradLnBY)
                               -(dyG3-2*dyG2)*(dxTperp/Tperp - gradLnBX));
        double dxU = 0.5*(dxFU + dxBU);
        double dyU = 0.5*(dyFU + dyBU);
        dtQperp -= N*Tperp*Uperp*(
                        U*divbp + bpX*dxU + bpY*dyU
                        +1/z*(3*Tpara + mu*U*U)*divCurvKappa
                        +1/z*(3*(curvKappaX*dxTpara+curvKappaY*dyTpara)+2*mu*U*(curvKappaX*dxU+curvKappaY*dyU))
                        -3/z*Tperp*divCurvKappa
                        +3/z*(curvNablaX*dxTperp + curvNablaY*dyTperp)
                        +divuE0+divuE2);
        dtQpara -= N*Tpara*Upara*(
                        U*divbp + bpX*dxU + bpY*dyU
                        +1/z*(7*Tpara + mu*U*U)*divCurvKappa
                        +1/z*(7*(curvKappaX*dxTpara+curvKappaY*dyTpara)+2*mu*U*(curvKappaX*dxU+curvKappaY*dyU))
                        -1/z*Tperp*divCurvKappa
                        +1/z*(curvNablaX*dxTperp + curvNablaY*dyTperp)
                        +divuE0);
        // temperature transfer
        vX = N*Tpara/mu*bpX + 1/z*N*Tpara*(Upara+2*mu*U)*curvKappaX + 1/z*N*Tperp*Uperp*curvNablaX
            +N*Uperp*( -b_2*E1Y);
        vY = N*Tpara/mu*bpY + 1/z*N*Tpara*(Upara+2*mu*U)*curvKappaY + 1/z*N*Tperp*Uperp*curvNablaY
            +N*Uperp*(  b_2*E1X);
        dtQperp -=    vX*dxTperp + vY*dyTperp;
        dtQpara -= 3*(vX*dxTpara + vY*dyTpara);
        // velocity transfer
	dtQperp -= N*Tperp*(Uperp*bpX + 2*mu/z*U*Uperp*curvKappaX +
			1/z*Tperp*curvNablaX +( -b_2*E1Y))*dxU;
	dtQperp -= N*Tperp*(Uperp*bpY + 2*mu/z*U*Uperp*curvKappaY +
			1/z*Tperp*curvNablaY +(  b_2*E1X))*dyU;

        dtQpara -= 3*N*Tpara*(Upara*bpX + 2/z*(Tpara + mu*U*Upara)*curvKappaX)*dxU;
        dtQpara -= 3*N*Tpara*(Upara*bpY + 2/z*(Tpara + mu*U*Upara)*curvKappaY)*dyU;

        // Force terms
        dtQperp +=   N*Tperp*(
			((Tperp-Tpara)/mu-Uperp*U)*(divb+divbp)
                    	+1/z * ( 3*Uperp*(Tperp-Tpara) + U*(Tperp - 2*Tpara)
                            	-Tpara*Upara - mu*Uperp*U*U ) * divCurvKappa
                    	-z/mu*(bpX*E1X+bpY*E1Y)
                        -Uperp*(curvKappaX*(E0X+E2X)+curvKappaY*(E0Y+E2Y))
                        -U    *(curvKappaX*E1X + curvKappaY*E1Y)
                        -Uperp*(curvNablaX*(E0X+E1X+E2X) + curvNablaY*(E0Y+E1Y+E2Y))
		    );
        dtQpara += 3*N*Tpara*( (2/z*Tperp*Uperp + Tperp/z*Upara) * divCurvKappa
                        -  Upara*(curvKappaX*E0X + curvKappaY*E0Y)
                        -2*Uperp*(curvKappaX*E1X + curvKappaY*E1Y));

    },
        q.at("ST U")[s],
        q.at("ST dxF U")[s],     q.at("ST dyF U")[s],
        q.at("ST dxB U")[s],     q.at("ST dyB U")[s],
        q.at("ST dxF Qperp")[s], q.at("ST dyF Qperp")[s],
        q.at("ST dxB Qperp")[s], q.at("ST dyB Qperp")[s],
        q.at("ST dxF Qpara")[s], q.at("ST dyF Qpara")[s],
        q.at("ST dxB Qpara")[s], q.at("ST dyB Qpara")[s],
        q.at("ST N")[s],     q.at("ST dx N")[s],     q.at("ST dy N")[s],
        q.at("ST Tperp")[s], q.at("ST dx Tperp")[s], q.at("ST dy Tperp")[s],
        q.at("ST Tpara")[s], q.at("ST dx Tpara")[s], q.at("ST dy Tpara")[s],
        aparST, dxaparST, dyaparST,
        q.at("ST Uperp")[s], q.at("ST Upara")[s],
                            q.at("ST dx Psi0")[s], q.at("ST dy Psi0")[s],
        q.at("ST Psi1")[s], q.at("ST dx Psi1")[s], q.at("ST dy Psi1")[s],
        q.at("ST Psi2")[s], q.at("ST dx Psi2")[s], q.at("ST dy Psi2")[s],
        q.at("ST Psi3")[s], q.at("ST dx Psi3")[s], q.at("ST dy Psi3")[s],
        m_curvNabla[0], m_curvNabla[1],
        m_curvKappa[0], m_curvKappa[1],
        m_gradLnB[0], m_gradLnB[1],
        m_divCurvKappa, m_b_2, m_divb,
        yp[3][s], yp[4][s], yp[5][s]
    );
}
} //namespace thermal

