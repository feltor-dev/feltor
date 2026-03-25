#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "parameters.h"

namespace feltor
{

struct BPerp{
    //b_perp
    DG_DEVICE void operator()(double A,
        double d0A, double d1A, double d2A,
        double& bp0, double& bp1, double& bp2, //bperp
        double b_0,         double b_1,         double b_2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        bp0 = (b_2*d1A - b_1*d2A + A*curvKappa0);
        bp1 = (b_0*d2A - b_2*d0A + A*curvKappa1);
        bp2 = (b_1*d0A - b_0*d1A + A*curvKappa2);
    }
};
template< class Geometry, class Matrix, class Container >
struct PerpDynamics
{
    PerpDynamics( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField, dg::file::WrappedJsonValue);

    void update_derivatives(
        std::map<std::string, std::array<Container,2>>& q,
        const Container& apar, std::array<Container,3>& dA);
    void update_STderivatives(
        std::map<std::string, std::array<Container,2>>& q,
        const Container& aparST, std::array<Container,3>& dAST);

    void compute_density( double t,
        const std::map<std::string, std::array<Container,2>>& q,
        const Container& apar, const std::array<Container,3>& dA,
        std::array<Container,2>& densityDOT) const;
    void compute_velocity( double t,
        const std::map<std::string, std::array<Container,2>>& q,
        const Container& aparST, const std::array<Container,3>& dAST,
        std::array<Container,2>& velocityDOT) const;
    void compute_diffusiveN( double alpha, const Container& density,
            Container& temp0, Container& temp1, double beta, Container& result ) const
    {
        // density = full N
        // result = alpha Lambda_N + beta result
        if( m_p.nu_perp_n > 0)
        {
            dg::blas1::transform( density, temp0, dg::PLUS<double>(-m_p.nbc));
            for( unsigned s=0; s<m_p.diff_order; s++)
            {
                using std::swap;
                swap( temp0, temp1);
                dg::blas2::symv( 1., m_lapperpN, temp1, 0., temp0);
            }
            dg::blas1::axpby( -alpha*m_p.nu_perp_n, temp0, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
    }
    void compute_diffusiveU( double alpha, const Container& velocity,
            const Container& density,
            Container& temp0, Container& temp1, Container& temp2, Container& temp3, double beta, Container& result) const
    {
        // density = full N
        // result = alpha Lambda_U + beta result
        if( m_p.nu_perp_u > 0)
        {
            dg::blas1::copy( velocity, temp0);
            for( unsigned s=0; s<m_p.diff_order; s++)
            {
                using std::swap;
                swap( temp0, temp1);
                dg::blas2::symv( 1., m_lapperpU, temp1, 0., temp0);
            }
            if( !m_p.modify_diff)
                dg::blas1::pointwiseDivide( -alpha*m_p.nu_perp_u, temp0, density, beta, result);
            else
                dg::blas1::axpby( -alpha*m_p.nu_perp_u, temp0, beta, result);
        }
        else
            dg::blas1::scal( result, beta);
        double nu = m_p.nu_perp_n;
        if( m_p.modify_diff)
            nu += m_p.nu_perp_u;
        if( nu > 0 )
        {

            dg::blas1::transform( density, temp0, dg::PLUS<double>(-m_p.nbc));
            for( unsigned s=0; s<m_p.diff_order-1; s++)
            {
                using std::swap;
                swap( temp0, temp1);
                dg::blas2::symv( 1., m_lapperpN, temp1, 0., temp0);
            }

            // - v_x dx U
            if( m_p.diff_dir == dg::centered)
                dg::blas2::symv( m_dxC, temp0, temp1);
            else if( m_p.diff_dir == dg::forward)
                dg::blas2::symv( m_dxF_N, temp0, temp1);
            else
                dg::blas2::symv( m_dxB_N, temp0, temp1);
            dg::blas1::pointwiseDivide( -nu, temp1, density, 0., temp1);
            dg::blas2::symv( m_dxB_U, velocity, temp2);
            dg::blas2::symv( m_dxF_U, velocity, temp3);
            dg::blas1::evaluate( result, dg::minus_equals(), dg::UpwindProduct(),
                    temp1, temp2, temp3);
            // - v_y dy U
            if( m_p.diff_dir == dg::centered)
                dg::blas2::symv( m_dyC, temp0, temp1);
            else if( m_p.diff_dir == dg::forward)
                dg::blas2::symv( m_dyF_N, temp0, temp1);
            else
                dg::blas2::symv( m_dyB_N, temp0, temp1);
            dg::blas1::pointwiseDivide( -nu, temp1, density, 0., temp1);
            dg::blas2::symv( m_dyB_U, velocity, temp2);
            dg::blas2::symv( m_dyF_U, velocity, temp3);
            dg::blas1::evaluate( result, dg::minus_equals(), dg::UpwindProduct(),
                    temp1, temp2, temp3);
        }
    }
    // y = alpha*(-Delta) X + beta * y
    void compute_lapMperpN (double alpha, const Container& density, Container& temp0, double beta, Container& result) const
    {
        // positive Laplacian
        dg::blas1::transform( density, temp0, dg::PLUS<double>(-m_p.nbc));
        dg::blas2::symv( alpha, m_lapperpN, temp0, beta, result);
    }
    void compute_lapMperpU (const Container& velocity, Container& result) const
    {
        dg::blas2::symv( m_lapperpU, velocity, result);
    }
    const dg::SparseTensor<Container>& projection() const{
        return m_hh;
    }
    const std::array<Container, 3> & curv () const {
        return m_curv;
    }
    const std::array<Container, 3> & curvKappa () const {
        return m_curvKappa;
    }
    const Container& divCurvKappa() const {
        return m_divCurvKappa;
    }
    const Container& bphi( ) const { return m_bphi; }
    const Container& binv( ) const { return m_binv; }
    const Container& divb( ) const { return m_divb; }
    //volume with dG weights
    const Container& vol3d() const { return m_lapperpN.weights();}
    const Container& weights() const { return m_lapperpN.weights();}
    //bhat / sqrt{g} / B
    const std::array<Container, 3> & bhatgB () const {
        // covariant components
        return m_b;
    }
    void compute_bperp( const Container& apar, const std::array<Container,3>& dA, std::array<Container,3>& bperp)
    {
        dg::blas1::subroutine( BPerp(), apar,
            dA[0], dA[1], dA[2],
            bperp[0], bperp[1], bperp[2], // bperp on output
            m_b[0], m_b[1], m_b[2],
            m_curvKappa[0], m_curvKappa[1], m_curvKappa[2]
        );
    }

    // Compute divergence using centered derivatives
    // note that no matter how divergence is computed you always loose one order
    // unless the polarisation term or the Laplacian of N,U is computed
    // then the correct direction must be chosen
    // prefactor cannot alias result!!
    // Div ( f v)
    template<class Container2>
    void centered_div( const Container2& prefactor,
            const std::array<Container, 3>& contra_vec,
            Container& temp0, Container& result) const
    {
        dg::blas1::pointwiseDot( 1., prefactor, m_detg, contra_vec[0], 0., temp0);
        dg::blas2::symv( m_dxC, temp0, result);
        dg::blas1::pointwiseDot( 1., prefactor, m_detg, contra_vec[1], 0., temp0);
        dg::blas2::symv( 1., m_dyC, temp0, 1., result);
        if( m_compute_in_3d)
        {
            dg::blas1::pointwiseDot( 1., prefactor, m_detg, contra_vec[2], 0., temp0);
            dg::blas2::symv( 1., m_dz, temp0, 1., result);
        }
        dg::blas1::pointwiseDivide( 1., result, m_detg, 0., result);
    }
    void centered_v_dot_nabla( const std::array<Container, 3>& contra_vec,
            const Container& f, Container& temp1, Container& result) const
    {
        dg::blas2::symv( m_dxC, f, temp1);
        dg::blas1::pointwiseDot( contra_vec[0], temp1, result);
        dg::blas2::symv( m_dyC, f, temp1);
        dg::blas1::pointwiseDot( 1., contra_vec[1], temp1, 1., result);
        if( m_compute_in_3d)
        {
            dg::blas2::symv( m_dz, f, temp1);
            dg::blas1::pointwiseDot( 1., contra_vec[2], temp1, 1., result);
        }
    }
    void compute_gradN( const Container& in, std::array<Container,3>& gradS) const{
        dg::blas2::symv( m_dxF_N, in, gradS[0]);
        dg::blas2::symv( m_dyF_N, in, gradS[1]);
        if(m_compute_in_3d)dg::blas2::symv( m_dz, in, gradS[2]);
    }

    const Matrix& dxC() const { return m_dxC;}
    const Matrix& dyC() const { return m_dyC;}
    const Matrix& dzC() const { return m_dz;}
    private:
    //these should be considered const // m_curv is full curvature
    std::array<Container,3> m_curv, m_curvKappa, m_b; //m_b is bhat/ sqrt(g) / B
    Container m_divCurvKappa;
    Container m_bphi, m_binv, m_divb, m_detg;

    Matrix m_dxF_N, m_dxB_N, m_dxC_N, m_dxF_U, m_dxB_U, m_dxC_U, m_dx_P, m_dx_A;
    Matrix m_dyF_N, m_dyB_N, m_dyC_N, m_dyF_U, m_dyB_U, m_dyC_U, m_dy_P, m_dy_A, m_dz;
    Matrix m_dxC, m_dyC;
    dg::Elliptic3d< Geometry, Matrix, Container> m_lapperpN, m_lapperpU;
    dg::SparseTensor<Container> m_hh;

    Container m_temp;

    const feltor::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    bool m_reversed_field = false, m_compute_in_3d = true;
};

template<class Grid, class Matrix, class Container>
PerpDynamics<Grid, Matrix, Container>::PerpDynamics( const Grid& g,
    feltor::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue js
    ):
    m_dxF_N( dg::create::dx( g, p.bcxN, dg::forward) ),
    m_dxB_N( dg::create::dx( g, p.bcxN, dg::backward) ),
    m_dxC_N( dg::create::dx( g, p.bcxN, dg::centered) ),
    m_dxF_U( dg::create::dx( g, p.bcxU, dg::forward) ),
    m_dxB_U( dg::create::dx( g, p.bcxU, dg::backward) ),
    m_dxC_U( dg::create::dx( g, p.bcxU, dg::centered) ),
    m_dx_P(  dg::create::dx( g, p.bcxP, p.pol_dir) ),
    m_dx_A(  dg::create::dx( g, p.bcxA, p.pol_dir) ),
    m_dyF_N( dg::create::dy( g, p.bcyN, dg::forward) ),
    m_dyB_N( dg::create::dy( g, p.bcyN, dg::backward) ),
    m_dyC_N( dg::create::dy( g, p.bcyN, dg::centered) ),
    m_dyF_U( dg::create::dy( g, p.bcyU, dg::forward) ),
    m_dyB_U( dg::create::dy( g, p.bcyU, dg::backward) ),
    m_dyC_U( dg::create::dy( g, p.bcyU, dg::centered) ),
    m_dy_P(  dg::create::dy( g, p.bcyP, p.pol_dir) ),
    m_dy_A(  dg::create::dy( g, p.bcyA, p.pol_dir) ),
    m_dz( dg::create::dz( g, dg::PER) ),
    m_dxC(   dg::create::dx( g, dg::NEU, dg::centered) ), // for divergence
    m_dyC(   dg::create::dy( g, dg::NEU, dg::centered) ), // for divergence
    m_p(p), m_js(js)
{
    dg::assign( dg::evaluate( dg::zero, g), m_temp );
    //due to the various approximations bhat and mag not always correspond
    dg::geo::CylindricalVectorLvl0 curvNabla, curvKappa;
    m_reversed_field = false;
    if( mag.ipol()( g.x0(), g.y0()) < 0)
        m_reversed_field = true;
    if( p.curvmode == "true" )
    {
        curvNabla = dg::geo::createTrueCurvatureNablaB(mag);
        curvKappa = dg::geo::createTrueCurvatureKappa(mag);
        dg::assign(  dg::pullback(dg::geo::TrueDivCurvatureKappa(mag), g),
            m_divCurvKappa);
    }
    else if( p.curvmode == "low beta")
    {
        if( m_reversed_field)
            curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag, -1);
        else
            curvNabla = curvKappa = dg::geo::createCurvatureNablaB(mag, +1);
        dg::assign( dg::evaluate(dg::zero, g), m_divCurvKappa);
    }
    else if( p.curvmode == "toroidal")
    {
        if( m_reversed_field)
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
        m_curv[0], m_curv[1], m_curv[2], g);
    dg::pushForward(curvKappa.x(), curvKappa.y(), curvKappa.z(),
        m_curvKappa[0], m_curvKappa[1], m_curvKappa[2], g);
    dg::blas1::axpby( 1., m_curvKappa, 1., m_curv);
    dg::assign(  dg::pullback(dg::geo::InvB(mag), g), m_binv);
    dg::assign(  dg::pullback(dg::geo::Divb(mag), g), m_divb);

    // in Poisson we take EPhi except for the true curvmode
    auto bhat = dg::geo::createEPhi(+1);
    if( p.curvmode == "true")
        bhat = dg::geo::createBHat(mag);
    else if( m_reversed_field)
        bhat = dg::geo::createEPhi(-1);
    dg::pushForward(bhat.x(), bhat.y(), bhat.z(), m_b[0], m_b[1], m_b[2], g);
    dg::SparseTensor<Container> metric = g.metric();
    // make bhat covariant:
    dg::tensor::inv_multiply3d( metric, m_b[0], m_b[1], m_b[2],
                                        m_b[0], m_b[1], m_b[2]);
    dg::assign( m_b[2], m_bphi); //save bphi for momentum conservation
    m_detg = dg::tensor::volume( metric);
    dg::blas1::pointwiseDivide( m_binv, m_detg, m_temp); //1/B/detg
    for( int i=0; i<3; i++)
        dg::blas1::pointwiseDot( m_temp, m_b[i], m_b[i]); //b_i/detg/B
    m_hh = dg::geo::createProjectionTensor( bhat, g);
    m_lapperpN.construct ( g, p.bcxN, p.bcyN, dg::PER,  p.diff_dir),
    m_lapperpU.construct ( g, p.bcxU, p.bcyU, dg::PER,  p.diff_dir),
    m_lapperpN.set_chi( m_hh);
    m_lapperpU.set_chi( m_hh);
    if( (p.curvmode == "true") && (p.symmetric == false))
        m_compute_in_3d = true;
    else
    {
        m_compute_in_3d = false;
        m_lapperpN.set_compute_in_2d(true);
        m_lapperpU.set_compute_in_2d(true);
    }
}

template<class Geometry, class Matrix, class Container>
void PerpDynamics<Geometry, Matrix, Container>::update_derivatives(
    std::map<std::string, std::array<Container,2>>& q,
    const Container& apar, std::array<Container,3>& dA)
{
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        //First compute forward and backward derivatives for upwind scheme
        dg::blas1::transform( q.at("N")[i], m_temp, dg::PLUS<double>(-m_p.nbc));
        dg::blas2::symv( m_dxF_N, m_temp, q.at("dxF N")[i]);
        dg::blas2::symv( m_dyF_N, m_temp, q.at("dyF N")[i]);
        dg::blas2::symv( m_dxB_N, m_temp, q.at("dxB N")[i]);
        dg::blas2::symv( m_dyB_N, m_temp, q.at("dyB N")[i]);
        if(m_compute_in_3d) dg::blas2::symv( m_dz, m_temp, q.at( "dzF N")[i]);
        if(m_compute_in_3d) dg::blas2::symv( m_dz, m_temp, q.at( "dzB N")[i]);
        if( m_p.diff_dir == dg::forward)
        {
            dg::blas2::symv( m_dxF_U, q.at("U")[i], q.at("dx U")[i]);
            dg::blas2::symv( m_dyF_U, q.at("U")[i], q.at("dy U")[i]);
        }
        else if( m_p.diff_dir == dg::backward)
        {
            dg::blas2::symv( m_dxB_U, q.at("U")[i], q.at("dx U")[i]);
            dg::blas2::symv( m_dyB_U, q.at("U")[i], q.at("dy U")[i]);
        }
        else
        {
            dg::blas2::symv( m_dxC_U, q.at("U")[i], q.at("dx U")[i]);
            dg::blas2::symv( m_dyC_U, q.at("U")[i], q.at("dy U")[i]);
        }
        if(m_compute_in_3d) dg::blas2::symv( m_dz, q.at("U")[i], q.at( "dz U")[i]);

        dg::blas2::symv( m_dx_P, q.at("Psi")[i], q.at("dx Psi")[i]);
        dg::blas2::symv( m_dy_P, q.at("Psi")[i], q.at("dy Psi")[i]);
        if( m_compute_in_3d) dg::blas2::symv( m_dz, q.at("Psi")[i], q.at("dz Psi")[i]);
    }
    dg::blas2::symv( m_dx_A, apar, dA[0]);
    dg::blas2::symv( m_dy_A, apar, dA[1]);
    if( m_compute_in_3d) dg::blas2::symv( m_dz, apar, dA[2]);
}
template<class Geometry, class Matrix, class Container>
void PerpDynamics<Geometry, Matrix, Container>::update_STderivatives(
    std::map<std::string, std::array<Container,2>>& q,
    const Container& aparST, std::array<Container,3>& dAST)
{
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        //First compute forward and backward derivatives for upwind scheme
        dg::blas1::transform( q.at("ST N")[i], m_temp, dg::PLUS<double>(-m_p.nbc));
        if( m_p.diff_dir == dg::forward)
        {
            dg::blas2::symv( m_dxF_U, m_temp, q.at("ST dx N")[i]);
            dg::blas2::symv( m_dyF_U, m_temp, q.at("ST dy N")[i]);
        }
        else if( m_p.diff_dir == dg::backward)
        {
            dg::blas2::symv( m_dxB_U, m_temp, q.at("ST dx N")[i]);
            dg::blas2::symv( m_dyB_U, m_temp, q.at("ST dy N")[i]);
        }
        else
        {
            dg::blas2::symv( m_dxC_U, m_temp, q.at("ST dx N")[i]);
            dg::blas2::symv( m_dyC_U, m_temp, q.at("ST dy N")[i]);
        }
        if(m_compute_in_3d) dg::blas2::symv( m_dz, m_temp, q.at( "ST dz N")[i]);

        dg::blas2::symv( m_dxF_U, q.at("ST U")[i], q.at("ST dxF U")[i]);
        dg::blas2::symv( m_dyF_U, q.at("ST U")[i], q.at("ST dyF U")[i]);
        dg::blas2::symv( m_dxB_U, q.at("ST U")[i], q.at("ST dxB U")[i]);
        dg::blas2::symv( m_dyB_U, q.at("ST U")[i], q.at("ST dyB U")[i]);
        if(m_compute_in_3d) dg::blas2::symv( m_dz, q.at("ST U")[i], q.at( "ST dzF U")[i]);
        if(m_compute_in_3d) dg::blas2::symv( m_dz, q.at("ST U")[i], q.at( "ST dzB U")[i]);
        dg::blas2::symv( m_dx_P, q.at("ST Psi")[i], q.at("ST dx Psi")[i]);
        dg::blas2::symv( m_dy_P, q.at("ST Psi")[i], q.at("ST dy Psi")[i]);
        if( m_compute_in_3d) dg::blas2::symv( m_dz, q.at("ST Psi")[i], q.at("ST dz Psi")[i]);
    }
    dg::blas2::symv( m_dx_A, aparST, dAST[0]);
    dg::blas2::symv( m_dy_A, aparST, dAST[1]);
    if( m_compute_in_3d) dg::blas2::symv( m_dz, aparST, dAST[2]);
}

template<class Geometry, class Matrix, class Container>
void PerpDynamics<Geometry, Matrix, Container>::compute_density(
    double,
    const std::map<std::string, std::array<Container,2>>& q,
    const Container& apar, const std::array<Container,3>& dA,
    std::array<Container,2>& densityDOT) const
{
    //y[0] = N, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        double mu = m_p.mu[i], tau = m_p.tau[i], beta = m_p.beta;
        dg::blas1::subroutine( [mu, tau, beta] DG_DEVICE (
                double N, double d0FN, double d1FN, double d2FN,
                          double d0BN, double d1BN, double d2BN,
                double U, double d0U, double d1U, double d2U,
                          double d0P, double d1P, double d2P,
                double A, double d0A, double d1A, double d2A,
                double b_0,         double b_1,         double b_2,
                double curv0,       double curv1,       double curv2,
                double curvKappa0,  double curvKappa1,  double curvKappa2,
                double divCurvKappa,
                double& dtN
            )
            {
                dtN = 0;
                // density - upwind scheme
                double v0 = (b_1*d2P - b_2*d1P) + tau*curv0 + mu*U*U*curvKappa0;
                double v1 = (b_2*d0P - b_0*d2P) + tau*curv1 + mu*U*U*curvKappa1;
                double v2 = (b_0*d1P - b_1*d0P) + tau*curv2 + mu*U*U*curvKappa2;
                double bp0 = 0., bp1 = 0., bp2 = 0.;
                if( beta != 0)
                {
                    bp0 = A * curvKappa0 + ( d1A*b_2 - d2A*b_1);
                    bp1 = A * curvKappa1 + ( d2A*b_0 - d0A*b_2);
                    bp2 = A * curvKappa2 + ( d0A*b_1 - d1A*b_0);

                    v0 += U * bp0;
                    v1 += U * bp1;
                    v2 += U * bp2;
                    //Q: doesn't U in U^2K_kappa and U b_perp create nonlinearity
                    //in velocity equation that may create shocks?
                    //A: we did some studies in the reconnection2d program and
                    //did not find shocks. LeVeque argues that for smooth
                    //solutions the upwind discretization should be fine but is
                    //wrong for shocks
                }
                dtN += ( v0 > 0 ) ? -v0*d0BN : -v0*d0FN;
                dtN += ( v1 > 0 ) ? -v1*d1BN : -v1*d1FN;
                dtN += ( v2 > 0 ) ? -v2*d2BN : -v2*d2FN;

                double KappaU = curvKappa0*d0U+curvKappa1*d1U+curvKappa2*d2U;
                double KP = curv0*d0P+curv1*d1P+curv2*d2P;

                dtN +=  - N * ( KP + mu * U * U * divCurvKappa
                                + 2. * mu * U * KappaU);
                if( beta != 0)
                {
                    double divbp = A*divCurvKappa
                                     - (curv0-curvKappa0)*d0A
                                     - (curv1-curvKappa1)*d1A
                                     - (curv2-curvKappa2)*d2A;
                    double bpU = bp0*d0U + bp1*d1U + bp2*d2U;
                    dtN +=  -N*( U*divbp + bpU);
                }
                return;
            },
            //species depdendent
            q.at("N")[i], q.at("dxF N")[i], q.at("dyF N")[i], q.at("dzF N")[i],
                          q.at("dxB N")[i], q.at("dyB N")[i], q.at("dzB N")[i],
            q.at("U")[i], q.at("dx U")[i], q.at("dy U")[i], q.at("dz U")[i],
                          q.at("dx Psi")[i], q.at("dy Psi")[i], q.at("dz Psi")[i],
            //aparallel
            apar, dA[0], dA[1], dA[2],
            //magnetic parameters
            m_b[0], m_b[1], m_b[2],
            m_curv[0], m_curv[1], m_curv[2],
            m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
            m_divCurvKappa, densityDOT[i]
        );
    }
}
template<class Geometry, class Matrix, class Container>
void PerpDynamics<Geometry, Matrix, Container>::compute_velocity(
    double,
    const std::map<std::string, std::array<Container,2>>& q,
    const Container& aparST, const std::array<Container,3>& dAST,
    std::array<Container,2>& velocityDOT) const
{
    //y[0] = N, y[1] = W; fields[0] = N, fields[1] = U
    for( unsigned i=0; i<2; i++)
    {
        ////////////////////perpendicular dynamics////////////////////////
        double mu = m_p.mu[i], tau = m_p.tau[i], beta = m_p.beta;
        dg::blas1::subroutine( [mu, tau, beta] DG_DEVICE (
                double N, double d0N, double d1N, double d2N,
                double U, double d0FU, double d1FU, double d2FU,
                          double d0BU, double d1BU, double d2BU,
                          double d0P, double d1P, double d2P,
                double A, double d0A, double d1A, double d2A,
                double b_0,         double b_1,         double b_2,
                double curv0,       double curv1,       double curv2,
                double curvKappa0,  double curvKappa1,  double curvKappa2,
                double divCurvKappa,
                double& dtU
            )
            {
                dtU = 0;
                // velocity - upwind scheme
                double v0 = (b_1*d2P - b_2*d1P) + tau*curv0 + mu*U*U*curvKappa0;
                double v1 = (b_2*d0P - b_0*d2P) + tau*curv1 + mu*U*U*curvKappa1;
                double v2 = (b_0*d1P - b_1*d0P) + tau*curv2 + mu*U*U*curvKappa2;
                double bp0 = 0., bp1 = 0., bp2 = 0.;
                if( beta != 0)
                {
                    bp0 = A * curvKappa0 + ( d1A*b_2 - d2A*b_1);
                    bp1 = A * curvKappa1 + ( d2A*b_0 - d0A*b_2);
                    bp2 = A * curvKappa2 + ( d0A*b_1 - d1A*b_0);

                    v0 += U * bp0;
                    v1 += U * bp1;
                    v2 += U * bp2;
                    //Q: doesn't U in U^2K_kappa and U b_perp create nonlinearity
                    //in velocity equation that may create shocks?
                    //A: we did some studies in the reconnection2d program and
                    //did not find shocks. LeVeque argues that for smooth
                    //solutions the upwind discretization should be fine but is
                    //wrong for shocks
                }
                // velocity - upwind scheme
                v0 += 2.*tau*curvKappa0;
                v1 += 2.*tau*curvKappa1;
                v2 += 2.*tau*curvKappa2;
                dtU += ( v0 > 0 ) ? -v0*d0BU : -v0*d0FU;
                dtU += ( v1 > 0 ) ? -v1*d1BU : -v1*d1FU;
                dtU += ( v2 > 0 ) ? -v2*d2BU : -v2*d2FU;

                // use centered derivatives
                double KappaN = curvKappa0*d0N+curvKappa1*d1N+curvKappa2*d2N;
                double KappaP = curvKappa0*d0P+curvKappa1*d1P+curvKappa2*d2P;

                dtU +=  - U * ( 2. * tau * KappaN / N + tau * divCurvKappa
                                + KappaP);
                if( beta != 0)
                {
                    double bpN = bp0 * d0N + bp1 * d1N + bp2 * d2N;
                    double bpP = bp0 * d0P + bp1 * d1P + bp2 * d2P;
                    dtU +=  - bpP/mu - tau/mu * bpN/N;
                }
                return;
            },
            //species depdendent
            q.at("ST N")[i], q.at("ST dx N")[i], q.at("ST dy N")[i], q.at("ST dz N")[i],
            q.at("ST U")[i], q.at("ST dxF U")[i], q.at("ST dyF U")[i], q.at("ST dzF U")[i],
                             q.at("ST dxB U")[i], q.at("ST dyB U")[i], q.at("ST dzB U")[i],
                             q.at("ST dx Psi")[i], q.at("ST dy Psi")[i], q.at("ST dz Psi")[i],
            //aparallel
            aparST, dAST[0], dAST[1], dAST[2],
            //magnetic parameters
            m_b[0], m_b[1], m_b[2],
            m_curv[0], m_curv[1], m_curv[2],
            m_curvKappa[0], m_curvKappa[1], m_curvKappa[2],
            m_divCurvKappa, velocityDOT[i]
        );
    }
}

} //namespace feltor
