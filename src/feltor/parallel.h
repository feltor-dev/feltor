#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "parameters.h"

namespace feltor
{
template< class Geometry, class IMatrix, class Matrix, class Container >
struct ParaDynamics
{
    ParaDynamics( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField, dg::file::WrappedJsonValue);
    const dg::geo::Fieldaligned<Geometry, IMatrix, Container>& fieldaligned() const
    {
        return m_fa;
    }
    const dg::geo::Fieldaligned<Geometry, IMatrix, Container>& fieldalignedHalf() const
    {
        return m_faHalf;
    }
    void set_sheath(double sheath_rate, const Container& sheath,
            const Container& sheath_coordinate)
    {
        m_sheath_rate = sheath_rate;
        dg::assign( sheath, m_sheath);
        dg::assign( sheath_coordinate, m_sheath_coordinate);
    }

    double get_sheath_rate() const {
        return m_sheath_rate;
    }
    const Container& get_sheath() const{
        return m_sheath;
    }
    const Container& get_sheath_coordinate() const{
        return m_sheath_coordinate;
    }
    void update_staggered_density_and_phi( double t,
        std::map<std::string, std::array<Container,2>>& q);
    void update_staggered_density_and_ampere( double t,
        std::map<std::string, std::array<Container,2>>& q);
    void update_velocity_and_apar( double t,
        std::map<std::string, std::array<Container,2>>& q,
        const Container& aparST, Container& apar);
    void compute_parallel(  const std::map<std::string, std::array<Container,2>>&,
                            std::array<std::array<Container,2>,2>& yp);
    void add_sheath_terms( const std::map<std::string, std::array<Container,2>>&,
                            std::array<std::array<Container,2>,2>& yp);
    void update_parallel_bc_1st( Container& minusST, Container& plusST,
            dg::bc bcx, double value) const
    {
        if( m_p.fci_bc == "along_field")
            dg::geo::assign_bc_along_field_1st( m_faHalf, minusST, plusST,
                    minusST, plusST, bcx, {value,value});
        else
        {
            if( bcx == dg::DIR)
            {
                dg::blas1::plus( minusST, -value);
                dg::geo::swap_bc_perp( m_fa, minusST, plusST,
                        minusST, plusST);
                dg::blas1::plus( minusST, +value);
            }
        }
    }
    void update_parallel_bc_2nd( const dg::geo::Fieldaligned<Geometry, IMatrix,
            Container>& fa, Container& minus, const Container& value0,
            Container& plus, dg::bc bcx, double value) const
    {
        if( m_p.fci_bc == "along_field")
        {
            dg::geo::assign_bc_along_field_2nd( fa, minus, value0,
                    plus, minus, plus, bcx, {value,value});
        }
        else
        {
            if( bcx == dg::DIR)
            {
                dg::blas1::plus( minus, -value);
                dg::geo::swap_bc_perp( fa, minus, plus,
                        minus, plus);
                dg::blas1::plus( minus, +value);
            }
        }
    }
    const Container& get_divNUb( unsigned i) const { return m_divNUb[i];}
    void add_densities_diffusion(
        const std::map<std::string, std::array<Container,2>>& q,
        std::array<std::array<Container,2>,2>& yp);
    void add_velocities_diffusion(
        const std::map<std::string, std::array<Container,2>>& q,
        std::array<std::array<Container,2>,2>& yp);

    private:

    void compute_parallel_flux(
             const Container& velocityKM,
             const Container& velocityKP,
             const Container& densityM,
             const Container& density,
             const Container& densityP,
             Container& fluxM,
             Container& fluxP,
             std::string slope_limiter)
    {
        compute_parallel_advection( velocityKM, velocityKP, densityM, density, densityP,
                fluxM, fluxP, slope_limiter);
        dg::blas1::pointwiseDot( fluxM, velocityKM, fluxM);
        dg::blas1::pointwiseDot( fluxP, velocityKP, fluxP);
    }
    void compute_parallel_advection(
             const Container& velocityKM,
             const Container& velocityKP,
             const Container& densityM,
             const Container& density,
             const Container& densityP,
             Container& fluxM,
             Container& fluxP,
             std::string slope_limiter);
    void compute_parallel_flux(
        const Container& velocity,
        const Container& minusST,
        const Container& plusST,
        Container& flux,
        std::string slope_limiter)
    {
        compute_parallel_advection( velocity, minusST, plusST, flux, slope_limiter);
        dg::blas1::pointwiseDot( velocity, flux, flux);
    }
    void compute_parallel_advection(
        const Container& velocity,
        const Container& minusST,
        const Container& plusST,
        Container& flux,
        std::string slope_limiter);




    dg::geo::Fieldaligned<Geometry, IMatrix, Container> m_fa, m_faHalf;

    Container m_temp;
    Container m_minus, m_plus;
    // Helper variables for compute_parallel_flux
    Container m_vbm, m_vbp, m_dN, m_dNMM, m_dNM, m_dNZ, m_dNP, m_dNPP;

    //std::array<std::array<Container,2>,2> m_divNUb;
    std::array<Container,2> m_divNUb;

    Container m_sheath_coordinate, m_sheath;

    const feltor::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    double m_sheath_rate = 0.;
    bool m_reversed_field = false;
};

template<class Grid, class IMatrix, class Matrix, class Container>
ParaDynamics<Grid, IMatrix, Matrix, Container>::ParaDynamics( const Grid& g,
    feltor::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue js
    ): m_p(p), m_js(js)
{
    dg::assign( dg::evaluate( dg::zero, g), m_temp );
    m_minus = m_plus = m_temp;

    m_vbm = m_vbp = m_temp;
    if( m_p.slope_limiter != "none")
        m_dN = m_dNMM = m_dNM = m_dNZ = m_dNP = m_dNPP = m_temp;

    m_divNUb = std::array<Container,2>{ m_temp, m_temp};

    m_reversed_field = false;
    if( mag.ipol()( g.x0(), g.y0()) < 0)
        m_reversed_field = true;
    //in DS we take the true bhat
    auto bhat = dg::geo::createBHat( mag);
    if( p.curvmode == "flutemode")
    {
        bhat = dg::geo::createToroidalBHat( mag);
        m_reversed_field = false;
    }
    // do not construct FCI if we just want to calibrate
    if( !p.calibrate )
    {
        m_fa.construct( bhat, g, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
            p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz, p.interpolation_method);
        m_faHalf.construct( bhat, g, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
            p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz/2., p.interpolation_method );
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix, Container>::update_staggered_density_and_phi(
    double,
    std::map<std::string, std::array<Container,2>>& q)
{
    for( unsigned i=0; i<2; i++)
    {

        m_fa( dg::geo::einsMinus, q.at("N")[i], q.at("N -1")[i]);
        m_fa( dg::geo::zeroForw,  q.at("N")[i], q.at("N 0")[i]);
        m_fa( dg::geo::einsPlus,  q.at("N")[i], q.at("N +1")[i]);
        update_parallel_bc_2nd( m_fa, q.at("N -1")[i], q.at("N 0")[i], q.at("N +1")[i],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);

        m_faHalf( dg::geo::zeroMinus, q.at("Psi")[i], m_minus);
        m_faHalf( dg::geo::einsPlus,  q.at("Psi")[i], m_plus);
        update_parallel_bc_1st( m_minus, m_plus,
                m_p.bcxP, 0.);
        dg::geo::ds_centered( m_faHalf, 1., m_minus, m_plus, 0., q.at( "ST ds Psi")[i]);
        dg::blas1::axpby( 0.5, m_minus, 0.5, m_plus, q.at("ST Psi")[i]);

        m_faHalf( dg::geo::zeroMinus, q.at("N")[i], q.at("ST N -1/2")[i]);
        m_faHalf( dg::geo::einsPlus,  q.at("N")[i], q.at("ST N +1/2")[i]);
        update_parallel_bc_1st( q.at("ST N -1/2")[i], q.at("ST N +1/2")[i],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
        dg::blas1::axpby( 0.5, q.at("ST N -1/2")[i], 0.5, q.at("ST N +1/2")[i], q.at("ST N")[i]);
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix, Container>::update_velocity_and_apar(
    double,
    std::map<std::string, std::array<Container,2>>& q,
    const Container& aparST, Container& apar)
{
    for( unsigned i=0; i<2; i++)
    {
        // Compute dsU and velocity
        m_faHalf( dg::geo::einsMinus, q.at("ST U")[i], q.at("U -1/2")[i]);
        m_faHalf( dg::geo::zeroPlus,  q.at("ST U")[i], q.at("U +1/2")[i]);
        update_parallel_bc_1st( q.at("U -1/2")[i], q.at("U +1/2")[i], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, q.at("U -1/2")[i], 0.5, q.at("U +1/2")[i], q.at("U")[i]);

        m_fa( dg::geo::einsMinus, q.at("ST U")[i], q.at("ST U -1")[i]);
        m_fa( dg::geo::zeroForw,  q.at("ST U")[i], q.at("ST U 0")[i]);
        m_fa( dg::geo::einsPlus,  q.at("ST U")[i], q.at("ST U +1")[i]);
        update_parallel_bc_2nd( m_fa, q.at("ST U -1")[i], q.at("ST U 0")[i],
                q.at("ST U +1")[i], m_p.bcxU, 0.);
    }
    // Compute apar
    m_faHalf( dg::geo::einsMinus, aparST, m_minus);
    m_faHalf( dg::geo::zeroPlus,  aparST, m_plus);
    update_parallel_bc_1st( m_minus, m_plus, m_p.bcxA, 0.);
    dg::blas1::axpby( 0.5, m_minus, 0.5, m_plus, apar);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix,
     Container>::compute_parallel_advection(
             const Container& velocityKM,
             const Container& velocityKP,
             const Container& densityM,
             const Container& density,
             const Container& densityP,
             Container& fluxM,
             Container& fluxP,
             std::string slope_limiter
             )
{
    // positive and negative velocities are defined wrt to the coordinate system
    // but we need it wrt to the b-field
    if( m_reversed_field)
    {
        dg::blas1::pointwiseDot( -1., velocityKM, m_vbm);
        dg::blas1::pointwiseDot( -1., velocityKP, m_vbp);
    }
    else
    {
        dg::blas1::copy( velocityKM, m_vbm);
        dg::blas1::copy( velocityKP, m_vbp);
    }
    dg::blas1::evaluate( fluxM, dg::equals(), dg::Upwind(),
            m_vbm, densityM, density);
    dg::blas1::evaluate( fluxP, dg::equals(), dg::Upwind(),
            m_vbp, density, densityP);
    if(slope_limiter != "none" )
    {
        // compute dn_k-1, dn_k, dn_k+1
        // By transforming dn to plus and minus planes
        dg::blas1::axpby( 1., densityP, -1., density, m_dNZ);

        m_fa( dg::geo::zeroForw, m_dNZ, m_dNP);
        m_fa( dg::geo::einsPlus, m_dNZ, m_dNPP);

        dg::blas1::axpby( 1., density, -1., densityM, m_dNZ);

        m_fa( dg::geo::zeroForw, m_dNZ, m_dNM);
        m_fa( dg::geo::einsMinus, m_dNZ, m_dNMM);

        // Let's keep the default boundaries of NEU
        // boundary values are (probably?) never used in the slope limiter branches
        //dg::blas1::copy(density, m_temp); // save density
        //update_parallel_bc_2nd( m_fa, m_temp, densityP, m_plus, dg::NEU, 0.);
        //dg::blas1::copy(density, m_temp);
        //update_parallel_bc_2nd( m_fa, m_minus, densityM, m_temp, dg::NEU, 0.);
        // dn is computed inside the limiter
        if( slope_limiter == "minmod")
        {
            dg::blas1::evaluate( fluxM, dg::plus_equals(),
                dg::SlopeLimiter<dg::MinMod>(), m_vbm,
                m_dNMM, m_dNM, m_dNP, 0.5, 0.5);
            dg::blas1::evaluate( fluxP, dg::plus_equals(),
                dg::SlopeLimiter<dg::MinMod>(), m_vbp,
                m_dNM, m_dNP, m_dNPP, 0.5, 0.5);
        }
        else if( slope_limiter == "vanLeer")
        {
            dg::blas1::evaluate( fluxM, dg::plus_equals(),
                dg::SlopeLimiter<dg::VanLeer>(), m_vbm,
                m_dNMM, m_dNM, m_dNP, 0.5, 0.5);
            dg::blas1::evaluate( fluxP, dg::plus_equals(),
                dg::SlopeLimiter<dg::VanLeer>(), m_vbp,
                m_dNM, m_dNP, m_dNPP, 0.5, 0.5);
        }
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix,
     Container>::compute_parallel_advection( const Container& velocity,
             const Container& minusST, const Container& plusST,
             Container& flux,
             std::string slope_limiter
             )
{
    if( m_reversed_field)
    {
        dg::blas1::pointwiseDot( -1., velocity, m_vbp);
    }
    else
    {
        dg::blas1::copy( velocity, m_vbp);
    }
    dg::blas1::evaluate( flux, dg::equals(), dg::Upwind(),
            m_vbp, minusST, plusST);
    if(slope_limiter != "none" )
    {
        // compute dn_k-1, dn_k, dn_k+1
        // By transforming dn to plus and minus planes
        dg::blas1::axpby( 1., plusST, -1., minusST, m_dN);
        m_fa( dg::geo::einsMinus, m_dN, m_dNM);
        m_fa( dg::geo::zeroForw,  m_dN, m_dNZ);
        m_fa( dg::geo::einsPlus,  m_dN, m_dNP);
        // Let's keep the default boundaries of NEU
        // boundary values are (probably?) never used in the slope limiter branches
        //update_parallel_bc_2nd( m_fa, m_minus, m_temp, m_plus, dg::NEU, 0.);
        if( slope_limiter == "minmod")
        {
            dg::blas1::evaluate( flux, dg::plus_equals(),
                dg::SlopeLimiter<dg::MinMod>(), m_vbp,
                m_dNM, m_dNZ, m_dNP, 0.5, 0.5);
        }
        else if( slope_limiter == "vanLeer")
        {
            dg::blas1::evaluate( flux, dg::plus_equals(),
                dg::SlopeLimiter<dg::VanLeer>(), m_vbp,
                m_dNM, m_dNZ, m_dNP, 0.5, 0.5);
        }
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix, Container>::compute_parallel(
    const std::map<std::string, std::array<Container,2>>& q,
    std::array<std::array<Container,2>,2>& yp)
{
    for( unsigned i=0; i<2; i++)
    {
        // "velocity-staggered-fieldaligned"
        //// compute qhat
        //compute_parallel_flux( q.at("U -1/2")[i], q.at("U +1/2")[i],
        //        m_minusN[i], m_zeroN[i], m_plusN[i],
        //        m_minus, m_plus, m_p.slope_limiter);
        //// Now compute divNUb
        //dg::geo::ds_divCentered( m_faHalf, 1., m_minus, m_plus, 0.,
        //        m_divNUb[i]);
        //dg::blas1::axpby( -1., m_divNUb[i], 1., yp[0][i]);

        //// compute grad U2/2
        //dg::blas1::axpby( 0.25, m_minusU[i], 0.25, m_zeroU[i], q.at("U -1/2")[i]);
        //dg::blas1::axpby( 0.25, m_zeroU[i],  0.25, m_plusU[i], q.at("U +1/2")[i]);
        //compute_parallel_flux( q.at("U -1/2")[i], q.at("U +1/2")[i],
        //        m_minusU[i], m_zeroU[i], m_plusU[i],
        //        m_minus, m_plus,
        //        m_p.slope_limiter);
        //dg::geo::ds_centered( m_faHalf, -1., m_minus, m_plus, 1., yp[1][i]);
        //
        // "velocity-staggered"
        compute_parallel_flux( q.at("ST U 0")[i], q.at("ST N -1/2")[i], q.at("ST N +1/2")[i],
                m_temp, m_p.slope_limiter);
        m_faHalf( dg::geo::zeroPlus,  m_temp, m_plus);
        m_faHalf( dg::geo::einsMinus, m_temp, m_minus);
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_divCentered( m_faHalf, 1., m_minus, m_plus, 0., m_divNUb[i]);
        dg::blas1::axpby( -1., m_divNUb[i], 1., yp[0][i]);

        // compute fhat
        compute_parallel_flux( q.at("U")[i], q.at("U -1/2")[i], q.at("U +1/2")[i],
                m_temp, m_p.slope_limiter);
        m_faHalf( dg::geo::einsPlus, m_temp, m_plus);
        m_faHalf( dg::geo::zeroMinus, m_temp, m_minus);
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_centered( m_faHalf, -0.5, m_minus, m_plus, 1, yp[1][i]);

        // Add density gradient and electric field
        double tau = m_p.tau[i], mu = m_p.mu[i], delta = m_fa.deltaPhi();
        dg::blas1::subroutine( [tau, mu, delta ]DG_DEVICE ( double& WDot,
                    double dsP, double QN, double PN, double bphi)
                {
                    WDot -= 1./mu*dsP;
                    WDot -= tau/mu*bphi*(PN-QN)/delta/2.*(1/PN + 1/QN);
                },
                yp[1][i], q.at("ST ds Psi")[i], q.at("ST N -1/2")[i], q.at("ST N +1/2")[i], m_fa.bphi()
        );
    }
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix, Container>::add_sheath_terms(
    const std::map<std::string, std::array<Container,2>>& q,
        std::array<std::array<Container,2>,2>& yp)
{
    // add sheath boundary conditions
    if( m_sheath_rate != 0)
    {
        ////density
        ////Here, we need to find out where "downstream" is
        //!! Simulations does not really work without
        for( unsigned i=0; i<2; i++)
        {
            //The coordinate automatically sees the reversed field
            //but m_plus and m_minus are defined wrt the angle coordinate
            if( m_reversed_field) //bphi negative (exchange + and -)
                dg::blas1::evaluate( m_temp, dg::equals(), dg::Upwind(),
                     m_sheath_coordinate, q.at("N +1")[i], q.at("N -1")[i]);
            else
                dg::blas1::evaluate( m_temp, dg::equals(), dg::Upwind(),
                     m_sheath_coordinate, q.at("N -1")[i], q.at("N +1")[i]);
            dg::blas1::pointwiseDot( m_sheath_rate, m_temp,  m_sheath,
                                    -m_sheath_rate, q.at("N")[i], m_sheath,
                                     1., yp[0][i]);
        }
        //compute sheath velocity
        if( "wall" == m_p.sheath_bc)
        {
            for( unsigned i=0; i<2; i++)
            {
                //dg::blas1::axpby( +m_sheath_rate*m_p.nwall, m_sheath, 1., yp[0][i] );
                dg::blas1::axpby( +m_sheath_rate*m_p.uwall, m_sheath, 1., yp[1][i] );
            }
        }
        else
        {
            //velocity c_s
            double cs = sqrt(1.+m_p.tau[1]), sheath_rate = m_sheath_rate;
            if( "insulating" == m_p.sheath_bc)
            {
                // u_e,sh = s*sqrt(1+tau) Ni/ne
                dg::blas1::evaluate( yp[1][0], dg::plus_equals(),
                        [cs, sheath_rate]DG_DEVICE( double sheath_coord, double
                            sheath, double ne, double ni) {
                            return cs*sheath_rate*sheath_coord*ni/ne*sheath;
                        },
                        m_sheath_coordinate, m_sheath, q.at("ST N")[0],
                        q.at("ST N")[1]);
            }
            else // "bohm" == m_p.sheath_bc
            {
                //u_e,sh = s*1/sqrt(|mu_e|2pi) exp(-phi)
                double mue = fabs(m_p.mu[0]), tau = m_p.tau[1];
                dg::blas1::evaluate( yp[1][0], dg::plus_equals(),
                    [mue, sheath_rate, tau]DG_DEVICE(
                        double sheath_coord, double sheath, double phi) {
                        return sheath_rate * sheath_coord * sheath *
                            sqrt(1.+tau) * exp(-phi) / sqrt( mue*2.*M_PI);
                    },
                    m_sheath_coordinate, m_sheath, q.at("ST Psi")[0]);
            }
            // u_i,sh = s*sqrt(1+tau)
            dg::blas1::pointwiseDot( sheath_rate*cs,
                    m_sheath, m_sheath_coordinate, 1.,  yp[1][1]);
        }
        // Apply to U, not W
        for( unsigned i=0; i<2; i++)
            dg::blas1::pointwiseDot( -m_sheath_rate, m_sheath, q.at("ST U")[i],
                1., yp[1][i]);
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix, Container>::add_densities_diffusion(
    const std::map<std::string, std::array<Container,2>>& q,
        std::array<std::array<Container,2>,2>& yp)
{
    for( unsigned i=0; i<2; i++)
    {
        if( m_p.nu_parallel_n > 0)
        {
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_n,
                    q.at("N -1")[i], q.at("N 0")[i], q.at("N +1")[i], 1., yp[0][i]);
        }
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void ParaDynamics<Geometry, IMatrix, Matrix, Container>::add_velocities_diffusion(
    const std::map<std::string, std::array<Container,2>>& q,
        std::array<std::array<Container,2>,2>& yp)
{
    // velocityST[0] := u_e^dagger
    // velocityST[1] := U_i^dagger
    for( unsigned i=0; i<2; i++)
    {
        // Add parallel viscosity
        if( m_p.nu_parallel_u[i] > 0)
        {
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[i],
                    q.at("ST U -1")[i], q.at("ST U 0")[i], q.at("ST U +1")[i], 0., m_temp);
            if( !m_p.modify_diff)
                dg::blas1::pointwiseDivide( 1., m_temp, q.at("ST N")[i], 1., yp[1][i]);
            else
                dg::blas1::axpby( 1., m_temp, 1., yp[1][i]);
        }
        double nu = m_p.nu_parallel_n;
        if( m_p.modify_diff)
            nu += m_p.nu_parallel_u[i];
        if( nu > 0)
        {
            // Add density gradient correction
            double delta = m_fa.deltaPhi();
            dg::blas1::subroutine( [delta, nu]DG_DEVICE ( double& WDot,
                        double QN, double PN, double UM, double U0, double UP,
                        double bphi)
                    {
                        //upwind scheme
                        double nST = (PN+QN)/2.;
                        double current = -nu*bphi*(PN-QN)/delta/nST;
                        if( current > 0)
                            WDot += - current*bphi*(U0-UM)/delta;
                        else
                            WDot += - current*bphi*(UP-U0)/delta;

                    },
                    yp[1][i], q.at("ST N -1/2")[i], q.at("ST N +1/2")[i],
                    q.at("ST U -1")[i], q.at("ST U 0")[i], q.at("ST U +1")[i], m_fa.bphi()
            );
        }
    }
}
} //namespace feltor
