#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "perpendicular.h"
#include "parameters.h"

namespace thermal
{
// Sources and Damping
template< class Geometry, class IMatrix, class Matrix, class Container >
struct Sources
{
    Sources( const Geometry&, thermal::Parameters,
        dg::geo::TokamakMagneticField, dg::file::WrappedJsonValue);
    void add_wall_terms( unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        const std::array<std::vector<Container>,6>& y,
        std::array<std::vector<Container>,6>& yp) const;
    void add_source_terms(
        unsigned s,
        const Container& phi,
        const std::map<std::string, std::vector<Container>>& q,
        const std::array<std::vector<Container>,6>& y,
        std::array<std::vector<Container>,6>& yp);
    //source strength, profile - 1
    void set_source(
        bool fixed_profile, // cannot be mixed among species or equations due to quasineutrality and transformation
        const std::array<std::vector<double>,3>& source_rate,
        const std::array<std::vector<dg::x::HVec>,3>& profile, // for influx this can be ignored
        const std::array<std::vector<dg::x::HVec>,3>& source_region,  // for fixed profile this contains damping
        const std::vector<double>& minn,
        double mint,
        double minrate,
        double minbeta)
    {
        m_fixed_profile = fixed_profile;
        m_source_rate = source_rate;
        for( unsigned u=0; u<3; u++)
        {
            m_profile[u].resize( m_p.num_species);
            m_source_region[u].resize( m_p.num_species);
            for( unsigned s=0; s<m_p.num_species; s++)
            {
                m_profile[u][s] = profile[u][s];
                m_source_region[u][s]  = source_region[u][s];
            }
        }
        m_minn = minn;
        m_mint = mint;
        m_minrate = minrate;
        m_minbeta = minbeta;
    }
    void set_wall(const Container& wall)
    {
        dg::assign( wall, m_wall);
    }
    const Container& get_wall() const{
        return m_wall;
    }
    // for influx this is the source (missing source rate), for fixed profile
    // this contains the source region/damping
    const Container& get_source_region(unsigned u, unsigned s) const{
        return m_source_region[u][s];
    }
    // for fixed profile this is the profile that is forced
    const Container& get_source_prof(unsigned u, unsigned s) const{
        return m_profile[u][s];
    }
    const Container& get_source( unsigned u, unsigned s) const{
        return m_ss[u][s];
    }
    const Container& Btorinv( ) const { return m_Btorinv; }
    void transform_density_pperp( unsigned s, const Container& density,
        const Container& pperp, const Container& phi,
        Container& gydensity, Container& gypperp) const // can be called inplace!
    {
        double mus = m_p.mu[s], zs = m_p.z[s];
        if( s == 0) // electrons
        {
            dg::blas1::copy( density, gydensity);
            dg::blas1::copy( pperp, gypperp);
            return;
        }
        //compute FLR corrections S_N = (S_n-0.5 Lap S_p) - Div ( S_n phi)
        dg::blas1::pointwiseDot( mus/zs/zs, pperp, m_Btorinv, m_Btorinv, 0., m_tempgy0);
        dg::blas2::gemv( m_lapperp, m_tempgy0, m_tempgy1);
        dg::blas1::axpby( 1., density, 0.5, m_tempgy1);
        // potential part of FLR correction S_N += -div*(mu S_n grad*Phi/B^2)
        dg::blas1::pointwiseDot( mus/zs, density, m_Btorinv, m_Btorinv, 0., m_tempgy0);
        m_lapperpP.symv( 1., phi, 0., m_tempgy0, 1., m_tempgy1);
        dg::blas1::copy( m_tempgy1, gydensity); //gydensity can alias density!
        // Pressure trafo
        dg::blas1::pointwiseDot( mus/zs, pperp, m_Btorinv, m_Btorinv, 0., m_tempgy0);
        dg::blas1::copy( pperp, gypperp);
        m_lapperpP.symv( 1., phi, 0., m_tempgy0, 1., gypperp);
    }
    private:
    const thermal::Parameters m_p;
    mutable Container m_temp0, m_temp1, m_tempgy0, m_tempgy1;
    Container m_wall, m_Btorinv;
    std::array<std::vector<Container>,3> m_ss; // source terms for all species
    std::array<std::vector<Container>,3> m_source_region, m_profile; // (physical) source terms for all species
    dg::Elliptic2d< Geometry, Matrix, Container> m_lapperp, m_lapperpP;

    std::array<std::vector<double>,3> m_source_rate;
    bool m_fixed_profile = false;

    std::vector<double> m_minn;
    double m_mint, m_minrate  = 0., m_minbeta;

};
template<class Grid, class IMatrix, class Matrix, class Container>
Sources<Grid, IMatrix, Matrix, Container>::Sources( const Grid& g,
    thermal::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue
    ): m_p(p)
{
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_tempgy0 = m_tempgy1 = m_temp1 = m_temp0;
    for( int i=0; i<3; i++)
        m_ss[i].resize( m_p.num_species, m_temp0);
    m_source_region = m_profile = m_ss;
    m_lapperp.construct ( g, p.bcx, p.bcy, p.diff_dir);
    m_lapperpP.construct ( g, p.bcxP, p.bcyP,  p.pol_dir),
    dg::assign(  dg::pullback(dg::geo::InvBtor(mag), g), m_Btorinv);
}


template<class Geometry, class IMatrix, class Matrix, class Container>
void Sources<Geometry, IMatrix, Matrix, Container>::add_source_terms(
    unsigned s,
    const Container& phi,
    const std::map<std::string, std::vector<Container>>& q,
    const std::array<std::vector<Container>,6>& y,
    std::array<std::vector<Container>,6>& yp)
{
    // First, add minimum density and pressure source
    if( m_minrate != 0.0)
    {
        double alpha = m_minbeta*m_minn[s];
        // do not make lower forcing a velocity source
        // 1. Density
        dg::blas1::transform( q.at("N")[s], m_temp0, dg::PolynomialHeaviside(
                    m_minn[s]-alpha[0]/2., alpha[0]/2., -1) );
        dg::blas1::transform( q.at("N")[s], m_temp1, dg::PLUS<double>( -m_minn[s]));
        dg::blas1::pointwiseDot( -m_minrate, m_temp1, m_temp0, 0., m_temp0);
        dg::blas1::axpby( 1., m_temp0, 1., yp[0][s]);

        // also add to perp and para eqs
        dg::blas1::pointwiseDot( 1., q.at("Tperp")[s], m_temp0, 1., yp[1][s]);
        dg::blas1::pointwiseDot( 1., q.at("Tpara")[s], m_temp0, 1., yp[2][s]);

        // 2. Tperp and Tpara
        alpha = m_minbeta*mint;
        dg::blas1::transform( q.at("Tperp")[s], m_temp0, dg::PolynomialHeaviside(
                    m_minn[s]-alpha/2., alpha/2., -1) );
        dg::blas1::transform( q.at("Tperp")[s], m_temp1, dg::PLUS<double>( -mint));
        dg::blas1::pointwiseDot( -m_minrate, q.at("N")[s], m_temp1, m_temp0, 1., yp[1][s]);

        dg::blas1::transform( q.at("Tpara")[s], m_temp0, dg::PolynomialHeaviside(
                    m_minn[s]-alpha/2., alpha/2., -1) );
        dg::blas1::transform( q.at("Tpara")[s], m_temp1, dg::PLUS<double>( -mint));
        dg::blas1::pointwiseDot( -m_minrate, q.at("N")[s], m_temp1, m_temp0, 1., yp[2][s]);

    }

    if( m_fixed_profile )
    {
        // If fixed profile transform profiles first
        transform_density_pperp( s, m_profile[0][s],
            m_profile[1][s], phi, m_ss[0][s], m_ss[1][s]);
        dg::blas1::copy( m_profile[2][s], m_ss[2][s]);
        for( unsigned u=0; u<3; u++)
        {
            // w Chi ( Prof - N )
            dg::blas1::pointwiseDot( m_source_rate[u][s], m_source_region[u][s], m_ss[u][s],
                -m_source_rate[u][s], m_source_region[u][s], y[u][s], 0, m_ss[u][s]);
        }
    }
    else // influx
    {
        for( unsigned u=0; u<3; u++)
            dg::blas1::axpby( m_source_rate[u][s], m_source_region[u][s], 0., m_ss[u][s]);
        // if influx transform sources last
        transform_density_pperp( s, m_ss[0][s],  m_ss[1][s], phi,
            m_ss[0][s], m_ss[1][s]);
    }
    //Add all to the right hand side
    for( unsigned u=0; u<3; u++)
        dg::blas1::axpby( 1., m_ss[u][s], 1.0, yp[u][s]);

    // Recompute S_N^dagger on staggered grid for velocity source
    if( m_fixed_profile )
    {
        // If fixed profile transform profiles first
        transform_density_pperp( s, m_profile[0][s],
            m_profile[1][s], q.at("ST Psi0")[0], m_temp0, m_temp1);
            // w Chi ( Prof - N )
        dg::blas1::pointwiseDot( m_source_rate[0][s], m_source_region[0][s], m_temp0,
            -m_source_rate[0][s], m_source_region[0][s], q.at("ST N")[s], 0, m_temp0);
    }
    else // influx
    {
        dg::blas1::axpby( m_source_rate[0][s], m_source_region[0][s], 0., m_temp0);
        dg::blas1::axpby( m_source_rate[1][s], m_source_region[1][s], 0., m_temp1);
        // if influx transform sources last
        transform_density_pperp( s, m_temp0,  m_temp1, q.at("ST Psi0")[0],
            m_temp0, m_temp1);
    }
    // Now m_temp0 is S_N^dagger
    // Compute - U S_N /N ^dagger
    dg::blas1::pointwiseDot( q.at("ST U")[s], m_temp0, m_temp0);
    dg::blas1::pointwiseDivide( -1., m_temp0, q.at("ST N")[s], 1., yp[3][s]);
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Sources<Geometry, IMatrix, Matrix, Container>::add_wall_terms(
    unsigned s,
    const std::map<std::string, std::vector<Container>>& q,
    const std::array<std::vector<Container>,6>& y,
    std::array<std::vector<Container>,6>& yp) const
{
    // add wall boundary conditions
    if( m_p.wall_rate != 0)
    {
        std::array<double,6> wall_bc = {
            m_p.nwall[s],
            m_p.twall*m_p.nwall[s],
            m_p.twall*m_p.nwall[s],
            m_p.uwall,
            m_p.qwall,
            m_p.qwall
        };
        if( wall_bc == "floating")
        {
            // chi_w ( 1 - chi_w )
            dg::blas1::pointwiseDot ( 1., 1., m_wall, -1., m_wall, m_wall, 0., m_temp0);
            double norm = dg::blas1::dot( m_lapperp.weights(), m_temp0);
            wall_bc[0] = dg::blas2::dot( y[0][s], m_lapperp.weights(), m_temp0);
            double pperp_avg = dg::blas2::dot( y[1][s], m_lapperp.weights(), m_temp0);
            double ppara_avg = dg::blas2::dot( y[2][s], m_lapperp.weights(), m_temp0);
            wall_bc[1] = wall_bc[2] = (ppara_avg + 2*pperp_avg)/3;
        }
        for( unsigned u=0; u<6; u++)
        {
            if( u == 3)
                // Apply to U, not W
                dg::blas1::pointwiseDot( -m_p.wall_rate, m_wall, q.at("ST U")[s],
                    1., yp[u][s]);
            else
                dg::blas1::pointwiseDot( -m_p.wall_rate, m_wall, y[u][s],
                    1., yp[u][s]);
            dg::blas1::axpby( m_p.wall_rate*wall_bc[s], m_wall, 1., yp[u][s] );
        }
    }
}

} //namespace thermal
