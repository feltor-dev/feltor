#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "perpendicular.h"
#include "parallel.h"
#include "parameters.h"

namespace feltor
{
// Sources and Damping
template< class Geometry, class Matrix, class Container >
struct Sources
{
    Sources( const Geometry&, feltor::Parameters,
        dg::geo::TokamakMagneticField, dg::file::WrappedJsonValue,
        const feltor::PerpDynamics<Geometry, Matrix, Container>& perp);
    void add_wall_terms(
        const std::map<std::string, std::array<Container,2>>& q,
        std::array<std::array<Container,2>,2>& yp) const;
    template<class IMatrix>
    void add_source_terms(
        const std::map<std::string, std::array<Container,2>>& q,
        const PerpDynamics<Geometry,Matrix,Container>& perp,
        const ParaDynamics<Geometry,IMatrix,Matrix,Container>& para,
        std::array<std::array<Container,2>,2>& yp);
    //source strength, profile - 1
    void set_source( bool fixed_profile, const Container& profile, double source_rate, const Container& source, double minne, double minrate, double minalpha)
    {
        m_fixed_profile = fixed_profile;
        m_profne = profile;
        m_source_rate = source_rate;
        m_source = source;
        m_minne = minne;
        m_minrate = minrate;
        m_minalpha = minalpha;
    }
    void set_wall(const Container& wall)
    {
        dg::assign( wall, m_wall);
    }
    const Container& get_wall() const{
        return m_wall;
    }
    // for fixed profile this is the profile that is forced
    const Container& get_source_prof() const{
        return m_profne;
    }
    // for influx this is the source (missing source rate), for fixed profile
    // this contains the source region/damping
    const Container& get_source( ) const{
        return m_source;
    }
    const Container& get_density_source ( unsigned i) const{
        return m_ss[0][i];
    }
    void compute_lapMperpP (const Container& in, Container& result)
    {
        m_lapperpP.set_chi( 1.);
        dg::blas2::gemv( m_lapperpP, in, result);
    }
    void compute_source_pol( double alpha, const Container& density, const Container& phi,  Container& temp, double beta, Container& result)
    {
        // we don't want jumps in phi in here so we use lapperpP
        dg::blas1::pointwiseDot( m_p.mu[1], density, m_binv, m_binv, 0., temp);
        m_lapperpP.set_chi( temp);
        dg::blas2::symv( -alpha, m_lapperpP, phi, beta, result);
    }
    private:
    const feltor::Parameters m_p;
    mutable Container m_temp0, m_temp1, m_minus, m_plus;
    Container m_wall, m_binv;
    std::array<std::array<Container,2>,2> m_ss; // source terms for all species
    Container m_source, m_profne; // (physical) source terms for all species
    dg::Elliptic3d< Geometry, Matrix, Container> m_lapperpP;

    double m_source_rate = 0;
    bool m_fixed_profile = true;

    double m_minne = 0., m_minrate  = 0., m_minalpha = 0.;

};
template<class Geometry, class Matrix, class Container>
Sources<Geometry, Matrix, Container>::Sources( const Geometry& g,
    feltor::Parameters p, dg::geo::TokamakMagneticField,
    dg::file::WrappedJsonValue,
    const feltor::PerpDynamics<Geometry, Matrix, Container>& perp
    ): m_p(p)
{
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_minus = m_plus = m_temp1 = m_temp0;
    for( int i=0; i<2; i++)
    for( int j=0; j<2; j++)
        m_ss[i][j] = m_temp0;
    m_source = m_profne = m_temp0;
    dg::assign(  perp.binv(), m_binv);

    m_lapperpP.construct ( g, p.bcxP, p.bcyP, dg::PER,  p.pol_dir),
    m_lapperpP.set_chi( perp.projection());
    if( (p.curvmode == "true") && (p.symmetric == false))
        ;
    else
    {
        m_lapperpP.set_compute_in_2d(true);
    }
    m_lapperpP.set_jfactor(0); //we don't want jump terms in source
}


template<class Geometry, class Matrix, class Container>
template<class IMatrix>
void Sources<Geometry, Matrix, Container>::add_source_terms(
    const std::map<std::string, std::array<Container,2>>& q,
    const PerpDynamics<Geometry,Matrix,Container>& perp,
    const ParaDynamics<Geometry,IMatrix,Matrix,Container>& para,
    std::array<std::array<Container,2>,2>& yp)
{
    if( m_source_rate != 0.0)
    {
        if( m_fixed_profile )
            dg::blas1::subroutine(
                [] DG_DEVICE ( double& result, double ne, double profne,
                    double source, double source_rate){
                    result = source_rate*source*(profne - ne);
                    },
                m_ss[0][0], q.at("N")[0], m_profne, m_source, m_source_rate);
        else
            dg::blas1::axpby( m_source_rate, m_source, 0., m_ss[0][0]);
    }
    else
        dg::blas1::copy( 0., m_ss[0][0]);
    // add prevention to get below lower limit
    if( m_minrate != 0.0)
    {
        // do not make lower forcing a velocity source
        // MW it may be that this form does not go well with the potential
        dg::blas1::transform( q.at("N")[0], m_temp0, dg::PolynomialHeaviside(
                    m_minne-m_minalpha/2., m_minalpha/2., -1) );
        dg::blas1::transform( q.at("N")[0], m_temp1, dg::PLUS<double>( -m_minne));
        dg::blas1::pointwiseDot( -m_minrate, m_temp1, m_temp0, 1., yp[0][0]);
        dg::blas1::transform( q.at("N")[1], m_temp0, dg::PolynomialHeaviside(
                    m_minne-m_minalpha/2., m_minalpha/2., -1) );
        dg::blas1::transform( q.at("N")[1], m_temp1, dg::PLUS<double>( -m_minne));
        dg::blas1::pointwiseDot( -m_minrate, m_temp1, m_temp0, 1., yp[0][1]);
    }

    //compute FLR corrections S_N = (1-0.5*mu*tau*Lap)*S_n
    perp.compute_lapMperpU( m_ss[0][0], m_temp0); // lapU avoids subtraction of nbc
    dg::blas1::axpby( 1., m_ss[0][0], 0.5*m_p.tau[1]*m_p.mu[1], m_temp0, m_ss[0][1]);
    // potential part of FLR correction S_N += -div*(mu S_n grad*Phi/B^2)
    dg::blas1::pointwiseDot( m_p.mu[1], m_ss[0][0], m_binv, m_binv, 0., m_temp0);
    m_lapperpP.set_chi( m_temp0);
    m_lapperpP.symv( 1., q.at("Psi")[0], 1., m_ss[0][1]);

    // S_U = - U S_N/N
    for(int i=0; i<2; i++)
    {
        // transform to adjoint plane and add to velocity source
        para.fieldalignedHalf()( dg::geo::zeroMinus, m_ss[0][i], m_minus);
        para.fieldalignedHalf()( dg::geo::einsPlus,  m_ss[0][i], m_plus);
        para.update_parallel_bc_1st( m_minus, m_plus, m_p.bcxN, 0.);
        dg::geo::ds_average( para.fieldalignedHalf(), 1., m_minus, m_plus, 0., m_temp0);
        dg::blas1::evaluate( m_ss[1][i], dg::equals(), []DG_DEVICE(
                    double sn, double u, double n){ return -u*sn/n;},
                m_temp0, q.at("ST U")[i], q.at("ST N")[i]);
    }
    //Add all to the right hand side
    dg::blas1::axpby( 1., m_ss, 1.0, yp);
}

template<class Geometry, class Matrix, class Container>
void Sources<Geometry, Matrix, Container>::add_wall_terms(
    const std::map<std::string, std::array<Container,2>>& q,
    std::array<std::array<Container,2>,2>& yp) const
{
    // add wall boundary conditions
    if( m_p.wall_rate != 0)
    {
        for( unsigned i=0; i<2; i++)
        {
            double nwall = m_p.nwall, uwall = m_p.uwall;
            if( m_p.wall_bc == "floating")
            {
                // chi_w ( 1 - chi_w )
                dg::blas1::pointwiseDot ( 1., 1., m_wall, -1., m_wall, m_wall, 0., m_temp0);
                double norm = dg::blas1::dot( m_lapperp.weights(), m_temp0);
                nwall = dg::blas2::dot( q.at("N")[i], m_lapperpP.weights(), m_temp0)/norm;
            }
            dg::blas1::axpby( +m_p.wall_rate*nwall, m_wall, 1., yp[0][i] );
            dg::blas1::axpby( +m_p.wall_rate*uwall, m_wall, 1., yp[1][i] );
            dg::blas1::pointwiseDot( -m_p.wall_rate, m_wall, q.at("N")[i], 1., yp[0][i]);
            dg::blas1::pointwiseDot( -m_p.wall_rate, m_wall, q.at("ST U")[i], 1., yp[1][i]);
        }
    }
}

} //namespace feltor

