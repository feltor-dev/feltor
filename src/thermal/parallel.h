#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "parameters.h"

namespace thermal
{
template< class Geometry, class IMatrix, class Matrix, class Container >
struct ParaDynamics
{
    ParaDynamics( const Geometry&, thermal::Parameters,
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
    // call before computing AparST
    // Update the parallel transform quantities for the densities N, Pperp and
    // Ppara, Tperp and Tpara
    void compute_staggered_densities(
        const std::array<std::vector<Container>,6>& y,
        std::map<std::string, std::vector<Container>>& q
    );

    const Container& get_divNUb( unsigned u, unsigned s) const { return m_divNUb[u][s];}

    void compute_apar( const Container& aparST, Container& apar)
    {
        m_faHalf( dg::geo::einsMinus, aparST, m_tminus);
        m_faHalf( dg::geo::zeroPlus,  aparST, m_tplus);
        update_parallel_bc_1st( m_tminus, m_tplus, m_p.bcxA, 0.);
        dg::blas1::axpby( 0.5, m_tminus, 0.5, m_tplus, apar);
    }

    void compute_parallel_transformations(
        const std::array<std::vector<Container>,6>& y,
        std::map<std::string, std::vector<Container>>& q
    );

    // Call after update_quantities
    void add_densities_advection(
        unsigned s,
        const std::array<std::vector<Container>,6>& y,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp);
    void add_velocities_advection(
        unsigned s,
        const std::array<std::vector<Container>,6>& y,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp);
    void add_densities_diffusion(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp);
    void add_velocities_diffusion(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp);

    void add_sheath_neumann_terms(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        const std::array<std::vector<Container>,6>& y,
        std::array<std::vector<Container>,6>& yp);
    void add_sheath_velocity_terms(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp);
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

    private:
    // could be a free function?
    void compute_parallel_flux( const Container& velocity,
             const Container& minusST, const Container& plusST,
             Container& flux
             )
    {
        if( m_reversed_field)
        {
            // Upwind( -v, minusST, plusST) == Upwind( v, plusST, minusST)
            dg::blas1::evaluate( flux, dg::equals(), dg::Upwind(),
                velocity, plusST, minusST);
        }
        else
        {
            dg::blas1::evaluate( flux, dg::equals(), dg::Upwind(),
                velocity, minusST, plusST);
        }
        dg::blas1::pointwiseDot( velocity, flux, flux);
    }
    dg::geo::Fieldaligned<Geometry, IMatrix, Container> m_fa, m_faHalf;

    Container m_temp, m_tminus, m_tplus;
    Container m_divb;

    std::array<std::vector<Container>,6> m_divNUb;

    Container m_sheath_coordinate, m_sheath;

    const thermal::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    bool m_reversed_field = false;
    double m_sheath_rate = 0.;
};

template<class Grid, class IMatrix, class Matrix, class Container>
ParaDynamics<Grid, IMatrix, Matrix, Container>::ParaDynamics( const Grid& g,
    thermal::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue js
    ): m_p(p), m_js(js)
{
    dg::assign( dg::evaluate( dg::zero, g), m_temp );
    m_tminus = m_tplus = m_temp;

    std::fill( m_divNUb.begin(), m_divNUb.end(),
        std::vector<Container>( m_p.num_species, m_temp) );

    m_reversed_field = false;
    if( mag.ipol()( g.x0(), g.y0()) < 0)
        m_reversed_field = true;
    //in DS we take the true bhat
    auto bhat = dg::geo::createBHat( mag);
    // do not construct FCI if we just want to calibrate
    if( !p.calibrate )
    {
        m_fa.construct( bhat, g, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
            p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz, p.interpolation_method);
        m_faHalf.construct( bhat, g, dg::NEU, dg::NEU, dg::geo::NoLimiter(),
            p.rk4eps, p.mx, p.my, 2.*M_PI/(double)p.Nz/2., p.interpolation_method );
    }
    dg::assign(  dg::pullback(dg::geo::Divb(mag), g), m_divb);
}
template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::compute_staggered_densities(
        const std::array<std::vector<Container>,6>& y,
        std::map<std::string, std::vector<Container>>& q
)
{
    for( unsigned s=0; s<m_p.num_species; s++)
    {
        std::vector<const Container*> in = {
            &y[0][s], &y[1][s], &y[2][s]}; // "N", "Pperp", "Ppara"
        std::vector<std::string> out = {"ST N", "ST Pperp", "ST Ppara"};
        for( unsigned u=0; u<in.size(); u++)
        {
            // -1/2,+1/2
            m_faHalf( dg::geo::zeroMinus, *in[u], q.at(out[u]+" -1/2")[s]);
            m_faHalf( dg::geo::einsPlus,  *in[u], q.at(out[u]+" +1/2")[s]);
            update_parallel_bc_1st( q.at(out[u]+" -1/2")[s], q.at(out[u]+" +1/2")[s], m_p.bcx, 0.);
        }

        // Note the sequence in which variables are transformed!
        dg::blas1::axpby( 0.5, q.at("ST N -1/2")[s], 0.5, q.at("ST N +1/2")[s], q.at("ST N")[s]);
        dg::blas1::axpby( 0.5, q.at("ST Pperp -1/2")[s], 0.5, q.at("ST Pperp +1/2")[s], m_temp);
        dg::blas1::pointwiseDivide( m_temp, q.at("ST N")[s], q.at("ST Tperp")[s]);
        dg::blas1::pointwiseDivide(  y[4][s], m_temp,     q.at("ST Uperp")[s]); // UperpST
        dg::blas1::axpby( 0.5, q.at("ST Ppara -1/2")[s], 0.5, q.at("ST Ppara +1/2")[s], m_temp);
        dg::blas1::pointwiseDivide( m_temp, q.at("ST N")[s], q.at("ST Tpara")[s]);
        dg::blas1::pointwiseDivide(  y[5][s], m_temp,     q.at("ST Upara")[s]); // UparaST
        // "ST ds Tperp" and "ST ds Tpara"
        std::vector<std::string> in2 = {"Tperp", "Tpara"};
        for( unsigned u=0; u<in2.size(); u++)
        {
            m_faHalf( dg::geo::zeroMinus, q.at(in2[u])[s], m_tminus);
            m_faHalf( dg::geo::einsPlus,  q.at(in2[u])[s], m_tplus);
            update_parallel_bc_1st( m_tminus, m_tplus, m_p.bcx, 0.);
            dg::geo::ds_centered( m_faHalf, 1., m_tminus, m_tplus, 0., q.at("ST ds "+in2[u])[s]);
        }
    }
}

template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::compute_parallel_transformations(
        const std::array<std::vector<Container>,6>& y,
        std::map<std::string, std::vector<Container>>& q
)
{
    for( unsigned s=0; s<m_p.num_species; s++)
    {
        std::vector<std::string> in = { "Psi0", "Psi1", "Psi2", "Psi3"}; // -> "ST Psi0", ...
        // 0 transform psi
        for( unsigned u=0; u<in.size(); u++)
        {
            m_faHalf( dg::geo::zeroMinus, q.at(in[u])[s], m_tminus);
            m_faHalf( dg::geo::einsPlus,  q.at(in[u])[s], m_tplus);
            update_parallel_bc_1st( m_tminus, m_tplus, m_p.bcxP, 0.);
            dg::blas1::axpby( 0.5, m_tminus, 0.5, m_tplus, q.at("ST "+in[u])[s]);
            if( u == 0)
                dg::geo::ds_centered( m_faHalf, 1., m_tminus, m_tplus, 0., q.at("ST ds Psi0")[s]);
            if( u == 1)
                dg::geo::ds_centered( m_faHalf, 1., m_tminus, m_tplus, 0., q.at("ST ds Psi1")[s]);
            if( u==1)
            {
                m_fa( dg::geo::einsMinus, q.at(in[u])[s], m_tminus);
                m_fa( dg::geo::zeroForw,  q.at(in[u])[s], m_temp);
                m_fa( dg::geo::einsPlus,  q.at(in[u])[s], m_tplus);
                update_parallel_bc_2nd( m_fa, m_tminus, m_temp, m_tplus, m_p.bcxP,
                    0.);
                dg::geo::ds_centered( m_fa, 1., m_tminus, m_tplus, 0., q.at("ds Psi1")[s]);
            }
        }
        // 2nd transform all 0, +1, -1
        in = std::vector<std::string>{
            "N", "Tperp", "Tpara", "ST U", "ST Uperp", "ST Upara"};
        for( unsigned u=0; u<in.size(); u++)
        {
            m_fa( dg::geo::einsMinus, q.at(in[u])[s], q.at(in[u]+" -1")[s]);
            m_fa( dg::geo::zeroForw,  q.at(in[u])[s], q.at(in[u]+" 0")[s]);
            m_fa( dg::geo::einsPlus,  q.at(in[u])[s], q.at(in[u]+" +1")[s]);
            update_parallel_bc_2nd( m_fa, q.at(in[u]+" -1")[s],
                q.at(in[u]+" 0")[s], q.at(in[u]+" +1")[s], m_p.bcx, 0.);
        }
        // 3rd transform velocities
        std::vector<const Container*> inp = {
            &q.at("ST U")[s], &y[4][s], &y[5][s]}; // ST Qperp, ST Qpara
        std::vector<std::string> out = {"U", "Qperp", "Qpara"};
        for( unsigned u=0; u<inp.size(); u++)
        {
            m_faHalf( dg::geo::einsMinus, *inp[u], q.at(out[u]+" -1/2")[s]);
            m_faHalf( dg::geo::zeroPlus,  *inp[u], q.at(out[u]+" +1/2")[s]);
            update_parallel_bc_1st( q.at(out[u]+" -1/2")[s], q.at(out[u]+" +1/2")[s], m_p.bcx, 0.);
            if( u == 0)
                dg::blas1::axpby( 0.5, q.at(out[u]+" -1/2")[s], 0.5, q.at(out[u]+" +1/2")[s], q.at("U")[s]);
            if( u == 1)
            {
                dg::blas1::axpby( 0.5, q.at(out[u]+" -1/2")[s], 0.5, q.at(out[u]+" +1/2")[s], m_temp); // Qperp
                dg::blas1::pointwiseDivide( m_temp, y[1][s], q.at("Uperp")[s]); // Qperp/Pperp
            }
            if( u == 2)
            {
                dg::blas1::axpby( 0.5, q.at(out[u]+" -1/2")[s], 0.5, q.at(out[u]+" +1/2")[s], m_temp); // Qpara
                dg::blas1::pointwiseDivide( m_temp, y[2][s], q.at("Upara")[s]); // Qpara/Ppara
            }
        }
    }

}

template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::add_densities_advection(
        unsigned s,
        const std::array<std::vector<Container>,6>& y,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp)
{
    // "velocity-staggered" - "flux-centered"
    for( unsigned u=0; u<3; u++)
    {
        if( u == 0)
        {
            compute_parallel_flux( q.at("ST U")[s], q.at("ST N -1/2")[s], q.at("ST N +1/2")[s],
                m_temp);
        }
        else if( u==1)
        {
            dg::blas1::axpby( 1., q.at("ST U")[s], 1., q.at("ST Uperp")[s], m_tminus); // U + U_perp
            compute_parallel_flux( m_tminus, q.at("ST Pperp -1/2")[s], q.at("ST Pperp +1/2")[s],
                m_temp);
        }
        else if( u==2)
        {
            dg::blas1::axpby( 1., q.at("ST U")[s], 1., q.at("ST Upara")[s], m_tminus); // U + U_para
            compute_parallel_flux( m_tminus, q.at("ST Ppara -1/2")[s], q.at("ST Ppara +1/2")[s],
                m_temp);
        }
        m_faHalf( dg::geo::zeroPlus,  m_temp, m_tplus);
        m_faHalf( dg::geo::einsMinus, m_temp, m_tminus);
        update_parallel_bc_1st( m_tminus, m_tplus, dg::NEU, 0.);
        dg::geo::ds_divCentered( m_faHalf, 1., m_tminus, m_tplus, 0., m_divNUb[u][s]);
        dg::blas1::axpby( -1., m_divNUb[u][s], 1., yp[u][s]);
    }
    // -2P_para GradPar U
    dg::geo::ds_centered( m_faHalf, 1., q.at("U -1/2")[s], q.at("U +1/2")[s], 0., m_temp);
    dg::blas1::pointwiseDot( -2., y[2][s], m_temp, 1., yp[2][s]);
    // -2z N U_perp E_1,para
    dg::geo::ds_centered( m_fa, 1., q.at("Tperp -1")[s], q.at("Tperp +1")[s], 0., m_temp); //GradPar Tperp
    dg::blas1::pointwiseDivide( m_temp, q.at("Tperp")[s], m_temp);
    dg::blas1::axpby(1., m_divb, 1., m_temp);
    dg::blas1::pointwiseDot( -1., q.at("Psi2")[s], m_temp, +1., q.at("Psi1")[s], m_temp, 0., m_temp);
    dg::blas1::axpby( 1., q.at("ds Psi1")[s], 1., m_temp); // E_1,para
    dg::blas1::pointwiseDot( -2.*m_p.z[s], q.at("N")[s], q.at("Uperp")[s], m_temp, 1., yp[2][s]);
}

template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::add_velocities_advection(
        unsigned s,
        const std::array<std::vector<Container>,6>& y,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp)
{
    // "velocity-staggered" - "flux-centered"
    for( unsigned u=3; u<6; u++)
    {
        if( u==3)
            compute_parallel_flux( q.at("U")[s], q.at("U -1/2")[s], q.at("U +1/2")[s],
                m_temp);
        else if ( u == 4)
            compute_parallel_flux( q.at("U")[s], q.at("Qperp -1/2")[s], q.at("Qperp +1/2")[s],
                m_temp);
        else if ( u == 5)
            compute_parallel_flux( q.at("U")[s], q.at("Qpara -1/2")[s], q.at("Qpara +1/2")[s],
                m_temp);
        m_faHalf( dg::geo::einsPlus,  m_temp, m_tplus);
        m_faHalf( dg::geo::zeroMinus, m_temp, m_tminus);
        update_parallel_bc_1st( m_tminus, m_tplus, dg::NEU, 0.);
        if( u==3)
            dg::geo::ds_centered( m_faHalf, 0.5, m_tminus, m_tplus, 0, m_divNUb[3][s]);
        else
            dg::geo::ds_divCentered( m_faHalf, 1., m_tminus, m_tplus, 0., m_divNUb[u][s]);
        dg::blas1::axpby( -1., m_divNUb[u][s], 1., yp[u][s]);
    }
    dg::blas1::pointwiseDot( -1./m_p.mu[s], q.at("ST N")[s], q.at("ST Tpara")[s], q.at("ST ds Tperp")[s], 1., yp[4][s]);
    dg::blas1::pointwiseDot( -3./m_p.mu[s], q.at("ST N")[s], q.at("ST Tpara")[s], q.at("ST ds Tpara")[s], 1., yp[5][s]);

    dg::geo::ds_centered( m_fa, 1., q.at("ST U -1")[s], q.at("ST U +1")[s], 0., m_temp); //ST ds U
    dg::blas1::pointwiseDot( -1., y[4][s], m_temp, 1., yp[4][s]);
    dg::blas1::pointwiseDot( -3., y[5][s], m_temp, 1., yp[5][s]);

    // Add pressure gradient
    double z = m_p.z[s], mu = m_p.mu[s], delta = m_fa.deltaPhi();
    dg::blas1::subroutine( [mu, delta ]DG_DEVICE (
        double& WDot, double N_mh, double N_ph,
        double Ppara_mh, double Ppara_ph, double bphi)
        {
            WDot -= bphi*(Ppara_ph-Ppara_mh)/delta/2.*(1/N_ph + 1/N_mh)/mu;
        },
        yp[3][s], q.at("ST N -1/2")[s], q.at("ST N +1/2")[s],
        q.at("ST Ppara -1/2")[s], q.at("ST Ppara +1/2")[s], m_fa.bphi()
    );
    // and parallel electric field
    dg::blas1::subroutine( [z, mu, delta ]DG_DEVICE (
        double& WDot, double& QperpDot,
        double N, double Tperp, double dsTperp,
        double dsG0, double dsG1,
        double G1, double G2, double divb
            )
        {
            WDot     -= z/mu*(dsG0 - G1*(dsTperp/Tperp + divb));
            QperpDot -= z/mu*N*Tperp*(dsG1 - (G2-G1)*(dsTperp/Tperp + divb));
        },
        yp[3][s], yp[4][s],
        q.at("ST N")[s], q.at("ST Tperp")[s], q.at("ST ds Tperp")[s],
        q.at("ST ds Psi0")[s], q.at("ST ds Psi1")[s],
        q.at("ST Psi1")[s], q.at("ST Psi2")[s], m_divb
    );
}

template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::add_densities_diffusion(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp)
{
    std::vector<std::string> in = {"N", "Tperp", "Tpara"};
    for( unsigned u=0; u<3; u++)
        dg::geo::dssd_centered( m_fa, m_p.nu_parallel[u],
            q.at(in[u]+" -1")[s], q.at(in[u]+" 0")[s], q.at(in[u]+" +1")[s], 1., yp[u][s]);
    // Add para temp generation through friction
    dg::geo::ds_centered( m_faHalf, 1., q.at("U -1/2")[s], q.at("U +1/2")[s], 0., m_temp); //dsU
    dg::blas1::pointwiseDot( +2.*m_p.mu[s]*m_p.nu_parallel[3], m_temp, m_temp, 1., yp[2][s]);
}
template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::add_velocities_diffusion(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp)
{
    // Add parallel viscosity
    dg::geo::dssd_centered( m_fa, m_p.nu_parallel[3],
        q.at("ST U -1")[s], q.at("ST U 0")[s], q.at("ST U +1")[s], 0., m_temp);
    dg::blas1::pointwiseDivide( 1., m_temp, q.at("ST N")[s], 1., yp[3][s]);
    dg::geo::dssd_centered( m_fa, m_p.nu_parallel[4],
        q.at("ST Uperp -1")[s], q.at("ST Uperp 0")[s], q.at("ST Uperp +1")[s], 1., yp[4][s]);
    dg::geo::dssd_centered( m_fa, m_p.nu_parallel[5],
        q.at("ST Upara -1")[s], q.at("ST Upara 0")[s], q.at("ST Upara +1")[s], 1.,  yp[5][s]);

    // Add density gradient correction
    double delta = m_fa.deltaPhi(), nu = m_p.nu_parallel[0];
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
            yp[3][s], q.at("ST N -1/2")[s], q.at("ST N +1/2")[s], q.at("ST U -1")[s], q.at("ST U 0")[s],
            q.at("ST U +1")[s], m_fa.bphi()
    );
}

template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::add_sheath_neumann_terms(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        const std::array<std::vector<Container>,6>& y,
        std::array<std::vector<Container>,6>& yp)
{
    // add sheath boundary conditions
    if( m_sheath_rate != 0)
    {
        ////densities sheath
        ////Here, we need to find out where "downstream" is
        // !! Simulations do not really work without
        std::vector<std::string> in = {"N", "Tperp", "Tpara", "ST U", "ST Uperp", "ST Upara"};
        for( int i=0; i<6; i++)
        {
            if( i == 3) continue;
            //The coordinate automatically sees the reversed field
            //but m_plus and m_minus are defined wrt the angle coordinate
            if( m_reversed_field) //bphi negative (exchange + and -)
                dg::blas1::evaluate( m_temp, dg::equals(), dg::Upwind(),
                     m_sheath_coordinate, q.at(in[i]+" +1")[s], q.at(in[i]+" -1")[s]);
            else
                dg::blas1::evaluate( m_temp, dg::equals(), dg::Upwind(),
                     m_sheath_coordinate, q.at(in[i]+" -1")[s], q.at(in[i]+" +1")[s]);
            if( i == 1 or i == 2 )
                dg::blas1::pointwiseDot( q.at("N")[s], m_temp, m_temp); // P = NT
            else if( i == 4)
                dg::blas1::pointwiseDot( 1., q.at("ST N")[s], q.at("ST Tperp")[s], m_temp, 0., m_temp); // Q = NTU
            else if( i == 5)
                dg::blas1::pointwiseDot( 1., q.at("ST N")[s], q.at("ST Tpara")[s], m_temp, 0., m_temp); // Q = NTU

            dg::blas1::pointwiseDot( m_sheath_rate, m_temp,  m_sheath,
                                    -m_sheath_rate, y[i][s], m_sheath,
                                     1., yp[i][s]);
        }
    }
}
template<class Grid, class IMatrix, class Matrix, class Container>
void ParaDynamics<Grid, IMatrix, Matrix, Container>::add_sheath_velocity_terms(
    unsigned s,
    const std::map<std::string, std::vector<Container>>& q,
    std::array<std::vector<Container>,6>& yp)
{
    // Velocity sheath
    if( m_sheath_rate != 0)
    {
        if( "wall" == m_p.sheath_bc)
        {
            // No densities wall sheath boundary
            dg::blas1::axpby( +m_sheath_rate*m_p.uwall, m_sheath, 1., yp[3][s] );
        }
        else if( "insulating" == m_p.sheath_bc)
        {
            //velocity c_s
            if( s == 0) // electron species
            {
                // TODO: Is it better to assume toroidal alignment of T and N in sheath?
                dg::blas1::copy( 0, m_temp);
                for( unsigned k=1; k<m_p.num_species; k++)
                {
                    double zk = m_p.z[k], mk = m_p.mu[k];
                    dg::blas1::evaluate( m_temp, dg::plus_equals(),
                        [ zk,  mk ] DG_DEVICE( double nk, double ne, double tk, double te)
                        {
                            double ck = sqrt( ( tk + zk * te )/mk);
                            return zk * nk / ne * ck;
                        }, q.at("ST N")[k], q.at("ST N")[0], q.at("ST Tpara")[k], q.at("ST Tpara")[0]);
                }
            }
            else
            {
                double zs = m_p.z[s], ms = m_p.mu[s];
                dg::blas1::evaluate( m_temp, dg::equals(),
                    [ zs, ms] DG_DEVICE( double ts, double te)
                    {
                        return sqrt( ( ts + zs * te )/ms);
                    }, q.at("ST Tpara")[s], q.at("ST Tpara")[0]);
            }
            dg::blas1::pointwiseDot( m_sheath_rate, m_sheath,
                m_sheath_coordinate, m_temp, 1.,  yp[3][s]);
        }
        // Apply to U, not W
        dg::blas1::pointwiseDot( -m_sheath_rate, m_sheath, q.at("ST U")[s],
            1., yp[3][s]);
    }
}

}//namespace thermal
