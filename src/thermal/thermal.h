#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"

#include "../feltor/common.h"

#include "parameters.h"
#include "solvers.h"
#include "perpendicular.h"
#include "parallel.h"
#include "collisions.h"
#include "sources.h"


namespace thermal
{

template< class Geometry, class IMatrix, class Matrix, class Container >
struct Explicit
{
    // y[0] == N
    // y[1] == Pperp
    // y[2] == Ppara
    // y[3] == ST W
    // y[4] == ST Qperp
    // y[5] == ST Qpara
    using vector = std::array<std::vector<Container>,4>;
    using container = Container;
    Explicit( const Geometry& g, thermal::Parameters p,
        dg::geo::TokamakMagneticField mag, dg::file::WrappedJsonValue js );

    void operator()( double t,
        const std::array<std::vector<Container>,6>& y,
        std::array<std::vector<Container>,6>& yp);
    /// ///////////////////DIAGNOSTIC MEMBERS //////////////////////
    /// 3d output
    const Container& potential() const {
        return m_phi;
    }
    const Container& apar() const {
        return m_apar;
    }
    const Container& BperpX() const {
        return m_BperpX;
    }
    const Container& BperpY() const {
        return m_BperpY;
    }
    const Container& get(std::string id, unsigned s) { return m_q.at(id)[s];}
    void compute_dot_apar( Container& tmp) const {
        m_old_apar.derive( tmp);
    }
    const PerpDynamics<Geometry, IMatrix, Matrix, Container>& perp() const { return m_perp;}
    const ParaDynamics<Geometry, IMatrix, Matrix, Container>& para() const { return m_para;}
    const Sources<Geometry, IMatrix, Matrix, Container>& sources() const { return m_sources;}
    const Solvers<Geometry, Matrix, Container>& solvers() const { return m_solvers;}
    const Collisions<Geometry, Matrix, Container>& collisions() const { return m_collisions;}

    unsigned called() const { return m_called;}
    /// ///////////////END DIAGNOSTIC MEMBERS //////////////////////

    void set_source(
        bool fixed_profile, // cannot be mixed among species or equations due to quasineutrality and transformation
        const std::array<std::vector<double>,3>& source_rate,
        const std::array<std::vector<dg::x::HVec>,3>& profile, // for influx this can be ignored
        const std::array<std::vector<dg::x::HVec>,3>& source,  // for fixed profile this contains damping
        const std::vector<double>& minne,
        double minrate,
        const std::vector<double>& minalpha)
    {
        m_sources.set_source( fixed_profile, source_rate, profile, source,
            minne, minrate, minalpha);
    }
    void set_wall(const Container& wall)
    {
        m_sources.set_wall( wall);
    }

    void set_sheath(double sheath_rate, const Container& sheath,
            const Container& sheath_coordinate)
    {
        m_para.set_sheath( sheath_rate, sheath, sheath_coordinate);
    }
    // Called in Feltor init
    const dg::geo::Fieldaligned<Geometry, IMatrix, Container>& fieldaligned() const
    {
        return m_para.fieldaligned();
    }

  private:
    PerpDynamics<Geometry, IMatrix, Matrix, Container> m_perp;
    ParaDynamics<Geometry, IMatrix, Matrix, Container> m_para;
    Solvers<Geometry, Matrix, Container> m_solvers;
    Collisions<Geometry, Matrix, Container> m_collisions;
    Sources<Geometry, IMatrix, Matrix, Container> m_sources;

    dg::Extrapolation<Container> m_old_apar; // for diagnostics

    std::map<std::string, std::vector<Container>> m_q;

    const std::vector<std::string> q_names = {
        "N",     "N 0",     "N +1",     "N -1",
        "Tperp", "Tperp 0", "Tperp +1", "Tperp -1",
        "Tpara", "Tpara 0", "Tpara +1", "Tpara -1",
        "Psi0",
        "Psi1", "ds Psi1",
        "U", "U +1/2", "U -1/2",
        "Uperp",
        "Upara",
        "Qperp +1/2", "Qperp -1/2",
        "Qpara +1/2", "Qpara -1/2",
        // Staggered variables
        "ST N", "ST N +1/2", "ST N -1/2",
        "ST Tperp", "ST ds Tperp",
        "ST Tpara", "ST ds Tpara",
        "ST Pperp +1/2", "ST Pperp -1/2",
        "ST Ppara +1/2", "ST Ppara -1/2",
        "ST Psi0", "ST ds Psi0",
        "ST Psi1", "ST ds Psi1",
        "ST U",     "ST U 0",     "ST U +1",     "ST U -1",
        "ST Uperp", "ST Uperp 0", "ST Uperp +1", "ST Uperp -1",
        "ST Upara", "ST Upara 0", "ST Upara +1", "ST Upara -1",
        // perp derivatives
        "dxF N",     "dxB N",     "dyF N",     "dyB N",
        "dxF Pperp", "dxB Pperp", "dyF Pperp", "dyB Pperp",
        "dxF Ppara", "dxB Ppara", "dyF Ppara", "dyB Ppara",
        "dx Tperp", "dy Tperp",
        "dx Tpara", "dy Tpara",
        "dx Psi0",  "dy Psi0",
        "dx Psi1",  "dy Psi1",
        "dx U",     "dy U",
        "dx Uperp", "dy Uperp",
        "dx Upara", "dy Upara",
        // Staggered perp derivatives
        "ST dx N",     "ST dy N",
        "ST dx Tperp", "ST dy Tperp",
        "ST dx Tpara", "ST dy Tpara",
        "ST dx Psi0",  "ST dy Psi0",
        "ST dx Psi1",  "ST dy Psi1",
        "ST dxF U",     "ST dxB U",     "ST dyF U",     "ST dyB U",
        "ST dxF Qperp", "ST dxB Qperp", "ST dyF Qperp", "ST dyB Qperp",
        "ST dxF Qpara", "ST dxB Qpara", "ST dyF Qpara", "ST dyB Qpara"
    };

    Container m_phi, m_apar, m_BperpX, m_BperpY, m_aparST, m_BperpXST, m_BperpYST; // same for all species

    const thermal::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    std::vector<bool> m_upToDate;

    unsigned m_called = 0;
};

template<class Grid, class IMatrix, class Matrix, class Container>
Explicit<Grid, IMatrix, Matrix, Container>::Explicit( const Grid& g,
    thermal::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue js
    ):
    m_perp( g, p, mag, js),
    m_para( g, p, mag, js),
    m_solvers( g, p, mag, js),
    m_collisions( g, p, mag, js),
    m_sources( g, p, mag, js),
    m_old_apar( 2, dg::evaluate( dg::zero, g)),
    m_p(p), m_js(js)
{
    //--------------------------init vectors to 0-----------------//
    dg::assign( dg::evaluate( dg::zero, g), m_phi );
    m_apar = m_BperpX = m_BperpY = m_BperpXST = m_BperpYST = m_aparST = m_phi;
    m_q["N"] = std::vector<Container>( m_p.num_species, m_phi);
    for( auto name : q_names)
        m_q[name] = m_q["N"];
    //--------------------------Construct-------------------------//
    m_upToDate.resize( m_p.num_species);
    for( unsigned s=0; s<m_p.num_species; s++)
        m_upToDate[s] = false;

}


template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::operator()(
    double t,
    const std::array<std::vector<Container>,6>& y,
    std::array<std::vector<Container>,6>& yp)
{
    m_called++;
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    //DG_RANK0 std::cout << "## time "<<time<<" dt "<<dt<<" t_out "<<t_output<<" step "<<step<<" failed "<<var.nfailed<<"\n";
    DG_RANK0 std::cout << "## time "<<t<<"\n";

    dg::Timer timer;
    double accu = 0.;//accumulated time
    timer.tic();


    const std::vector<Container>&  density    = y[0];
    const std::vector<Container>&  pperp      = y[1];
    const std::vector<Container>&  ppara      = y[2];
    const std::vector<Container>&  wST        = y[3];
    //const std::vector<Container>&  qperpST    = y[4];
    //const std::vector<Container>&  qparaST    = y[5];


    // 1. Transform Pperp, Ppara to Tperp, Tpara for all species
    dg::blas1::copy( y[0], m_q.at("N"));
    dg::blas1::pointwiseDivide( pperp, density, m_q.at("Tperp"));
    dg::blas1::pointwiseDivide( ppara, density, m_q.at("Tpara"));

    //2. Solve for potential phi given density and temperature

    m_solvers.compute_phi( t, density, pperp, m_phi, m_p.penalize_wall,
        m_sources.get_wall(), m_p.penalize_sheath, m_para.get_sheath());

    m_solvers.compute_psi( t, m_phi, m_q.at("Psi0"), m_q.at("Psi1"));

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute phi and psi               took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );
    // Given densities transform to fieldaligned grids:
    // (Guaranteed to compute "ST N")
    m_para.compute_staggered_densities( y, m_q);

    // Compute m_aparST if beta != 0
    if( m_p.beta != 0)
    {
        m_solvers.compute_aparST( t, m_q.at("ST N"), wST,
            m_aparST, true);
    }
    m_para.compute_apar( m_aparST, m_apar);
    m_old_apar.update( t, m_apar);
    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute Apar and staggered N      took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic();
    for( unsigned s=0; s<m_p.num_species; s++)
        dg::blas1::axpby( 1., wST[s], -m_p.z[s]/m_p.mu[s], m_aparST, m_q.at("ST U")[s]);

    // Compute all the rest of parallel trafos
    m_para.compute_parallel_transformations( y, m_q);
    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute Parallel transformations  took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic();
    // and the perp derivatives
    m_perp.update_derivatives( m_apar, m_BperpX, m_BperpY, y, m_q);
    m_perp.update_STderivatives( m_aparST, m_BperpXST, m_BperpYST, y, m_q);
    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute perpendicular derivatives took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic();
    // main species loop
    dg::blas1::copy( 0., yp);
    for( unsigned s=0; s<m_p.num_species; s++)
    {
        m_upToDate[s] = false;
        // Add perp dynamics
        m_perp.add_densities_advection(  s, m_apar, m_BperpX, m_BperpY,
            m_q, yp);
        m_perp.add_velocities_advection( s, m_aparST, m_BperpXST, m_BperpYST,
            m_q, yp);

        m_perp.add_densities_diffusion(  s, m_q, yp);
        m_perp.add_velocities_diffusion( s, m_q, yp);

        // Add parallel dynamics
        m_para.add_densities_advection(  s, y, m_q, yp);
        m_para.add_velocities_advection( s, y, m_q, yp);

        m_para.add_densities_diffusion( s, m_q, yp);
        m_para.add_velocities_diffusion( s, m_q, yp);

        // Add collisions
        m_collisions.add_coulomb_collisions( s, m_q, yp);
        m_collisions.add_lorentz_collisions( s, m_q, y, yp);

        // Multiply penalization (must come before adding wall and sheath terms!)
        // F*(1-chi_w-chi_s)
        for( unsigned u=0; u<6; u++)
            common::multiply_rhs_penalization( yp[u][s], m_p.penalize_wall,
                m_sources.get_wall(), m_p.penalize_sheath, m_para.get_sheath());

        // -w_sh chi_sh ( y - y_sh)
        // -w_w chi_w ( y - y_w)
        m_para.add_sheath_neumann_terms( s, m_q, y, yp);
        m_para.add_sheath_velocity_terms( s, m_q, yp);
        m_sources.add_wall_terms( s, m_q, y, yp);

        // And sources
        m_sources.add_source_terms( s, m_phi, m_q, y, yp );
    }

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout <<"## Main species loop perp and para   took "
                       << timer.diff() << "s\t A: "<<accu<<"s\n";
}



} //namespace thermal
