#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "parameters.h"
#include "common.h"
#include "solvers.h"
#include "perpendicular.h"
#include "parallel.h"
#include "sources.h"

#define FELTORPARALLEL 1
#define FELTORPERP 1

#ifdef WRITE_POL_FILE
int counter = 0;
dg::file::NcFile pol_file;
#endif // WRITE_POL_FILE


//Latest measurement: m = 10.000 per step

namespace feltor
{

template< class Geometry, class IMatrix, class Matrix, class Container >
struct Explicit
{
    using vector = std::array<std::array<Container,2>,2>;
    using container = Container;
    Explicit( const Geometry& g, feltor::Parameters p,
        dg::geo::TokamakMagneticField mag, dg::file::WrappedJsonValue js );

    //Given N_i initialize n_e such that phi=0
    void initializene( const Container& ni, Container& ne, std::string initphi);
    //Given n_e initialize N_i such that phi=0
    void initializeni( const Container& ne, Container& ni, std::string initphi);

    void operator()( double t,
        const std::array<std::array<Container,2>,2>& y,
        std::array<std::array<Container,2>,2>& yp);
    void implicit( double t,
        const std::array<std::array<Container,2>,2>& y,
        std::array<std::array<Container,2>,2>& yp);
    /// ///////////////////RESTART    MEMBERS //////////////////////
    const Container& restart_density(int i)const{
        return m_q.at("N")[i];
    }
    const Container& restart_velocity(int i)const{
        return m_q.at("ST U")[i];
    }
    const Container& restart_aparallel() const {
        return m_aparST;
    }
    /// ///////////////////DIAGNOSTIC MEMBERS //////////////////////
    const Geometry& grid() const {
        return m_solvers.grid();
    }
    //potential[0]: electron potential, potential[1]: ion potential
    const Container& uE2() const {
        return m_solvers.uE2();
    }
    const Container& density(int i)const{
        return m_q.at("N")[i];
    }
    const Container&  gammaNi() const{
        if( m_p.tau[1] == 0)
            return m_q.at("N")[1];
        return m_solvers.old_gammaN_head();
    }
    const Container&  gammaPhi() const{
        if( m_p.tau[1] == 0)
            return m_q.at("Psi")[0];
        return m_solvers.old_psi_head();
    }


    const Container& density_source(int i)const{
        return m_sources.get_density_source( i);
    }
    const Container& velocity(int i)const{
        return m_q.at("U")[i];
    }
    const Container& potential(int i) const {
        return m_q.at("Psi")[i];
    }
    const Container& aparallel() const {
        return m_apar;
    }
    const std::array<Container, 3> & gradN (int i) {
        update_diag();
        // m_gradN is updated to the diff_dir direction derivative
        return m_gradN[i];
    }
    const std::array<Container, 3> & gradU (int i) {
        update_diag();
        // m_gradU is updated to the diff_dir direction derivative
        return m_gradU[i];
    }
    const std::array<Container, 3> & gradP (int i) {
        update_diag();
        return m_gradP[i];
    }
    const std::array<Container, 3> & gradA () {
        update_diag();
        return m_dA;
    }
    const Container& divNUb( int i) const{
        return m_para.get_divNUb(i);
    }
    const Container& dsN (int i) {
        update_diag();
        return m_dsN[i];
    }
    const Container& dsU (int i) {
        update_diag();
        return m_dsU[i];
    }
    const Container& dsP(int i) {
        update_diag();
        return m_dsP[i];
    }
    const Container& dssU(int i){
        update_diag();
        return m_dssU[i];
    }
    const Container& lapParU( unsigned i) {
        update_diag();
        return m_lapParU[i];
    }
    const Container& lapParN( unsigned i) {
        update_diag();
        return m_lapParN[i];
    }
    void compute_gradN( const Container& in, std::array<Container,3>& gradN) const{
        m_perp.compute_gradN( in, gradN);
    }
    void compute_dot_aparallel( Container& tmp) const {
        m_old_apar.derive( tmp);
    }
    const dg::SparseTensor<Container>& projection() const{
        return m_perp.projection();
    }
    const std::array<Container, 3> & curv () const {
        return m_perp.curv();
    }
    const std::array<Container, 3> & curvKappa () const {
        return m_perp.curvKappa();
    }
    const Container& divCurvKappa() const {
        return m_perp.divCurvKappa();
    }
    // Covariant phi component of bhat \approx \pm R
    const Container& bphi( ) const { return m_perp.bphi(); }
    const Container& binv( ) const { return m_perp.binv(); }
    const Container& divb( ) const { return m_perp.divb(); }
    //volume with dG weights
    const Container& vol3d() const { return m_perp.weights();}
    const Container& weights() const { return m_perp.weights();}
    //bhat / sqrt{g} / B
    const std::array<Container, 3> & bhatgB () const {
        // covariant components
        return m_perp.bhatgB();
    }
    void compute_perp_diffusiveN( double alpha, const Container& density,
            Container& temp0, Container& temp1, double beta, Container& result ) const
    {
        return m_perp.compute_diffusiveN( alpha, density, temp0, temp1, beta, result);
    }
    void compute_perp_diffusiveU( double alpha, const Container& velocity,
            const Container& density,
            Container& temp0, Container& temp1, Container& temp2, Container& temp3, double beta, Container& result) const
    {
        return m_perp.compute_diffusiveU( alpha, velocity, density, temp0, temp1, temp2, temp3, beta, result);
    }
    void compute_lapMperpN (double alpha, const Container& density, Container& temp0, double beta, Container& result) const
    {
        m_perp.compute_lapMperpN( alpha, density, temp0, beta, result);
    }
    void compute_lapMperpU (int i, Container& result)
    {
        m_perp.compute_lapMperpU( m_q.at("U")[i], result);
    }
    void compute_lapMperpP (int i, Container& result)
    {
        m_sources.compute_lapMperpP( m_q.at("Psi")[i], result);
    }
    void compute_lapMperpA ( Container& result)
    {
        m_perp.compute_lapMperpU(m_apar, result);
    }
    void compute_bperp( std::array<Container,3>& bperp)
    {
        update_diag();
        m_perp.compute_bperp( m_apar, m_dA, bperp);
    }
    const Container& get_source() const{
        return m_sources.get_source();
    }
    const Container& get_source_prof() const{
        return m_sources.get_source_prof();
    }
    const Container& get_wall() const{
        return m_sources.get_wall();
    }
    const Container& get_sheath() const{
        return m_para.get_sheath();
    }
    const Container& get_sheath_coordinate() const{
        return m_para.get_sheath_coordinate();
    }

    void compute_parallel_diffusiveN( int i, Container& result)
    {
        dg::blas1::axpby( m_p.nu_parallel_n, lapParN(i), 0., result);
    }
    void compute_parallel_diffusiveU( int i, Container& result)
    {
        double nu = m_p.nu_parallel_n;
        if( m_p.modify_diff)
            nu += m_p.nu_parallel_u[i];
        if( nu > 0)
        {
            dg::blas1::pointwiseDot( dsN(i), dsU(i), result);
            dg::blas1::pointwiseDivide( nu, result, density(1), 0., result);
        }
        else
            dg::blas1::copy( 0, result);
        if( m_p.nu_parallel_u[i] > 0)
        {
            if( !m_p.modify_diff)
                dg::blas1::pointwiseDivide( m_p.nu_parallel_u[i], lapParU(i), density(i), 1., result);
            else
                dg::blas1::axpby( m_p.nu_parallel_u[i], lapParU(i), 1., result);
        }
    }


    template<class Container2>
    void centered_div( const Container2& prefactor,
            const std::array<Container, 3>& contra_vec,
            Container& temp0, Container& result)
    {
        m_perp.centered_div( prefactor, contra_vec, temp0, result);
    }
    void centered_v_dot_nabla( const std::array<Container, 3>& contra_vec,
            const Container& f, Container& temp1, Container& result)
    {
        m_perp.centered_v_dot_nabla( contra_vec, f, temp1, result);
    }
    void compute_pol( double alpha, const Container& density, Container& temp, double beta, Container& result)
    {
        m_solvers.compute_pol( alpha, density, m_q.at("Psi")[0], temp, beta, result);
    }
    void compute_source_pol( double alpha, const Container& density, Container& temp, double beta, Container& result)
    {
        m_sources.compute_source_pol( alpha, density, m_q.at("Psi")[0], temp, beta, result);
    }
    unsigned called() const { return m_called;}

    /// //////////////////////DIAGNOSTICS END////////////////////////////////
    void update_diag(){
        // assume m_density, m_q.at("Psi"), m_velocity, m_velocityST, m_apar
        // compute dsN, dsU, dsP, lapParU, dssU and perp derivatives
        if( !m_upToDate)
        {
            //update_perp_derivatives( m_density, m_velocity, m_q.at("Psi"), m_apar);
            for( unsigned i=0; i<2; i++)
            {
                // density m_dsN, m_lapParN
                //m_para.fieldaligned()( dg::geo::einsMinus, m_density[i], m_minus);
                //m_para.fieldaligned()( dg::geo::zeroForw,  m_density[i], m_zero);
                //m_para.fieldaligned()( dg::geo::einsPlus,  m_density[i], m_plus);
                //update_parallel_bc_2nd( m_para.fieldaligned(), m_minus, m_zero, m_plus,
                //        m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
                dg::geo::ds_centered( m_para.fieldaligned(), 1., m_q.at("N -1")[i], m_q.at("N +1")[i], 0., m_dsN[i]);
                dg::geo::dssd_centered( m_para.fieldaligned(), 1.,
                        m_minus, m_zero, m_plus, 0., m_lapParN[i]);
                // potential m_dsP
                m_para.fieldaligned()( dg::geo::einsMinus, m_q.at("Psi")[i], m_minus);
                m_para.fieldaligned()( dg::geo::einsPlus,  m_q.at("Psi")[i], m_plus);
                m_para.update_parallel_bc_2nd( m_para.fieldaligned(), m_minus, m_q.at("Psi")[i], m_plus,
                        m_p.bcxP, 0.);
                dg::geo::ds_centered( m_para.fieldaligned(), 1., m_minus, m_plus, 0., m_dsP[i]);
                // velocity m_dssU, m_lapParU m_dsU
                m_para.fieldaligned()( dg::geo::einsMinus, m_q.at("U")[i], m_minus);
                m_para.fieldaligned()( dg::geo::zeroForw,  m_q.at("U")[i], m_zero);
                m_para.fieldaligned()( dg::geo::einsPlus,  m_q.at("U")[i], m_plus);
                m_para.update_parallel_bc_2nd( m_para.fieldaligned(), m_minus, m_zero, m_plus,
                        m_p.bcxU, 0.);
                dg::geo::dssd_centered( m_para.fieldaligned(), 1.,
                        m_minus, m_zero, m_plus, 0., m_lapParU[i]);
                dg::geo::dss_centered( m_para.fieldaligned(), 1., m_minus,
                    m_zero, m_plus, 0., m_dssU[i]);
                dg::geo::ds_centered( m_para.fieldaligned(), 1., m_minus, m_plus, 0.,
                        m_dsU[i]);
            }
            for( unsigned i=0; i<2; i++)
            {
                dg::blas1::copy( m_q.at("dx Psi")[i], m_gradP[i][0]);
                dg::blas1::copy( m_q.at("dy Psi")[i], m_gradP[i][1]);
                dg::blas1::copy( m_q.at("dz Psi")[i], m_gradP[i][2]);

                dg::blas1::copy( m_q.at("dx U")[i], m_gradU[i][0]);
                dg::blas1::copy( m_q.at("dy U")[i], m_gradU[i][1]);
                dg::blas1::copy( m_q.at("dz U")[i], m_gradU[i][2]);

                // update m_gradN to the diff_dir direction derivative
                if( m_p.diff_dir == dg::forward)
                {
                    dg::blas1::copy( m_q.at("dxF N")[i], m_gradN[i][0]);
                    dg::blas1::copy( m_q.at("dyF N")[i], m_gradN[i][1]);
                    dg::blas1::copy( m_q.at("dzF N")[i], m_gradN[i][2]);
                }
                else if( m_p.diff_dir == dg::backward)
                {
                    dg::blas1::copy( m_q.at("dxB N")[i], m_gradN[i][0]);
                    dg::blas1::copy( m_q.at("dyB N")[i], m_gradN[i][1]);
                    dg::blas1::copy( m_q.at("dzB N")[i], m_gradN[i][2]);
                }
                else
                {
                    dg::blas1::axpby( 1./2., m_q.at("dxB N")[i], 1./2., m_q.at("dxF N")[i], m_gradN[i][0]);
                    dg::blas1::axpby( 1./2., m_q.at("dyB N")[i], 1./2., m_q.at("dyF N")[i], m_gradN[i][1]);
                    dg::blas1::axpby( 1./2., m_q.at("dzB N")[i], 1./2., m_q.at("dzF N")[i], m_gradN[i][2]);
                }
            }
            m_upToDate = true;
        }

    }

    //source strength, profile - 1
    void set_source( bool fixed_profile, const Container& profile, double source_rate, const Container& source, double minne, double minrate, double minalpha)
    {
        m_sources.set_source( fixed_profile, profile, source_rate, source, minne, minrate, minalpha);
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
    const dg::geo::Fieldaligned<Geometry, IMatrix, Container>& fieldaligned() const
    {
        return m_para.fieldaligned();
    }
  private:
    Solvers<Geometry, Matrix, Container> m_solvers;
    PerpDynamics<Geometry, Matrix, Container> m_perp;
    ParaDynamics<Geometry, IMatrix, Matrix, Container> m_para;
    Sources<Geometry, Matrix, Container> m_sources;


    std::map<std::string, std::array<Container,2>> m_q;

    const std::vector<std::string> q_names = {
        "N",     "N 0",     "N +1",     "N -1",
        "Psi",
        "U", "U +1/2", "U -1/2",
        // Staggered variables
        "ST N", "ST N +1/2", "ST N -1/2",
        "ST Psi", "ST ds Psi",
        "ST U",     "ST U 0",     "ST U +1",     "ST U -1",
        // perp derivatives
        "dxF N",     "dxB N",     "dyF N",     "dyB N", "dzF N", "dzB N",
        "dx Psi",  "dy Psi",  "dz Psi",
        "dx U",    "dy U",    "dz U",
        // Staggered perp derivatives
        "ST dx N",   "ST dy N",   "ST dz N",
        "ST dx Psi", "ST dy Psi", "ST dz Psi",
        "ST dxF U",  "ST dxB U",  "ST dyF U",  "ST dyB U", "ST dzF U", "ST dzB U"
    };
    // Only set once every call to operator()
    Container m_apar, m_aparST;
    std::array<Container,3> m_dA, m_dAST;

    // Set by diag_update
    std::array<Container,2> m_dsN, m_dsU, m_dsP;
    std::array<Container,2> m_dssU, m_lapParU, m_lapParN;
    std::array<std::array<Container,3>,2> m_gradN, m_gradU, m_gradP;

    // Helper variables can be overwritten any time (except by compute_parallel)!!
    Container m_temp0, m_temp1;
    Container m_minus, m_zero, m_plus;

    // Helper to compute Dot Apar in diag file
    dg::Extrapolation<Container> m_old_apar;

    const feltor::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    bool m_upToDate = false;
    unsigned m_called = 0;

};

template<class Grid, class IMatrix, class Matrix, class Container>
Explicit<Grid, IMatrix, Matrix, Container>::Explicit( const Grid& g,
    feltor::Parameters p, dg::geo::TokamakMagneticField mag,
    dg::file::WrappedJsonValue js
    ):
    m_solvers( g, p, mag, js),
    m_perp( g, p, mag, js),
    m_para( g, p, mag, js),
    m_sources( g, p, mag, js, m_perp),
    m_old_apar( 2, dg::evaluate( dg::zero, g)),
    m_p(p), m_js(js)
{
    //--------------------------init vectors to 0-----------------//
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_temp1 = m_temp0;
    m_apar = m_aparST = m_temp0;
    m_plus = m_zero = m_minus = m_temp0;

    m_q["N"] = std::array<Container,2>{ m_temp0, m_temp0};
    for( auto name : q_names)
        m_q[name] = m_q["N"];

    m_dsN = m_dsU = m_dsP = m_dssU = m_lapParU = m_lapParN = m_q["N"];

    m_dA[0] = m_dA[1] = m_dA[2] = m_temp0;
    m_dAST = m_dA;
    m_gradP = m_gradU = m_gradN = {m_dA, m_dA};

    //--------------------------Construct-------------------------//

#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    if( m_p.modify_diff)
        DG_RANK0 std::cout << "# Optional parameter \"modify-diff\" activated\n";
    if( m_p.no_diff_penalization)
        DG_RANK0 std::cout << "# Optional parameter \"no-diff-penalization\" activated\n";

}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::initializene(
    const Container& src, Container& target, std::string initphi)
{
    // ne  = Ni
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        if( initphi == "zero")
        {
            m_solvers.invert_gammaN( src, target);
        }
        else if( initphi == "balance")
        {
            //add FLR correction -0.5*tau*mu*Delta n_e
            m_perp.compute_lapMperpN( 0.5*m_p.tau[1]*m_p.mu[1], src, m_temp0, 1.0, target);
            //wird stark negativ falls alpha klein!!
        }
        else if( !(initphi == "zero_pol"))
        {
            throw dg::Error(dg::Message(_ping_)<<"Warning! initphi value '"<<initphi<<"' not recognized. I have tau = "<<m_p.tau[1]<<" ! I don't know what to do! I exit!\n");
        }
    }
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::initializeni(
    const Container& src, Container& target, std::string initphi)
{
    //According to Markus we should actually always invert
    //so we should reconsider this function
    // Ni = ne
    dg::blas1::copy( src, target);
    if (m_p.tau[1] != 0.) {
        if( initphi == "zero")
        {
            //add FLR correction -0.5*tau*mu*Delta n_e
            m_perp.compute_lapMperpN( 0.5*m_p.tau[1]*m_p.mu[1], src, m_temp0, 1.0, target);
            //wird stark negativ falls alpha klein!!
        }
        else if( initphi == "balance")
        {
            //add FLR correction +0.5*tau*mu*Delta n_e
            m_perp.compute_lapMperpN( -0.5*m_p.tau[1]*m_p.mu[1], src, m_temp0, 1.0, target);
            //wird stark negativ falls alpha klein!!
        }
        else if( !(initphi == "zero_pol"))
        {
            throw dg::Error(dg::Message(_ping_)<<"Warning! initphi value '"<<initphi<<"' not recognized. I have tau = "<<m_p.tau[1]<<" ! I don't know what to do! I exit!\n");
        }
    }
}




#ifndef WITH_NAVIER_STOKES
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::operator()(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    std::array<std::array<Container,2>,2>& yp)
{
    m_called++;
    m_upToDate = false;
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    //DG_RANK0 std::cout << "## time "<<time<<" dt "<<dt<<" t_out "<<t_output<<" step "<<step<<" failed "<<var.nfailed<<"\n";
    DG_RANK0 std::cout << "## time "<<t<<"\n";
    /* y[0][0] := n_e
       y[0][1] := N_i
       y[1][0] := w_e^dagger
       y[1][1] := W_i^dagger
    */

    dg::Timer timer;
    double accu = 0.;//accumulated time
    timer.tic();

    dg::blas1::copy( y[0], m_q.at("N")),

#if FELTORPERP == 1

    // set Psi[0]
    m_solvers.compute_phi( t, m_q.at("N"), m_q.at("Psi")[0], m_p.penalize_wall,
        m_sources.get_wall(), m_p.penalize_sheath, m_para.get_sheath());
    // set Psi[1] and m_uE2 --- needs Psi[0]
    m_solvers.compute_psi( t, m_q.at("Psi")[0], m_q.at("Psi")[1]);

#endif

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute phi and psi               took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );

    //Compute ST N and ST Psi
    m_para.update_staggered_density_and_phi( t, m_q);

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute phi and psi ST            took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );

    // Compute m_aparST and m_q.at("ST U") if necessary
    dg::blas1::copy( y[1], m_q.at("ST U"));
    if( m_p.beta != 0)
    {
        m_solvers.compute_aparST( t, m_q.at("ST N"), m_q.at("ST U"), m_aparST, true);
    }
    //Compute m_velocity and m_apar
    m_para.update_velocity_and_apar( t, m_q, m_aparST, m_apar);
    m_old_apar.update( t, m_apar);

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute Apar and staggered        took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );

#if FELTORPERP == 1

    // Set perpendicular dynamics in yp
    m_perp.update_derivatives( m_q, m_apar, m_dA);
    m_perp.compute_density(  t, m_q, m_apar, m_dA, yp[0]);
    m_perp.update_STderivatives( m_q, m_aparST, m_dAST);
    m_perp.compute_velocity( t, m_q, m_aparST, m_dAST, yp[1]);

#else

    dg::blas1::copy( 0., yp);

#endif

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute perp dynamics             took "
                       << timer.diff() << "s\t A: "<<accu<<"s\n";
    timer.tic();

    // Add parallel dynamics
#if FELTORPARALLEL == 1

    m_para.compute_parallel( m_q, yp);
    m_para.add_densities_diffusion( m_q, yp);
    m_para.add_velocities_diffusion( m_q, yp);

#endif
#if FELTORPERP == 1
    //------------------Add Resistivity--------------------------//
    double eta = m_p.eta, mu0 = m_p.mu[0], mu1 = m_p.mu[1];
    dg::blas1::subroutine( [eta,mu0,mu1] DG_DEVICE (
            double ne, double ni,
            double ue, double ui, double& dtUe, double& dtUi){
                double current = ni*ui-ne*ue;
                dtUe += -eta/mu0 * current;
                dtUi += -eta/mu1 * ne/ni * current;
            },
        m_q.at("ST N")[0], m_q.at("ST N")[1],
        m_q.at("ST U")[0], m_q.at("ST U")[1], yp[1][0], yp[1][1]);
#endif

    if( m_p.no_diff_penalization)
    {
        for( unsigned i=0; i<2; i++)
        for( unsigned j=0; j<2; j++)
        {
            common::multiply_rhs_penalization( yp[i][j], m_p.penalize_wall, m_sources.get_wall(),
                    m_p.penalize_sheath, m_para.get_sheath()); // F*(1-chi_w-chi_s)
        }
    }
#if FELTORPERP == 1
    for( unsigned i=0; i<2; i++)
    {
        m_perp.compute_diffusiveN( 1., m_q.at("N")[i], m_temp0,
                m_temp1, 1., yp[0][i]);
        m_perp.compute_diffusiveU( 1., m_q.at("ST U")[i], m_q.at("ST N")[i], m_temp0,
                m_temp1, m_zero, m_plus, 1., yp[1][i]);
    }
#endif
    if( !m_p.no_diff_penalization)
    {
        for( unsigned i=0; i<2; i++)
        for( unsigned j=0; j<2; j++)
        {
            common::multiply_rhs_penalization( yp[i][j], m_p.penalize_wall, m_sources.get_wall(),
                    m_p.penalize_sheath, m_para.get_sheath()); // F*(1-chi_w-chi_s)
        }
    }

    m_para.add_sheath_terms( m_q, yp);
    m_sources.add_wall_terms( m_q, yp);
    //Add source terms
    m_sources.add_source_terms( m_q, m_perp, m_para, yp );

    timer.toc();
    accu += timer.diff();
    #ifdef MPI_VERSION
        if(rank==0)
    #endif
    std::cout << "## Add parallel dynamics and sources took "<<timer.diff()
              << "s\t A: "<<accu<<"\n";
}


#else // WITH_NAVIER_STOKES
#include "../navier_stokes/navier_stokes.h"
#endif // WITH_NAVIER_STOKES

} //namespace feltor
