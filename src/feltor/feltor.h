#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "parameters.h"
#include "common.h"
#include "solvers.h"
#include "perpendicular.h"
#include "parallel.h"

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
    void add_implicit( double t,
        const std::map<std::string, std::array<Container,2>>& q,
        double beta,
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
        return m_s[0][i];
    }
    const Container& velocity(int i)const{
        return m_q.at("U")[i];
    }
    const Container& velocity_source(int i){
        update_diag();
        return m_s[1][i];
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
    void compute_gradSN( int i, std::array<Container,3>& gradS) const{
        // MW: don't like this function, if we need more gradients we might
        // want a more flexible solution
        // grad S_ne and grad S_ni
        m_perp.compute_gradN( m_s[0][i], gradS);
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
        m_lapperpP.set_chi( 1.);
        dg::blas2::gemv( m_lapperpP, m_q.at("Psi")[i], result);
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
        return m_source;
    }
    const Container& get_source_prof() const{
        return m_profne;
    }
    const Container& get_wall() const{
        return m_wall;
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
        // we don't want jumps in phi in here so we use lapperpP
        dg::blas1::pointwiseDot( m_p.mu[1], density, m_perp.binv(), m_perp.binv(), 0., temp);
        m_lapperpP.set_chi( temp);
        dg::blas2::symv( -alpha, m_lapperpP, m_q.at("Psi")[0], beta, result);
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
                // velocity source
                dg::blas1::evaluate( m_s[1][i], dg::equals(), []DG_DEVICE(
                            double sn, double u, double n){ return -u*sn/n;},
                        m_s[0][i], m_q.at("U")[i], m_q.at("N")[i]);
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
    void set_source( bool fixed_profile, Container profile, double source_rate, Container source, double minne, double minrate, double minalpha)
    {
        m_fixed_profile = fixed_profile;
        m_profne = profile;
        m_source_rate = source_rate;
        m_source = source;
        m_minne = minne;
        m_minrate = minrate;
        m_minalpha = minalpha;
    }
    void set_wall(double wall_rate, const Container& wall, double nwall, double uwall)
    {
        m_wall_rate = wall_rate;
        dg::blas1::copy( wall, m_wall);
        m_nwall = nwall;
        m_uwall = uwall;
    }
    void set_sheath(double sheath_rate, const Container& sheath,
            const Container& sheath_coordinate)
    {
        m_para.set_sheath( sheath_rate, sheath, sheath_coordinate);
    }
    void add_source_terms(  const std::map<std::string, std::array<Container,2>>&,
                            std::array<std::array<Container,2>,2>& yp);
    const dg::geo::Fieldaligned<Geometry, IMatrix, Container>& fieldaligned() const
    {
        return m_para.fieldaligned();
    }
  private:
    Solvers<Geometry, Matrix, Container> m_solvers;
    PerpDynamics<Geometry, Matrix, Container> m_perp;
    ParaDynamics<Geometry, IMatrix, Matrix, Container> m_para;

    Container m_source, m_profne;
    Container m_wall;

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

    // overwritten by diag_update and/or set once by operator()
    std::array<Container,3> m_dA, m_dAST;
    std::array<Container,2> m_dsN, m_dsU, m_dsP;
    std::array<std::array<Container,2>,2> m_s;

    // Set by diag_update
    std::array<Container,2> m_dssU, m_lapParU, m_lapParN;
    std::array<std::array<Container,3>,2> m_gradN, m_gradU, m_gradP;

    // Helper variables can be overwritten any time (except by compute_parallel)!!
    Container m_temp0, m_temp1;
    Container m_minus, m_zero, m_plus;

    //matrices and solvers

    dg::Elliptic3d< Geometry, Matrix, Container> m_lapperpP;

    dg::Extrapolation<Container> m_old_apar;

    const feltor::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    double m_source_rate = 0., m_wall_rate = 0.;
    double m_minne = 0., m_minrate  = 0., m_minalpha = 0.;
    double m_nwall = 0., m_uwall = 0.;
    bool m_fixed_profile = true, m_compute_in_3d = true;
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
    m_old_apar( 2, dg::evaluate( dg::zero, g)),
    m_p(p), m_js(js)
{
    //--------------------------init vectors to 0-----------------//
    dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
    m_source = m_temp1 = m_temp0;
    m_apar = m_aparST = m_profne = m_wall = m_temp0;
    m_plus = m_zero = m_minus = m_temp0;

    m_q["N"] = std::array<Container,2>{ m_temp0, m_temp0};
    for( auto name : q_names)
        m_q[name] = m_q["N"];

    m_dsN = m_dsU = m_dsP = m_dssU = m_lapParU = m_lapParN = m_q["N"];

    m_dA[0] = m_dA[1] = m_dA[2] = m_temp0;
    m_dAST = m_dA;
    m_gradP = m_gradU = m_gradN = {m_dA, m_dA};
    m_s[0] = m_s[1] = m_dsN ;

    //--------------------------Construct-------------------------//

    m_lapperpP.construct ( g, p.bcxP, p.bcyP, dg::PER,  p.pol_dir),
    m_lapperpP.set_chi( m_perp.projection());
    if( (p.curvmode == "true") && (p.symmetric == false))
        m_compute_in_3d = true;
    else
    {
        m_compute_in_3d = false;
        m_lapperpP.set_compute_in_2d(true);
    }
    m_lapperpP.set_jfactor(0); //we don't want jump terms in source
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



template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_source_terms(
    const std::map<std::string, std::array<Container,2>>& q,
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
                m_s[0][0], q.at("N")[0], m_profne, m_source, m_source_rate);
        else
            dg::blas1::axpby( m_source_rate, m_source, 0., m_s[0][0]);
    }
    else
        dg::blas1::copy( 0., m_s[0][0]);
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
    m_perp.compute_lapMperpU( m_s[0][0], m_temp0); // lapU avoids subtraction of nbc
    dg::blas1::axpby( 1., m_s[0][0], 0.5*m_p.tau[1]*m_p.mu[1], m_temp0, m_s[0][1]);
    // potential part of FLR correction S_N += -div*(mu S_n grad*Phi/B^2)
    dg::blas1::pointwiseDot( m_p.mu[1], m_s[0][0], m_perp.binv(), m_perp.binv(), 0., m_temp0);
    m_lapperpP.set_chi( m_temp0);
    m_lapperpP.symv( 1., m_q.at("Psi")[0], 1., m_s[0][1]);

    // S_U = - U S_N/N
    for(int i=0; i<2; i++)
    {
        // transform to adjoint plane and add to velocity source
        m_para.fieldalignedHalf()( dg::geo::zeroMinus, m_s[0][i], m_minus);
        m_para.fieldalignedHalf()( dg::geo::einsPlus,  m_s[0][i], m_plus);
        m_para.update_parallel_bc_1st( m_minus, m_plus, m_p.bcxN, 0.);
        dg::geo::ds_average( m_para.fieldalignedHalf(), 1., m_minus, m_plus, 0., m_temp0);
        dg::blas1::evaluate( m_s[1][i], dg::equals(), []DG_DEVICE(
                    double sn, double u, double n){ return -u*sn/n;},
                m_temp0, m_q.at("ST U")[i], m_q.at("ST N")[i]);
    }
    //Add all to the right hand side
    dg::blas1::axpby( 1., m_s, 1.0, yp);
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
        m_wall, m_p.penalize_sheath, m_para.get_sheath());
    // set m_potential[1] and m_uE2 --- needs m_potential[0]
    m_solvers.compute_psi( t, m_q.at("Psi")[0], m_q.at("Psi")[1]);

#else


#endif

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Compute phi and psi               took "
                       << timer.diff()<<"s\t A: "<<accu<<"s\n";
    timer.tic( );

    //Compute ST N and ST Psi
    m_para.update_staggered_density_and_phi( t, m_q);

    //// Now refine potential on staggered grid
    //// set m_potentialST[0]
    //compute_phi( t, m_q.at("ST N"), m_potentialST[0], true);
    //// set m_potentialST[1]  --- needs m_potentialST[0]
    //compute_psi( t, m_potentialST[0], m_potentialST[1], true);
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

    if( !m_p.partitioned)
    {
        // explicit and implicit timestepper
        add_implicit( t, m_q, 1., yp);
    }
    else
    {
        // partitioned means imex timestepper
        for( unsigned i=0; i<2; i++)
        {
            for( unsigned j=0; j<2; j++)
                common::multiply_rhs_penalization( yp[i][j], m_p.penalize_wall, m_wall,
                    m_p.penalize_sheath, m_para.get_sheath()); // F*(1-chi_w-chi_s)
        }
    }

    m_para.add_sheath_terms( m_q, m_nwall, m_uwall, yp);
    // add wall boundary conditions
    if( m_wall_rate != 0)
    {
        for( unsigned i=0; i<2; i++)
        {
            dg::blas1::axpby( +m_wall_rate*m_nwall, m_wall, 1., yp[0][i] );
            dg::blas1::axpby( +m_wall_rate*m_uwall, m_wall, 1., yp[1][i] );
        }
    }
    //Add source terms
    // set m_s
    add_source_terms( m_q, yp );

    timer.toc();
    accu += timer.diff();
    #ifdef MPI_VERSION
        if(rank==0)
    #endif
    std::cout << "## Add parallel dynamics and sources took "<<timer.diff()
              << "s\t A: "<<accu<<"\n";
}
template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::add_implicit(
    double,
    const std::map<std::string, std::array<Container,2>>& q,
    double beta,
    std::array<std::array<Container,2>,2>& yp)
{
    dg::blas1::scal( yp, beta);
#if FELTORPARALLEL == 1
    m_para.add_densities_diffusion( m_q, yp);
    m_para.add_velocities_diffusion( m_q, yp);
#endif
    if( m_p.no_diff_penalization)
    {
        for( unsigned i=0; i<2; i++)
        {
            common::multiply_rhs_penalization( yp[0][i], m_p.penalize_wall, m_wall,
                    m_p.penalize_sheath, m_para.get_sheath()); // F*(1-chi_w-chi_s)
            common::multiply_rhs_penalization( yp[1][i], m_p.penalize_wall, m_wall,
                    m_p.penalize_sheath, m_para.get_sheath()); // F*(1-chi_w-chi_s)
            dg::blas1::pointwiseDot( -m_wall_rate, m_wall, q.at("N")[i],
                -m_para.get_sheath_rate(), m_para.get_sheath(), q.at("N")[i], 1., yp[0][i]); // -r N
            dg::blas1::pointwiseDot( -m_wall_rate, m_wall, q.at("ST U")[i],
                -m_para.get_sheath_rate(), m_para.get_sheath(), q.at("ST U")[i], 1., yp[1][i]); // -r U
        }
    }
#if FELTORPERP == 1
    for( unsigned i=0; i<2; i++)
    {
        m_perp.compute_diffusiveN( 1., q.at("N")[i], m_temp0,
                m_temp1, 1., yp[0][i]);
        m_perp.compute_diffusiveU( 1., q.at("ST U")[i], q.at("ST N")[i], m_temp0,
                m_temp1, m_zero, m_plus, 1., yp[1][i]);
    }
#endif
    if( !m_p.no_diff_penalization)
    {
        for( unsigned i=0; i<2; i++)
        {
            common::multiply_rhs_penalization( yp[0][i], m_p.penalize_wall, m_wall,
                    m_p.penalize_sheath, m_para.get_sheath()); // F*(1-chi_w-chi_s)
            common::multiply_rhs_penalization( yp[1][i], m_p.penalize_wall, m_wall,
                    m_p.penalize_sheath, m_para.get_sheath()); // F*(1-chi_w-chi_s)
            dg::blas1::pointwiseDot( -m_wall_rate, m_wall, q.at("N")[i],
                -m_para.get_sheath_rate(), m_para.get_sheath(), q.at("N")[i], 1., yp[0][i]); // -r N
            dg::blas1::pointwiseDot( -m_wall_rate, m_wall, q.at("ST U")[i],
                -m_para.get_sheath_rate(), m_para.get_sheath(), q.at("ST U")[i], 1., yp[1][i]); // -r U
        }
    }
}


#else // WITH_NAVIER_STOKES
#include "../navier_stokes/navier_stokes.h"
#endif // WITH_NAVIER_STOKES

} //namespace feltor
