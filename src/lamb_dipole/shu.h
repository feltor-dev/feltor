#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <exception>

#include <nlohmann/json.hpp>
#include "dg/file/json_utilities.h"
#include "dg/algorithm.h"
#include "init.h"

namespace shu
{

template< class IMatrix, class Container>
struct Filter
{
    using value_type = dg::get_value_type<Container>;
    Filter() = default;
    template<class Geometry>
    Filter( const Geometry& grid, dg::file::WrappedJsonValue& js) {
        m_type = js[ "regularization"].get( "type", "none").asString();
        if( m_type == "modal")
        {
            double alpha = js[ "regularization"].get( "alpha", 36).asDouble();
            double order = js[ "regularization"].get( "order", 8).asDouble();
            double eta_c = js[ "regularization"].get( "eta_c", 0.5).asDouble();
            auto op = dg::ExponentialFilter(alpha, eta_c, order, grid.n());
            m_filter = dg::create::fast_transform(
                    dg::create::modal_filter(op, grid.nx()),
                    dg::create::modal_filter(op, grid.ny()),
                    grid);
        }
        else if( m_type == "swm")
        {
            m_alpha0 = js["regularization"].get("alpha", 20).asDouble();
            m_iter = js["regularization"].get("iter", 4).asDouble();
        }
        else if( m_type == "dg-limiter")
        {
            m_alpha0 = js["regularization"].get("alpha", 10).asDouble();
        }
        if( m_type == "swm" || m_type == "median")
        {
            m_stencil = dg::create::window_stencil( {3,3}, grid, grid.bcx(), grid.bcy());
        }
        else if( m_type == "dg-limiter")
        {
            m_stencil = dg::create::limiter_stencil( dg::coo3d::x, grid, grid.bcx());
            m_stencilY = dg::create::limiter_stencil( dg::coo3d::y, grid, grid.bcy());
        }
        m_tmp = dg::evaluate( dg::zero, grid);
    }

    void operator()( Container& y){
        if( m_type == "none" || m_type == "viscosity") return;
        dg::Timer t;
        t.tic();
        if( m_type == "modal")
        {
            dg::blas2::symv( m_filter, y, m_tmp);
            using std::swap;
            swap( m_tmp, y);
        }
        else if( m_type == "median")
        {
            dg::blas2::stencil( dg::CSRMedianFilter(), m_stencil, y, m_tmp);
            using std::swap;
            swap( m_tmp, y);
        }
        else if( m_type == "swm")
        {
            value_type alpha = m_alpha0;
            for( unsigned i=0; i<m_iter; i++)
            {
                dg::blas2::stencil( dg::CSRSWMFilter<value_type>(alpha), m_stencil, y, m_tmp);
                using std::swap;
                swap( m_tmp, y);
                alpha*=0.8;
            }
        }
        else if ( m_type == "dg-limiter")
        {
            dg::blas2::stencil( dg::CSRSlopeLimiter<value_type>(m_alpha0), m_stencil, y, m_tmp);
            dg::blas2::stencil( dg::CSRSlopeLimiter<value_type>(m_alpha0), m_stencilY, m_tmp, y);
        }
        t.toc();
        std::cout << "Application of filter took "<<t.diff()<<"s\n";

        return;
    }
    private:
    std::string m_type;
    dg::MultiMatrix<dg::DMatrix, dg::DVec> m_filter;
    unsigned m_iter;
    value_type m_alpha0;
    IMatrix m_stencil, m_stencilY;
    Container m_tmp;
};

// Improvement: make Diffusion a friend to Shu to save memory and duplicate code
template< class Geometry, class Matrix, class Container>
struct Diffusion;

template< class Geometry, class Matrix, class Container >
struct Shu
{
    friend struct Diffusion<Geometry, Matrix, Container>;
    using value_type = dg::get_value_type<Container>;

    Shu( const Geometry& grid, dg::file::WrappedJsonValue& js);

    const dg::Elliptic<Matrix, Container, Container>& lap() const { return m_multi_laplaceM[0];}
    dg::ArakawaX<Geometry, Matrix, Container>& arakawa() {return m_arakawa;}

    const Container& potential( ) {return m_psi;}
    Matrix& dx() {return m_centered[0];}
    Matrix& dy() {return m_centered[1];}

    void rhs(double t, const Container& y, const Container& psi, Container& yp);
    void operator()(double t, const Container& y, Container& yp);
    void compute_omega( const Container& phi, Container& omega)
    {
        dg::blas2::symv( m_LaplacianM, phi, omega);
    }

    void add_implicit(double alpha, const Container& x, double beta, Container& yp);

    void set_mms_source( double sigma, double velocity, double ly) {
        m_mms = shu::MMSSource( sigma, velocity, ly);
        m_add_mms = true;
    }
    const Container& weights(){
        return m_LaplacianM.weights();
    }
  private:
    Container m_psi, m_v, m_temp[3], m_fine_psi, m_fine_v, m_fine_temp[3], m_fine_y, m_fine_yp;
    std::vector<dg::Elliptic<Geometry, Matrix, Container>> m_multi_laplaceM;
    dg::Elliptic<Geometry, Matrix,Container> m_LaplacianM;
    dg::ArakawaX<Geometry, Matrix, Container> m_arakawa;
    dg::Advection<Geometry, Matrix, Container> m_adv;
    dg::Extrapolation<Container> m_old_psi;
    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    dg::MultiMatrix<Matrix,Container> m_inter, m_project;
    Matrix m_forward[2], m_backward[2], m_centered[2];
    Matrix m_fine_forward[2], m_fine_backward[2], m_fine_centered[2];
    Matrix m_centered_phi[2]; // for variation
    std::vector<double> m_eps;
    std::string m_advection, m_multiplication;
    dg::SparseTensor<Container> m_metric;

    shu::MMSSource m_mms;
    bool m_add_mms = false;
    Container m_x, m_y;
    double m_nu = 0.; // for Diffusion
    unsigned m_order = 1; // for Diffusion
    bool m_partitioned = false, m_update = true;
};

template<class Geometry, class Matrix, class Container>
Shu< Geometry, Matrix, Container>::Shu(
        const Geometry& g, dg::file::WrappedJsonValue& js):
    m_old_psi( 2, dg::evaluate( dg::zero, g)),
    m_multigrid( g, 3),
    m_x( dg::evaluate( dg::cooX2d, g)),
    m_y( dg::evaluate( dg::cooY2d, g))
{
    m_advection = js[ "advection"].get( "type", "arakawa").asString();
    m_multiplication = js[ "advection"].get( "multiplication", "pointwise").asString();
    std::string stepper = js[ "timestepper"].get( "type",
            "FilteredExplicitMultistep").asString();
    if( stepper == "ImExMultistep" || stepper == "ARK")
        m_partitioned = true;
    if( stepper == "ImplicitMultistep" || stepper == "DIRK")
        m_update = false;
    m_metric = g.metric();

    m_psi = dg::evaluate( dg::zero, g);
    m_centered_phi[0] = dg::create::dx( g, g.bcx(), dg::centered);
    m_centered_phi[1] = dg::create::dy( g, g.bcy(), dg::centered);
    m_v = dg::evaluate( dg::zero, g);
    m_temp[0] = dg::evaluate( dg::zero, g);
    m_temp[1] = dg::evaluate( dg::zero, g);
    m_temp[2] = dg::evaluate( dg::zero, g);
    if( "projection" == m_multiplication)
    {
        Geometry fine_grid = g;
        fine_grid.set( 2*g.n(), g.Nx(), g.Ny());
        //theoretically we only need 2n-1 but it isn't wrong to take more
        m_inter = dg::create::fast_interpolation( g, 2, 1, 1);
        m_project = dg::create::fast_projection( fine_grid, 2, 1, 1);

        m_fine_centered[0] = dg::create::dx( fine_grid, g.bcx(), dg::centered);
        m_fine_centered[1] = dg::create::dy( fine_grid, g.bcy(), dg::centered);
        m_fine_forward[0] = dg::create::dx( fine_grid, dg::inverse( g.bcx()), dg::forward);
        m_fine_forward[1] = dg::create::dy( fine_grid, dg::inverse( g.bcy()), dg::forward);
        m_fine_backward[0] = dg::create::dx( fine_grid, dg::inverse( g.bcx()), dg::backward);
        m_fine_backward[1] = dg::create::dy( fine_grid, dg::inverse( g.bcy()), dg::backward);
        m_fine_psi = dg::evaluate( dg::zero, fine_grid);
        m_fine_y = dg::evaluate( dg::zero, fine_grid);
        m_fine_yp = dg::evaluate( dg::zero, fine_grid);
        m_fine_v = dg::evaluate( dg::zero, fine_grid);
        m_fine_temp[0] = dg::evaluate( dg::zero, fine_grid);
        m_fine_temp[1] = dg::evaluate( dg::zero, fine_grid);
        m_fine_temp[2] = dg::evaluate( dg::zero, fine_grid);
        m_arakawa.construct( fine_grid);
        m_adv.construct( fine_grid);
    }
    else
    {
        m_arakawa.construct( g);
        m_adv.construct(g);
    }
    m_centered[0] = dg::create::dx( g, g.bcx(), dg::centered);
    m_centered[1] = dg::create::dy( g, g.bcy(), dg::centered);
    m_forward[0] = dg::create::dx( g, dg::inverse( g.bcx()), dg::forward);
    m_forward[1] = dg::create::dy( g, dg::inverse( g.bcy()), dg::forward);
    m_backward[0] = dg::create::dx( g, dg::inverse( g.bcx()), dg::backward);
    m_backward[1] = dg::create::dy( g, dg::inverse( g.bcy()), dg::backward);

    unsigned stages = js[ "elliptic"].get( "stages", 3).asUInt();
    m_eps.resize(stages);
    m_eps[0] = js[ "elliptic"][ "eps_pol"].get( 0, 1e-6).asDouble();
    for( unsigned i=1;i<stages; i++)
    {
        m_eps[i] = js[ "elliptic"][ "eps_pol"].get( i, 1).asDouble();
        m_eps[i]*= m_eps[0];
    }
    //this is a hidden parameter
    //note that only centered works with double periodic boundary conditions
    enum dg::direction dir = dg::str2direction(
            js[ "elliptic"].get( "direction", "centered").asString());
    m_multi_laplaceM.resize(stages);
    for( unsigned u=0; u<stages; u++)
        m_multi_laplaceM[u].construct( m_multigrid.grid(u),  dir, 1);
    // explicit Diffusion term
    std::string regularization = js[ "regularization"].get( "type", "modal").asString();
    if( "viscosity" == regularization )
    {
        m_nu = js[ "regularization"].get( "nu", 1e-3).asDouble();
        m_order = js[ "regularization"].get( "order", 1).asUInt();
        enum dg::direction dir = dg::str2direction(
                js["regularization"].get( "direction", "centered").asString());
        m_LaplacianM.construct( g,  dir, 1);
    }
}

template< class Geometry, class Matrix, class Container>
void Shu<Geometry, Matrix, Container>::operator()(double t, const Container& y, Container& yp)
{
    //solve elliptic equation
    //if( m_update)
        m_old_psi.extrapolate( t, m_psi);
    //else
    //    dg::blas1::copy( 0., m_psi);
    std::vector<unsigned> number = m_multigrid.solve( m_multi_laplaceM, m_psi, y, m_eps);
    //if( m_update)
        m_old_psi.update( t, m_psi);
    rhs( t, y, m_psi, yp);
}
template< class Geometry, class Matrix, class Container>
void Shu<Geometry, Matrix, Container>::rhs(double t, const Container& y, const Container& phi, Container& yp)
{
    //now do advection with various schemes
    if( "pointwise" == m_multiplication)
    {
        if( "arakawa" == m_advection)
            m_arakawa( y, phi, yp); //A(y,psi)-> yp
        else if( "upwind" == m_advection)
        {
            dg::blas1::copy( 0., yp);
            // - dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], phi, 0., m_v); //v_x
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_x
            dg::blas2::symv( m_forward[0], m_temp[0], m_temp[1]);
            dg::blas2::symv( m_backward[0], m_temp[0], m_temp[2]);
            dg::blas1::evaluate( yp, dg::minus_equals(), dg::Upwind(), m_v, m_temp[2], m_temp[1]);
            // - dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], phi, 0., m_v); //v_y
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_y
            dg::blas2::symv( m_forward[1], m_temp[0], m_temp[1]);
            dg::blas2::symv( m_backward[1], m_temp[0], m_temp[2]);
            dg::blas1::evaluate( yp, dg::minus_equals(), dg::Upwind(), m_v, m_temp[2], m_temp[1]);
        }
        else if( "centered" == m_advection)
        {
            // - dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], phi, 0., m_v); //v_x
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_x
            dg::blas2::symv( -1., m_centered[0], m_temp[0], 0., yp);
            // - dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], phi, 0., m_v); //v_y
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_y
            dg::blas2::symv( -1., m_centered[1], m_temp[0], 1., yp);
        }
        else if( "upwind-advection" == m_advection)
        {
            //  - v_x dx n
            dg::blas2::symv( -1., m_centered[1], phi, 0., m_temp[0]); //v_x

            //  - v_y dy n
            dg::blas2::symv( 1., m_centered[0], phi, 0., m_temp[1]); //v_y
            m_adv.upwind( -1., m_temp[0], m_temp[1], y, 0., yp);
        }
        else if( "centered-advection" == m_advection)
        {
            //dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], phi, 0., m_v); //v_x
            dg::blas2::symv( -1., m_centered[0], y, 0., m_temp[0]); // -v_x dx omega
            dg::blas1::pointwiseDot( m_v, m_temp[0], yp);
            //dy ( nv_y)
            dg::blas2::symv(  1., m_centered[0], phi, 0., m_v); //v_y
            dg::blas2::symv( -1., m_centered[1], y, 0., m_temp[0]);
            dg::blas1::pointwiseDot( 1., m_v, m_temp[0], 1., yp); //f_y
        }
    }
    else // "projection " == multiplication
    {
        if( "arakawa" == m_advection)
        {
            dg::blas2::symv( m_inter, y, m_fine_y);
            dg::blas2::symv( m_inter, phi, m_fine_psi);
            m_arakawa( m_fine_y, m_fine_psi, m_fine_yp); //A(y,psi)-> yp
        }
        else if( "upwind" == m_advection)
        {
            dg::blas2::symv( m_inter, y, m_fine_y);
            dg::blas2::symv( m_inter, phi, m_fine_psi);
            dg::blas1::copy( 0., m_fine_yp);
            //dx ( nv_x)
            dg::blas2::symv( -1., m_fine_centered[1], m_fine_psi, 0., m_fine_v); //v_x
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_x
            dg::blas2::symv( m_fine_forward[0], m_fine_temp[0], m_fine_temp[1]);
            dg::blas2::symv( m_fine_backward[0], m_fine_temp[0], m_fine_temp[2]);
            dg::blas1::evaluate( m_fine_yp, dg::minus_equals(), dg::Upwind(), m_fine_v, m_fine_temp[2], m_fine_temp[1]);
            //dy ( nv_y)
            dg::blas2::symv( 1., m_fine_centered[0], m_fine_psi, 0., m_fine_v); //v_y
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_y
            dg::blas2::symv( m_fine_forward[1], m_fine_temp[0], m_fine_temp[1]);
            dg::blas2::symv( m_fine_backward[1], m_fine_temp[0], m_fine_temp[2]);
            dg::blas1::evaluate( m_fine_yp, dg::minus_equals(), dg::Upwind(), m_fine_v, m_fine_temp[2], m_fine_temp[1]);
        }
        else if( "centered" == m_advection)
        {
            dg::blas2::symv( m_inter, y, m_fine_y);
            dg::blas2::symv( m_inter, phi, m_fine_psi);
            //dx ( nv_x)
            dg::blas2::symv( -1., m_fine_centered[1], m_fine_psi, 0., m_fine_v); //v_x
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_x
            dg::blas2::symv( -1., m_fine_centered[0], m_fine_temp[0], 0., m_fine_yp);
            //dy ( nv_y)
            dg::blas2::symv( 1., m_fine_centered[0], m_fine_psi, 0., m_fine_v); //v_y
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_y
            dg::blas2::symv( -1., m_fine_centered[1], m_fine_temp[0], 1., m_fine_yp);
        }
        else if( "upwind-advection" == m_advection)
        {
            // v_x dx n
            dg::blas1::copy( 0., m_fine_yp);
            dg::blas2::symv( -1., m_centered[1], phi, 0., m_v); //v_x
            dg::blas2::symv( m_forward[0], y, m_temp[1]);
            dg::blas2::symv( m_backward[0], y, m_temp[2]);
            dg::blas2::symv( m_inter, m_v, m_fine_v);
            dg::blas2::symv( m_inter, m_temp[1], m_fine_temp[1]);
            dg::blas2::symv( m_inter, m_temp[2], m_fine_temp[2]);
            dg::blas1::evaluate( m_fine_yp, dg::minus_equals(), dg::UpwindProduct(), m_fine_v, m_fine_temp[2], m_fine_temp[1]);
            // v_y dy n
            dg::blas2::symv( 1., m_centered[0], phi, 0., m_v); //v_y
            dg::blas2::symv( m_forward[1], y, m_temp[1]);
            dg::blas2::symv( m_backward[1], y, m_temp[2]);
            dg::blas2::symv( m_inter, m_temp[1], m_fine_temp[1]);
            dg::blas2::symv( m_inter, m_temp[2], m_fine_temp[2]);
            dg::blas2::symv( m_inter, m_v, m_fine_v);
            dg::blas1::evaluate( m_fine_yp, dg::minus_equals(), dg::UpwindProduct(), m_fine_v, m_fine_temp[2], m_fine_temp[1]);
        }
        else if( "centered-advection" == m_advection)
        {
            // v_x dx n
            dg::blas2::symv( -1., m_centered[1], phi, 0., m_v); //v_x
            dg::blas2::symv( -1., m_centered[0], y, 0., m_temp[0]);
            dg::blas2::symv( m_inter, m_temp[0], m_fine_temp[0]);
            dg::blas2::symv( m_inter, m_v, m_fine_v);
            dg::blas1::pointwiseDot( m_fine_v, m_fine_temp[0], m_fine_yp);
            // v_y dy n
            dg::blas2::symv(  1., m_centered[0], phi, 0., m_v); //v_y
            dg::blas2::symv( -1., m_centered[1], y, 0., m_temp[0]);
            dg::blas2::symv( m_inter, m_temp[0], m_fine_temp[0]);
            dg::blas2::symv( m_inter, m_v, m_fine_v);
            dg::blas1::pointwiseDot( 1., m_fine_v, m_fine_temp[0], 1., m_fine_yp);
        }
        dg::blas2::symv( m_project, m_fine_yp, yp);
    }
    if( m_add_mms) //for the manufactured solution we need to add a source term
        dg::blas1::evaluate( yp, dg::plus_equals(), m_mms, m_x, m_y, t);

    if( !m_partitioned)
        add_implicit( 1., y, 1., yp);
}

template< class Geometry, class Matrix, class Container>
void Shu<Geometry,Matrix,Container>::add_implicit( double alpha, const Container& x, double beta, Container& yp)
{
    if( m_nu != 0)
    {
        dg::blas1::copy( x, m_temp[1]);
        for( unsigned p=0; p<m_order; p++)
        {
            using std::swap;
            swap( m_temp[0], m_temp[1]);
            dg::blas2::symv( m_nu, m_LaplacianM, m_temp[0], 0., m_temp[1]);
        }
        dg::blas1::axpby( -alpha, m_temp[1], beta, yp);
    }
    else
        dg::blas1::scal( yp, beta);
}

template< class Geometry, class Matrix, class Container>
struct Diffusion
{
    Diffusion( Shu<Geometry,Matrix,Container>& shu): m_shu(&shu){
        //m_weights = shu.m_temp[0];
        //dg::blas1::copy( 1., m_weights);
    }
    void operator()(double, const Container& x, Container& y)
    {
        m_shu->add_implicit( 1., x, 0., y);
    }
    const Container& weights(){ return m_shu->m_LaplacianM.weights();}
    const Container& precond(){ return m_shu->m_LaplacianM.precond();}
  private:
    Shu<Geometry,Matrix,Container>* m_shu;
};

template<class Geometry, class Matrix, class Container>
struct Implicit
{
    Implicit( const Geometry& grid, dg::file::WrappedJsonValue& js)
    {
        std::string type = js["timestepper"]["type"].asString();
        if( type == "ImplicitMultistep" || type == "DIRK")
        {
            m_system = js["timestepper"]["solver"]["system"].asString();
            if( m_system != "single" && m_system != "split" && m_system != "omega")
                throw dg::Error( dg::Message(_ping_) << "Solver system "<<m_system<<" not recognized!\n");
            unsigned stages = js["timestepper"]["solver"]["stages"].asUInt();
            m_nested = { grid, stages}, m_nested_split = { grid, stages};
            m_phi = m_nested.x(0);
            m_omS_split = m_nested_split.x(0), m_phi_split =  m_omS_split;
            unsigned restart = js["timestepper"]["solver"]["restart"].asUInt();
            double damping = js["timestepper"]["solver"]["damping"].asDouble();
            m_eps.resize(stages);
            m_eps[0] = js[ "timestepper"][ "solver"]["eps_time"].get( 0, 1e-6).asDouble();
            for( unsigned i=1;i<stages; i++)
            {
                m_eps[i] = js[ "timestepper"][ "solver"]["eps_time"].get( i, 1).asDouble();
                m_eps[i]*= m_eps[0];
            }
            m_imp.resize(stages), m_inv_imp.resize(stages),
                m_imp_split.resize(stages), m_inv_imp_split.resize(stages);
            for ( unsigned u=0; u<stages; u++)
            {
                // construct Equations
                m_weights.push_back( dg::create::weights(m_nested.grid(u) ));
                m_eqs.push_back( {m_nested.grid(u), js});
                if( m_system == "single")
                {
                    m_imp[u] = [&, u, m_omega = m_weights[u]]
                        ( const auto& phi, auto& f) mutable
                    {
                        // omega - a I(omega, phi)
                        m_eqs[u].compute_omega( phi, m_omega );
                        m_eqs[u].rhs( m_time, m_omega, phi, f);
                        dg::blas1::axpby( 1., m_omega, -m_alpha, f);
                    };
                    m_inv_imp[u] = [this, u, damping, restart,
                        acc = dg::AndersonAcceleration<Container>(m_weights[u], restart)]
                        ( const auto& omS, auto& phi) mutable
                    {
                        // Solve Implicit( phi) = omS
                        dg::Timer t;
                        t.tic();
                        unsigned number = acc.solve( m_imp[u], phi, omS,
                            m_weights[u], m_eps[u], m_eps[u], 10000, damping, restart, false);
                        t.toc();
                        std::cout << "# Implicit stage "<<u<<" solve took "<<number<<" iterations in "<<t.diff()<<"s\n";
                    };
                }
                if( m_system == "omega")
                {
                    m_imp[u] = [&, u, phi = m_weights[u], pcg = dg::PCG<Container>( m_weights[u], 10000)]
                        ( const auto& omega, auto& f) mutable
                    {
                        auto Laplace = [eqs = m_eqs[u]]( const auto& phi, auto& omega) mutable
                        {
                            eqs.compute_omega( phi, omega);
                        };
                        dg::blas1::copy( 0, phi); // ??
                        unsigned number = pcg.solve( Laplace, phi, omega, 1., m_weights[u], m_eps[u]);
                        std::cout << "# Number of PCG steps "<<number<<"\n";
                        // omega - a I(omega, phi)
                        m_eqs[u].rhs( m_time, omega, phi, f);
                        dg::blas1::axpby( 1., omega, -m_alpha, f);
                    };
                    m_inv_imp[u] = [this, u, damping, restart,
                        acc = dg::AndersonAcceleration<Container>(m_weights[u], restart)]
                        ( const auto& omS, auto& omega) mutable
                    {
                        // Solve Implicit( phi) = omS
                        dg::Timer t;
                        t.tic();
                        unsigned number = acc.solve( m_imp[u], omega, omS,
                            m_weights[u], m_eps[u], m_eps[u], 10000, damping, restart, false);
                        t.toc();
                        std::cout << "# Implicit stage "<<u<<" solve took "<<number<<" iterations in "<<t.diff()<<"s\n";
                    };
                }
                else
                {
                    m_imp_split[u] = [&, u] ( const auto& array, auto& f) {
                        // array = {omega, phi}
                        // f[0] = omega - a I(omega, phi)
                        // f[1] = omega + Delta Phi
                        m_eqs[u].compute_omega( array[1], f[1] );
                        m_eqs[u].rhs( m_time, array[0], array[1], f[0]);
                        dg::blas1::axpby( 1., array[0], -m_alpha, f[0]);
                        dg::blas1::axpby( 1., array[0], -1., f[1]);
                    };
                    m_inv_imp_split[u] = [this, u, damping, restart, acc =
                        dg::AndersonAcceleration<std::array<Container,2>>({m_weights[u], m_weights[u]}, restart),
                        weights = std::array<Container,2>({m_weights[u], m_weights[u]})]
                        ( const auto& omS, auto& phi) mutable
                    {
                        // omS = {omega*, 0}
                        // phi = {omega, phi}
                        // Solve Implicit( phi) = omS
                        dg::Timer t;
                        t.tic();
                        unsigned number = acc.solve( m_imp_split[u], phi, omS,
                            weights, m_eps[u], m_eps[u], 10000, damping, restart, false);
                        t.toc();
                        std::cout << "# Implicit solve took "<<number<<" iterations in "<<t.diff()<<"s\n";
                    };
                }
            }
        }
    }

    void operator()(double t, const Container& omega, Container& omS)
    {
        m_eqs[0]( t, omega, omS);
    }
    void operator()(double alpha, double time, Container& omega, const Container& omS)
    {
        // Note how the operators and solvers hold these as references
        // so they know when they are updated
        m_time = time, m_alpha = alpha;
        // Solve the implicit equation
        if( m_system == "single")
        {
            dg::nested_iterations( m_imp, m_phi, omS, m_inv_imp, m_nested);
            m_eqs[0].compute_omega( m_phi, omega);
        }
        else if ( m_system == "omega")
        {
            dg::nested_iterations( m_imp, omega, omS, m_inv_imp, m_nested);
        }
        else
        {
            dg::blas1::copy( omS, m_omS_split[0]);
            dg::blas1::copy( 0., m_omS_split[1]);
            dg::nested_iterations( m_imp_split, m_phi_split, m_omS_split, m_inv_imp_split, m_nested_split);
            dg::blas1::copy( m_phi_split[0], omega);
        }
    }
    private:
    // We can't copy due to capture of this
    Implicit( const Implicit&);
    Implicit( Implicit&&);
    std::string m_system;
    std::vector<double> m_eps;
    double m_time, m_alpha;
    std::vector< Container> m_weights;
    Container m_phi;
    std::vector<Shu<Geometry, Matrix,Container>> m_eqs;

    dg::NestedGrids< Geometry, Matrix, Container> m_nested;
    std::vector< std::function<void( const Container&, Container&)>> m_imp, m_inv_imp;

    dg::NestedGrids< Geometry, Matrix, std::array<Container,2>> m_nested_split;
    std::vector< std::function<void( const std::array<Container,2>&, std::array<Container,2>&)>>
        m_imp_split, m_inv_imp_split;
    std::array<Container,2> m_omS_split, m_phi_split;
};

}//namespace shu

#endif //_DG_SHU_CUH
