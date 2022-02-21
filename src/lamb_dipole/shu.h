#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <exception>
#include <cusp/ell_matrix.h>

#include "json/json.h"
#include "dg/file/json_utilities.h"
#include "dg/algorithm.h"
#include "init.h"

namespace shu
{

// Improvement: make Diffusion a friend to Shu to save memory and duplicate code
template< class Geometry, class Matrix, class Container>
struct Diffusion;

template< class Geometry, class Matrix, class Container >
struct Shu
{
    friend class Diffusion<Geometry, Matrix, Container>;
    using value_type = dg::get_value_type<Container>;

    Shu( const Geometry& grid, dg::file::WrappedJsonValue& js);

    const dg::Elliptic<Matrix, Container, Container>& lap() const { return m_multi_laplaceM[0];}
    dg::ArakawaX<Geometry, Matrix, Container>& arakawa() {return m_arakawa;}

    const Container& potential( ) {return m_psi;}
    Matrix& dx() {return m_centered[0];}
    Matrix& dy() {return m_centered[1];}

    void operator()(double t, const Container& y, Container& yp);
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
    if( stepper == "FilteredImplicitMultistep" || stepper == "DIRK")
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
    if( number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_eps[0]);
    //now do advection with various schemes
    if( "pointwise" == m_multiplication)
    {
        if( "arakawa" == m_advection)
            m_arakawa( y, m_psi, yp); //A(y,psi)-> yp
        else if( "upwind" == m_advection)
        {
            dg::blas1::copy( 0., yp);
            // - dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_v); //v_x
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_x
            dg::blas2::symv( m_forward[0], m_temp[0], m_temp[1]);
            dg::blas2::symv( m_backward[0], m_temp[0], m_temp[2]);
            dg::blas1::evaluate( yp, dg::minus_equals(), dg::Upwind(), m_v, m_temp[2], m_temp[1]);
            // - dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], m_psi, 0., m_v); //v_y
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_y
            dg::blas2::symv( m_forward[1], m_temp[0], m_temp[1]);
            dg::blas2::symv( m_backward[1], m_temp[0], m_temp[2]);
            dg::blas1::evaluate( yp, dg::minus_equals(), dg::Upwind(), m_v, m_temp[2], m_temp[1]);
        }
        else if( "centered" == m_advection)
        {
            // - dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_v); //v_x
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_x
            dg::blas2::symv( -1., m_centered[0], m_temp[0], 0., yp);
            // - dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], m_psi, 0., m_v); //v_y
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_y
            dg::blas2::symv( -1., m_centered[1], m_temp[0], 1., yp);
        }
        else if( "upwind-advection" == m_advection)
        {
            //  - v_x dx n
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_temp[0]); //v_x

            //  - v_y dy n
            dg::blas2::symv( 1., m_centered[0], m_psi, 0., m_temp[1]); //v_y
            m_adv.upwind( -1., m_temp[0], m_temp[1], y, 0., yp);
        }
        else if( "centered-advection" == m_advection)
        {
            //dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_v); //v_x
            dg::blas2::symv( -1., m_centered[0], y, 0., m_temp[0]); // -v_x dx omega
            dg::blas1::pointwiseDot( m_v, m_temp[0], yp);
            //dy ( nv_y)
            dg::blas2::symv(  1., m_centered[0], m_psi, 0., m_v); //v_y
            dg::blas2::symv( -1., m_centered[1], y, 0., m_temp[0]);
            dg::blas1::pointwiseDot( 1., m_v, m_temp[0], 1., yp); //f_y
        }
    }
    else // "projection " == multiplication
    {
        if( "arakawa" == m_advection)
        {
            dg::blas2::symv( m_inter, y, m_fine_y);
            dg::blas2::symv( m_inter, m_psi, m_fine_psi);
            m_arakawa( m_fine_y, m_fine_psi, m_fine_yp); //A(y,psi)-> yp
        }
        else if( "upwind" == m_advection)
        {
            dg::blas2::symv( m_inter, y, m_fine_y);
            dg::blas2::symv( m_inter, m_psi, m_fine_psi);
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
            dg::blas2::symv( m_inter, m_psi, m_fine_psi);
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
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_v); //v_x
            dg::blas2::symv( m_forward[0], y, m_temp[1]);
            dg::blas2::symv( m_backward[0], y, m_temp[2]);
            dg::blas2::symv( m_inter, m_v, m_fine_v);
            dg::blas2::symv( m_inter, m_temp[1], m_fine_temp[1]);
            dg::blas2::symv( m_inter, m_temp[2], m_fine_temp[2]);
            dg::blas1::evaluate( m_fine_yp, dg::minus_equals(), dg::UpwindProduct(), m_fine_v, m_fine_temp[2], m_fine_temp[1]);
            // v_y dy n
            dg::blas2::symv( 1., m_centered[0], m_psi, 0., m_v); //v_y
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
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_v); //v_x
            dg::blas2::symv( -1., m_centered[0], y, 0., m_temp[0]);
            dg::blas2::symv( m_inter, m_temp[0], m_fine_temp[0]);
            dg::blas2::symv( m_inter, m_v, m_fine_v);
            dg::blas1::pointwiseDot( m_fine_v, m_fine_temp[0], m_fine_yp);
            // v_y dy n
            dg::blas2::symv(  1., m_centered[0], m_psi, 0., m_v); //v_y
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
        m_weights =  m_shu->m_LaplacianM.weights();
    }
    void operator()(double t, const Container& x, Container& y)
    {
        m_shu->add_implicit( 1., x, 0., y);
    }
    const Container& weights(){
        return m_weights;
    }
    const Container& precond(){ return m_shu->m_LaplacianM.precond();}
  private:
    Container m_weights;
    Shu<Geometry,Matrix,Container>* m_shu;
};

}//namespace shu

#endif //_DG_SHU_CUH
