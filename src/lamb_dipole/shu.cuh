#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <exception>
#include <cusp/ell_matrix.h>

#include "json/json.h"
#include "file/json_utilities.h"
#include "dg/algorithm.h"

namespace shu
{

struct Upwind{
    DG_DEVICE
    void operator()( double& result, double fw, double bw, double v)
    {
        if( v > 0)
            result -= bw; // yp = - Div( F)
        else
            result -= fw;
    }
};
template< class Geometry, class Matrix, class Container>
struct Diffusion
{
    Diffusion( const Geometry& g, Json::Value& js, enum file::error mode)
    {
        std::string regularization = file::get( mode, js, "regularization", "type", "modal").asString();
        if( "viscosity" == regularization)
        {
            m_nu = file::get( mode, js, "regularization", "nu_perp", 1e-3).asDouble();
            m_order = file::get( mode, js, "regularization", "order", 1).asUInt();
            m_temp = dg::evaluate( dg::zero, g);
            m_LaplacianM.construct( g, dg::normed);
        }
    }
    void operator()(double t, const Container& x, Container& y)
    {
        dg::blas1::copy( x, y);
        for( unsigned p=0; p<m_order; p++)
        {
            using std::swap;
            swap( m_temp, y);
            dg::blas2::symv( m_nu, m_LaplacianM, m_temp, 0., y);
        }
        dg::blas1::scal( y, -1.);
    }
    const Container& weights(){ return m_LaplacianM.weights();}
    const Container& inv_weights(){ return m_LaplacianM.inv_weights();}
    const Container& precond(){ return m_LaplacianM.precond();}
  private:
    double m_nu;
    unsigned m_order;
    Container m_temp;
    dg::Elliptic<Geometry, Matrix,Container> m_LaplacianM;
};

template< class Geometry, class IMatrix, class Matrix, class Container >
struct Shu
{
    using value_type = dg::get_value_type<Container>;

    Shu( const Geometry& grid, Json::Value& js, enum file::error mode);

    const dg::Elliptic<Matrix, Container, Container>& lap() const { return m_multi_laplaceM[0];}
    dg::ArakawaX<Geometry, Matrix, Container>& arakawa() {return m_arakawa;}

    const Container& potential( ) {return m_psi;}

    void operator()(double t, const Container& y, Container& yp);

    void variation( const Container& phi, Container& variation_phi){
        dg::blas2::symv( m_centered_phi[0], phi,  m_temp[0]);
        dg::blas2::symv( m_centered_phi[1], phi,  m_temp[1]);
        dg::tensor::multiply2d( m_metric, m_temp[0], m_temp[1], variation_phi, m_temp[2]);
        dg::blas1::pointwiseDot( 1., m_temp[0], variation_phi, 1., m_temp[1], m_temp[2], 0., variation_phi);
    }
  private:
    Container m_psi, m_v, m_temp[3], m_fine_psi, m_fine_v, m_fine_temp[3], m_fine_y, m_fine_yp;
    std::vector<dg::Elliptic<Geometry, Matrix, Container>> m_multi_laplaceM;
    dg::ArakawaX<Geometry, Matrix, Container> m_arakawa;
    dg::Extrapolation<Container> m_old_psi;
    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    IMatrix m_inter, m_project;
    Matrix m_forward[2], m_backward[2], m_centered[2];
    Matrix m_centered_phi[2]; // for variation
    std::vector<double> m_eps;
    std::string m_advection, m_multiplication;
    dg::SparseTensor<Container> m_metric;
};

template<class Geometry, class IMatrix, class Matrix, class Container>
Shu< Geometry, IMatrix, Matrix, Container>::Shu(
        const Geometry& g, Json::Value& js, enum file::error mode):
    m_old_psi( 2, dg::evaluate( dg::zero, g)),
    m_multigrid( g, 3)
{
    m_advection = file::get( mode, js, "advection", "type", "arakawa").asString();
    m_multiplication = file::get( mode, js, "advection", "multiplication", "pointwise").asString();
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
        fine_grid.set( 2*g.n()-1, g.Nx(), g.Ny());
        m_inter = dg::create::interpolation( fine_grid, g);
        m_project = dg::create::projection( g, fine_grid);

        m_centered[0] = dg::create::dx( fine_grid, g.bcx(), dg::centered);
        m_centered[1] = dg::create::dy( fine_grid, g.bcy(), dg::centered);
        m_forward[0] = dg::create::dx( fine_grid, dg::inverse( g.bcx()), dg::forward);
        m_forward[1] = dg::create::dy( fine_grid, dg::inverse( g.bcy()), dg::forward);
        m_backward[0] = dg::create::dx( fine_grid, dg::inverse( g.bcx()), dg::backward);
        m_backward[1] = dg::create::dy( fine_grid, dg::inverse( g.bcy()), dg::backward);
        m_fine_psi = dg::evaluate( dg::zero, fine_grid);
        m_fine_y = dg::evaluate( dg::zero, fine_grid);
        m_fine_yp = dg::evaluate( dg::zero, fine_grid);
        m_fine_v = dg::evaluate( dg::zero, fine_grid);
        m_fine_temp[0] = dg::evaluate( dg::zero, fine_grid);
        m_fine_temp[1] = dg::evaluate( dg::zero, fine_grid);
        m_fine_temp[2] = dg::evaluate( dg::zero, fine_grid);
        m_arakawa.construct( fine_grid);
    }
    else
    {
        m_centered[0] = dg::create::dx( g, g.bcx(), dg::centered);
        m_centered[1] = dg::create::dy( g, g.bcy(), dg::centered);
        m_forward[0] = dg::create::dx( g, dg::inverse( g.bcx()), dg::forward);
        m_forward[1] = dg::create::dy( g, dg::inverse( g.bcy()), dg::forward);
        m_backward[0] = dg::create::dx( g, dg::inverse( g.bcx()), dg::backward);
        m_backward[1] = dg::create::dy( g, dg::inverse( g.bcy()), dg::backward);
        m_arakawa.construct( g);
    }

    unsigned stages = file::get( mode, js, "elliptic", "stages", 3).asUInt();
    m_eps.resize(stages);
    m_eps[0] = file::get_idx( mode, js, "elliptic", "eps_pol", 0, 1e-6).asDouble();
    for( unsigned i=1;i<stages; i++)
    {
        m_eps[i] = file::get_idx( mode, js, "elliptic", "eps_pol", i, 1).asDouble();
        m_eps[i]*= m_eps[0];
    }
    m_multi_laplaceM.resize(stages);
    for( unsigned u=0; u<stages; u++)
        m_multi_laplaceM[u].construct( m_multigrid.grid(u), dg::not_normed, dg::centered, 1);
}

template< class Geometry, class IMatrix, class Matrix, class Container>
void Shu<Geometry, IMatrix, Matrix, Container>::operator()(double t, const Container& y, Container& yp)
{
    //solve elliptic equation
    m_old_psi.extrapolate( t, m_psi);
    std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_laplaceM, m_psi, y, m_eps);
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
            //dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_v); //v_x
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_x
            dg::blas2::symv( m_forward[0], m_temp[0], m_temp[1]);
            dg::blas2::symv( m_backward[0], m_temp[0], m_temp[2]);
            dg::blas1::subroutine( shu::Upwind(), yp, m_temp[1], m_temp[2], m_v);
            //dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], m_psi, 0., m_v); //v_y
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_y
            dg::blas2::symv( m_forward[1], m_temp[0], m_temp[1]);
            dg::blas2::symv( m_backward[1], m_temp[0], m_temp[2]);
            dg::blas1::subroutine( shu::Upwind(), yp, m_temp[1], m_temp[2], m_v);
        }
        else if( "centered" == m_advection)
        {
            //dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], m_psi, 0., m_v); //v_x
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_x
            dg::blas2::symv( -1., m_centered[0], m_temp[0], 0., yp);
            //dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], m_psi, 0., m_v); //v_y
            dg::blas1::pointwiseDot( y, m_v, m_temp[0]); //f_y
            dg::blas2::symv( -1., m_centered[1], m_temp[0], 1., yp);
        }
    }
    else // "projection " == multiplication
    {
        dg::blas2::symv( m_inter, y, m_fine_y);
        dg::blas2::symv( m_inter, m_psi, m_fine_psi);
        if( "arakawa" == m_advection)
            m_arakawa( m_fine_y, m_fine_psi, m_fine_yp); //A(y,psi)-> yp
        else if( "upwind" == m_advection)
        {
            dg::blas1::copy( 0., m_fine_yp);
            //dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], m_fine_psi, 0., m_fine_v); //v_x
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_x
            dg::blas2::symv( m_forward[0], m_fine_temp[0], m_fine_temp[1]);
            dg::blas2::symv( m_backward[0], m_fine_temp[0], m_fine_temp[2]);
            dg::blas1::subroutine( shu::Upwind(), m_fine_yp, m_fine_temp[1], m_fine_temp[2], m_fine_v);
            //dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], m_fine_psi, 0., m_fine_v); //v_y
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_y
            dg::blas2::symv( m_forward[1], m_fine_temp[0], m_fine_temp[1]);
            dg::blas2::symv( m_backward[1], m_fine_temp[0], m_fine_temp[2]);
            dg::blas1::subroutine( shu::Upwind(), m_fine_yp, m_fine_temp[1], m_fine_temp[2], m_fine_v);
        }
        else if( "centered" == m_advection)
        {
            //dx ( nv_x)
            dg::blas2::symv( -1., m_centered[1], m_fine_psi, 0., m_fine_v); //v_x
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_x
            dg::blas2::symv( -1., m_centered[0], m_fine_temp[0], 0., m_fine_yp);
            //dy ( nv_y)
            dg::blas2::symv( 1., m_centered[0], m_fine_psi, 0., m_fine_v); //v_y
            dg::blas1::pointwiseDot( m_fine_y, m_fine_v, m_fine_temp[0]); //f_y
            dg::blas2::symv( -1., m_centered[1], m_fine_temp[0], 1., m_fine_yp);
        }

        dg::blas2::symv( m_project, m_fine_yp, yp);
    }

}

}//namespace shu

#endif //_DG_SHU_CUH
