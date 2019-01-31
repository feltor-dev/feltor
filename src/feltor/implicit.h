#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/geometries/geometries.h"

//This contains the implicit part of the feltor equations
//We even write a custom solver object for it

namespace feltor
{
namespace routines
{
//Resistivity (consistent density dependency,
//parallel momentum conserving, quadratic current energy conservation dependency)
struct AddResistivity{
    AddResistivity( double eta, std::array<double,2> mu): m_eta(eta){
        m_mu[0] = mu[0], m_mu[1] = mu[1];
    }
    DG_DEVICE
    void operator()( double ne, double ni, double ue,
        double ui, double& dtUe, double& dtUi) const{
        double current = (ne)*(ui-ue);
        dtUe += -m_eta/m_mu[0] * current;
        dtUi += -m_eta/m_mu[1] * (ne)/(ni) * current;
    }
    private:
    double m_eta;
    double m_mu[2];
};
}//namespace routines

template<class Geometry, class Matrix, class Container>
struct ImplicitDensity
{
    ImplicitDensity() = default;
    ImplicitDensity( const Geometry& g, feltor::Parameters p,
            dg::geo::TokamakMagneticField mag)
    {
        construct( g, p, mag);
    }

    void construct( const Geometry& g, feltor::Parameters p,
        dg::geo::TokamakMagneticField mag)
    {
        m_p = p;
        m_lapM_perpN.construct( g, p.bcxN, p.bcyN,dg::PER, dg::normed, dg::centered);
        dg::assign( dg::evaluate( dg::zero, g), m_temp);
        auto bhat = dg::geo::createEPhi(); //bhat = ephi except when "true"
        if( p.curvmode == "true")
            bhat = dg::geo::createBHat(mag);
        dg::SparseTensor<Container> hh
            = dg::geo::createProjectionTensor( bhat, g);
        //set perpendicular projection tensor h
        m_lapM_perpN.set_chi( hh);
    }

    void operator()( double t, const std::array<Container,2>& y,
        std::array<Container,2>& yp)
    {
#if FELTORPERP == 1
        /* y[0] := n_e - 1
           y[1] := N_i - 1
        */
        for( unsigned i=0; i<2; i++)
        {
            //dissipation acts on w!
            if( m_p.perp_diff == "hyperviscous")
            {
                dg::blas2::symv( m_lapM_perpN, y[i],      m_temp);
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpN, m_temp, 0., yp[i]);
            }
            else // m_p.perp_diff == "viscous"
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpN, y[i],  0., yp[i]);
        }
#else
        dg::blas1::copy( 0, yp);
#endif
    }

    const Container& weights() const{
        return m_lapM_perpN.weights();
    }
    const Container& inv_weights() const {
        return m_lapM_perpN.inv_weights();
    }
    const Container& precond() const {
        return m_lapM_perpN.precond();
    }

  private:
    feltor::Parameters m_p;
    Container m_temp;
    dg::Elliptic3d<Geometry, Matrix, Container> m_lapM_perpN;
};

template<class Geometry, class Matrix, class Container>
struct ImplicitVelocity
{
    ImplicitVelocity() = default;
    ImplicitVelocity( const Geometry& g, feltor::Parameters p,
            dg::geo::TokamakMagneticField mag){
        construct( g, p, mag);
    }
    void construct( const Geometry& g, feltor::Parameters p,
            dg::geo::TokamakMagneticField mag)
    {
        m_p=p;
        m_lapM_perpU.construct( g, p.bcxU,p.bcyU,dg::PER,
            dg::normed, dg::centered);
        dg::assign( dg::evaluate( dg::zero, g), m_temp);
        m_apar = m_temp;
        m_fields[0][0] = m_fields[0][1] = m_temp;
        m_fields[1][0] = m_fields[1][1] = m_temp;
        auto bhat = dg::geo::createEPhi(); //bhat = ephi except when "true"
        if( p.curvmode == "true")
            bhat = dg::geo::createBHat(mag);
        dg::SparseTensor<Container> hh
            = dg::geo::createProjectionTensor( bhat, g);
        //set perpendicular projection tensor h
        m_lapM_perpU.set_chi( hh);
        m_induction.construct(  g,
            p.bcxU, p.bcyU, dg::PER, -1., dg::centered);
        m_induction.elliptic().set_chi( hh);
        m_invert.construct( m_temp, g.size(), p.eps_pol,1 );
    }
    void set_density( const std::array<Container, 2>& dens){
        dg::blas1::transform( dens, m_fields[0], dg::PLUS<double>(+1));
        dg::blas1::axpby(  m_p.beta/m_p.mu[1], m_fields[0][1],
                          -m_p.beta/m_p.mu[0], m_fields[0][0], m_temp);
        m_induction.set_chi( m_temp);
    }

    void operator()( double t, const std::array<Container,2>& w,
        std::array<Container,2>& wp)
    {
        //Note that this operator works because N converges
        //independently of W and these terms are linear in W for fixed N
#if FELTORPERP == 1
        /* w[0] := w_e
           w[1] := W_i
        */
        dg::blas1::copy( w, m_fields[1]);
        if( m_p.beta != 0){
            //let us solve for apar
            dg::blas1::pointwiseDot(  1., m_fields[0][1], m_fields[1][1],
                                     -1., m_fields[0][0], m_fields[1][0],
                                      0., m_temp);
            m_invert( m_induction, m_apar, m_temp, weights(),
                inv_weights(), precond());
            //compute u_e and U_i
            dg::blas1::axpby( 1., m_fields[1][0], -m_p.beta/m_p.mu[0],
                m_apar, m_fields[1][0]);
            dg::blas1::axpby( 1., m_fields[1][1], -m_p.beta/m_p.mu[1],
                m_apar, m_fields[1][1]);
        }
        /* fields[0] := u_e
           fields[1] := U_i
        */

        for( unsigned i=0; i<2; i++)
        {
            if( m_p.perp_diff == "hyperviscous")
            {
                dg::blas2::symv( m_lapM_perpU, m_fields[1][i],      m_temp);
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpU, m_temp, 0., wp[i]);
            }
            else // m_p.perp_diff == "viscous"
                dg::blas2::symv( -m_p.nu_perp, m_lapM_perpU,
                    m_fields[1][i],  0., wp[i]);
        }
#else
        dg::blas1::copy( 0, wp);
#endif
        //------------------Add Resistivity--------------------------//
        dg::blas1::subroutine( routines::AddResistivity( m_p.eta, m_p.mu),
            m_fields[0][0], m_fields[0][1],
            m_fields[1][0], m_fields[1][1], wp[0], wp[1]);
    }

    const Container& weights() const{
        return m_lapM_perpU.weights();
    }
    const Container& inv_weights() const {
        return m_lapM_perpU.inv_weights();
    }
    const Container& precond() const {
        return m_lapM_perpU.precond();
    }

  private:
    feltor::Parameters m_p;
    Container m_temp, m_apar;
    dg::Invert<Container> m_invert;
    dg::Helmholtz3d<Geometry, Matrix, Container> m_induction;
    std::array<std::array<Container,2>,2> m_fields;
    dg::Elliptic3d<Geometry, Matrix, Container> m_lapM_perpU;
};

template<class Geometry, class Matrix, class Container>
struct Implicit
{
    Implicit() = default;
    Implicit( const Geometry& g, feltor::Parameters p,
            dg::geo::TokamakMagneticField mag):
            m_dens( g,p,mag), m_velo( g,p,mag){}
    void operator()( double t, const std::array<std::array<Container,2>,2>& y,
        std::array<std::array<Container,2>,2>& yp)
    {
        m_dens( t,y[0], yp[0]);
        m_velo.set_density( y[0]);
        m_velo( t,y[1], yp[1]);
    }
    private:
    ImplicitDensity <Geometry, Matrix, Container> m_dens;
    ImplicitVelocity<Geometry, Matrix, Container> m_velo;
};

//compute unnormalized y + alpha f(y,t)
template< class LinearOp, class Container>
struct Helmholtz
{
    using value_type = dg::get_value_type<Container>;
    Helmholtz(){}
    Helmholtz( value_type alpha, value_type t, LinearOp* f): f_(f), alpha_(alpha), t_(t){}
    void construct( value_type alpha, value_type t, LinearOp* f){
        f_ = f; alpha_=alpha; t_=t;
    }
    void symv( const std::array<Container,2>& x, std::array<Container,2>& y)
    {
        if( alpha_ != 0)
            (*f_)(t_,x,y);
        dg::blas1::axpby( 1., x, alpha_, y, y);
        dg::blas2::symv( f_->weights(), y, y);
    }
    const Container& weights() const{
        return f_->weights();
    }
    const Container& inv_weights() const {
        return f_->inv_weights();
    }
    const Container& precond() const {
        return f_->precond();
    }
  private:
    LinearOp* f_;
    value_type alpha_;
    value_type t_;
};

/*!@brief Multigrid Solver class for solving \f[ (y+\alpha\hat I(t,y)) = \rho\f]
*
*/
template< class Geometry, class Matrix, class Container>
struct FeltorSpecialSolver
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    FeltorSpecialSolver(){}

    FeltorSpecialSolver( const Geometry& grid, Parameters p,
        dg::geo::TokamakMagneticField mag ):
        m_multigrid(grid, p.stages),
        m_multi_imdens(p.stages),
        m_multi_imvelo(p.stages)
    {
        for( unsigned u=0; u<m_multigrid.stages(); u++)
        {
            m_multi_imdens[u].construct( m_multigrid.grid(u),
                p, mag);
            m_multi_imvelo[u].construct( m_multigrid.grid(u),
                p, mag);
        }
        std::array<Container,2> temp = dg::construct<std::array<Container,2>>( dg::evaluate( dg::zero, grid));
        m_multi_n = m_multigrid.project( temp);
        m_eps = p.eps_time;
    }

    //this must return the class used in the timestepper
    std::array<std::array<Container,2>,2> copyable()const{
        return std::array<std::array<Container,2>,2>{ m_multi_n[0], m_multi_n[0]};
    }

    //Solve y + a I(t,y) = rho
    void solve( value_type alpha,
        Implicit<Geometry, Matrix,Container>& im,
        value_type t,
        std::array<std::array<Container,2>,2>& y,
        const std::array<std::array<Container,2>,2>& rhs)
    {
        //1. First construct Helmholtz type functor hierarchy
        std::vector<
          Helmholtz<ImplicitDensity<Geometry,Matrix,Container>,
          Container>> multi_helm_dens( m_multigrid.stages());
        std::vector<
          Helmholtz<ImplicitVelocity<Geometry,Matrix,Container>,
          Container>> multi_helm_velo(
        m_multigrid.stages());
        for( unsigned i=0; i<m_multigrid.stages(); i++)
        {
            multi_helm_dens[i].construct( alpha, t, &m_multi_imdens[i]);
            multi_helm_velo[i].construct( alpha, t, &m_multi_imvelo[i]);
        }
        //2. Solve the density equation

        std::vector<unsigned> number =
            m_multigrid.direct_solve( multi_helm_dens, y[0], rhs[0], m_eps);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_eps);
#ifdef DG_BENCHMARK
        //std::cout << "# iterations implicit density:  "<<number[0]<<"\n";
#endif
        //3. project density to all grids and pass to velocity Equation
        m_multigrid.project( y[0], m_multi_n);
        for( unsigned i=0; i<m_multigrid.stages(); i++)
            m_multi_imvelo[i].set_density( m_multi_n[i]);
        //4. Solve Velocity equation
        number = m_multigrid.direct_solve( multi_helm_velo, y[1], rhs[1], m_eps);
        if(  number[0] == m_multigrid.max_iter())
            throw dg::Fail( m_eps);
#ifdef DG_BENCHMARK
        //std::cout << "# iterations implicit velocity: "<<number[0]<<"\n";
#endif

    }
    private:
    dg::MultigridCG2d< Geometry, Matrix, std::array<Container,2>> m_multigrid;
    std::vector<ImplicitDensity<Geometry,Matrix,Container>> m_multi_imdens;
    std::vector<ImplicitVelocity<Geometry,Matrix,Container>> m_multi_imvelo;
    std::vector<std::array<Container,2>> m_multi_n;
    value_type m_eps;
};

}//namespace feltor
namespace dg{
template< class M, class V>
struct TensorTraits< feltor::Helmholtz<M, V> >
{
    using value_type = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
} //namespace dg
