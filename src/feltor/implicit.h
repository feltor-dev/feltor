#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/geometries/geometries.h"
#include "feltor.h"

//This contains the implicit part of the feltor equations
//We even write a custom solver object for it

namespace feltor
{

template<class Geometry, class IMatrix, class Matrix, class Container>
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
        auto bhat = dg::geo::createEPhi(+1); //bhat = ephi except when "true"
        if( p.curvmode == "true")
            bhat = dg::geo::createBHat(mag);
        else if( mag.ipol()( g.x0(), g.y0()) < 0)
            bhat = dg::geo::createEPhi(-1);
        dg::SparseTensor<Container> hh
            = dg::geo::createProjectionTensor( bhat, g);
        //set perpendicular projection tensor h
        m_lapM_perpN.set_chi( hh);
        if( p.curvmode != "true")
            m_lapM_perpN.set_compute_in_2d( true);
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

template<class Geometry, class IMatrix, class Matrix, class Container>
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
        if( !(p.perp_diff == "viscous" || p.perp_diff == "hyperviscous") )
            throw dg::Error(dg::Message(_ping_)<<"Warning! perp_diff value '"<<p.perp_diff<<"' not recognized!! I do not know how to proceed! Exit now!");
        dg::assign( dg::evaluate( dg::zero, g), m_temp);
        m_apar = m_temp;
        m_fields[0][0] = m_fields[0][1] = m_temp;
        m_fields[1][0] = m_fields[1][1] = m_temp;
        auto bhat = dg::geo::createEPhi(+1); //bhat = ephi except when "true"
        if( p.curvmode == "true")
            bhat = dg::geo::createBHat(mag);
        else if( mag.ipol()( g.x0(), g.y0()) < 0)
            bhat = dg::geo::createEPhi(-1);
        dg::SparseTensor<Container> hh
            = dg::geo::createProjectionTensor( bhat, g);
        //set perpendicular projection tensor h
        m_lapM_perpU.set_chi( hh);
        if( p.curvmode != "true")
            m_lapM_perpU.set_compute_in_2d(true);
        //m_induction.construct(  g,
        //    p.bcxU, p.bcyU, dg::PER, -1., dg::centered);
        //m_induction.elliptic().set_chi( hh);
        //m_invert.construct( m_temp, g.size(), p.eps_pol[0],1 );
        //Multigrid setup
        m_multi_induction.resize(p.stages);
        m_multigrid.construct( g, p.stages);
        for( unsigned u=0; u<p.stages; u++)
        {
            dg::SparseTensor<Container> hh = dg::geo::createProjectionTensor(
                bhat, m_multigrid.grid(u));
            m_multi_induction[u].construct(  m_multigrid.grid(u),
                p.bcxU, p.bcyU, dg::PER, -1., dg::centered);
            m_multi_induction[u].elliptic().set_chi( hh);
            if( p.curvmode != "true")
                m_multi_induction[u].elliptic().set_compute_in_2d(true);
        }
        m_multi_chi = m_multigrid.project( m_temp);
        m_old_apar = dg::Extrapolation<Container>( 2, dg::evaluate( dg::zero, g));
    }
    void set_density( const std::array<Container, 2>& dens){
        dg::blas1::transform( dens, m_fields[0], dg::PLUS<double>(+1));
        if( m_p.beta != 0)
        {
            dg::blas1::axpby(  m_p.beta/m_p.mu[1], m_fields[0][1],
                              -m_p.beta/m_p.mu[0], m_fields[0][0], m_temp);
            m_multigrid.project( m_temp, m_multi_chi);
            for( unsigned u=0; u<m_p.stages; u++)
                m_multi_induction[u].set_chi( m_multi_chi[u]);
        }
    }
    void update(){
        if( m_p.beta != 0)
            m_old_apar.update( m_apar);
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
            dg::blas1::pointwiseDot(  m_p.beta, m_fields[0][1], m_fields[1][1],
                                     -m_p.beta, m_fields[0][0], m_fields[1][0],
                                      0., m_temp);
            //m_invert( m_induction, m_apar, m_temp, weights(),
            //    inv_weights(), precond());
            m_old_apar.extrapolate( m_apar);
            //dg::blas1::scal( m_apar, 0.);
            std::vector<unsigned> number = m_multigrid.direct_solve(
                m_multi_induction, m_apar, m_temp, m_p.eps_pol[0]); //eps_pol[0] on all grids
            //m_old_apar.update( m_apar); //don't update here: makes the solver potentially unstable
            if(  number[0] == m_multigrid.max_iter())
                throw dg::Fail( m_p.eps_pol[0]);

            //compute u_e and U_i from w_e, W_i and apar
            dg::blas1::axpby( 1., m_fields[1][0], -1./m_p.mu[0],
                m_apar, m_fields[1][0]);
            dg::blas1::axpby( 1., m_fields[1][1], -1./m_p.mu[1],
                m_apar, m_fields[1][1]);
        }
        /* fields[1][0] := u_e
           fields[1][1] := U_i
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
    //dg::Invert<Container> m_invert;
    //dg::Helmholtz3d<Geometry, Matrix, Container> m_induction;
    dg::MultigridCG2d<Geometry, Matrix, Container> m_multigrid;
    std::vector<dg::Helmholtz3d<Geometry, Matrix, Container>> m_multi_induction;
    dg::Extrapolation<Container> m_old_apar;
    std::vector<Container> m_multi_chi;
    std::array<std::array<Container,2>,2> m_fields;
    dg::Elliptic3d<Geometry, Matrix, Container> m_lapM_perpU;
};

template<class Geometry, class IMatrix, class Matrix, class Container>
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
    ImplicitDensity <Geometry, IMatrix, Matrix, Container> m_dens;
    ImplicitVelocity<Geometry, IMatrix, Matrix, Container> m_velo;
};

/*!@brief Solver class for solving \f[ (y+\alpha\hat I(t,y)) = \rho\f]
*/
template< class Geometry, class IMatrix, class Matrix, class Container>
struct FeltorSpecialSolver
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    FeltorSpecialSolver(){}

    FeltorSpecialSolver( const Geometry& grid, Parameters p,
        dg::geo::TokamakMagneticField mag)
    {
        std::array<Container,2> temp = dg::construct<std::array<Container,2>>( dg::evaluate( dg::zero, grid));
        m_eps = p.eps_time;
        m_solver = dg::DefaultSolver<std::array<Container,2>>(
            temp, grid.size(), p.eps_time);
        m_imdens.construct( grid, p, mag);
        m_imvelo.construct( grid, p, mag);
    }

    //this must return the class used in the timestepper
    std::array<std::array<Container,2>,2> copyable()const{

        return std::array<std::array<Container,2>,2>{ m_solver.copyable(), m_solver.copyable()};
    }

    //Solve y + a I(t,y) = rho
    void solve( value_type alpha,
        Implicit<Geometry, IMatrix, Matrix,Container>& im,
        value_type t,
        std::array<std::array<Container,2>,2>& y,
        const std::array<std::array<Container,2>,2>& rhs)
    {

        m_solver.solve( alpha, m_imdens, t, y[0], rhs[0]);
        m_imvelo.set_density( y[0]);
        m_solver.solve( alpha, m_imvelo, t, y[1], rhs[1]);
        m_imvelo.update();
    }
    private:
    dg::DefaultSolver<std::array<Container,2>> m_solver;
    ImplicitDensity<Geometry,IMatrix, Matrix,Container> m_imdens;
    ImplicitVelocity<Geometry,IMatrix, Matrix,Container> m_imvelo;
    value_type m_eps;
};

}//namespace feltor
