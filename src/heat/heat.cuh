#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "dg/geometries/geometries.h"

namespace heat
{

template< class Geometry, class IMatrix, class Matrix, class container>
struct Implicit
{
    Implicit( const Geometry& g, Parameters p, dg::geo::solovev::Parameters gp):
        p(p),
        m_ds(dg::geo::createSolovevField( gp), g, p.bcx, p.bcy,
              dg::geo::PsiLimiter(
                dg::geo::solovev::Psip(gp), gp.psipmaxlim),
              dg::forward,
              gp.rk4eps, p.mx, p.my),
        m_elliptic( g, dg::normed, dg::forward)
    {
        dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
        m_elliptic.set_x(
          dg::construct<container>(dg::pullback( dg::geo::BFieldR(c),g)));
        m_elliptic.set_y(
          dg::construct<container>(dg::pullback( dg::geo::BFieldZ(c),g)));
        m_elliptic.set_z(
          dg::construct<container>(dg::pullback( dg::geo::BFieldP(c),g)));
    }
    void operator()( double t, const container& x, container& y)
    {
        if (p.p_diff == "adjoint")    {
            m_ds.symv( p.nu_parallel, x, 0., y);
        }
        else if (p.p_diff == "elliptic")    {
            dg::blas2::symv( m_elliptic, x, y);
            dg::blas1::scal( y, -p.nu_parallel ); //laplace is negative
        }
        else
        {
            dg::blas1::scal( y,0.);
        }
    }
    const container& weights(){return m_elliptic.weights();}
    const container& inv_weights(){return m_elliptic.inv_weights();}
    const container& precond(){return m_elliptic.precond();}
  private:
    const heat::Parameters p;
    dg::geo::DS<Geometry, IMatrix, Matrix, container> m_ds;
    dg::GeneralEllipticSym<Geometry, Matrix, container> m_elliptic;

};

template< class Geometry, class IMatrix, class Matrix, class container >
struct Explicit
{
    Explicit( const Geometry& grid,
              heat::Parameters p,
              dg::geo::solovev::Parameters gp);

    const dg::geo::DS<Geometry,IMatrix,Matrix,container>& ds(){
        return m_ds;
    }
    void operator()( double t, const container& y, container& yp);

    double energy( ) const {
        return m_heat;
    }
    double energy_diffusion( ) const {
        return m_heat_diff;
    }
    double entropy( ) const {
        return m_entropy;
    }
    double entropy_diffusion( ) const {
        return m_entropy_diff;
    }
    void energies( const container& y);
  private:
    container chi, omega, lambda;
    container m_invB, m_divb;
//     ,pupil;
    const container  one;
    const container w3d, v3d;

    //matrices and solvers
    dg::geo::DS<Geometry,IMatrix,Matrix,container> m_ds;

    const heat::Parameters p;
    double m_heat = 0, m_heat_diff = 0, m_entropy = 0, m_entropy_diff = 0;
    dg::GeneralEllipticSym<Geometry, Matrix, container> m_elliptic;

};

template<class Geometry, class IMatrix, class Matrix, class container>
Explicit<Geometry,IMatrix,Matrix,container>::Explicit( const Geometry& g, heat::Parameters p, dg::geo::solovev::Parameters gp):
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi),
    one( dg::evaluate( dg::one, g)),
    w3d( dg::create::volume(g)), v3d( dg::create::inv_volume(g)),
    m_ds( dg::geo::createSolovevField(gp), g, p.bcx, p.bcy, dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), dg::forward),
    p(p),
    m_elliptic( g, dg::normed, dg::forward)
{
    //----------------------------init fields----------------------
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
    dg::assign(  dg::pullback(dg::geo::InvB(c), g), m_invB);
    dg::assign(  dg::pullback(dg::geo::Divb(c), g), m_divb);
    m_elliptic.set_x(
        dg::construct<container>(dg::pullback( dg::geo::BFieldR(c),g)));
    m_elliptic.set_y(
        dg::construct<container>(dg::pullback( dg::geo::BFieldZ(c),g)));
    m_elliptic.set_z(
        dg::construct<container>(dg::pullback( dg::geo::BFieldP(c),g)));
}

template<class G,class I, class M, class V>
void Explicit<G,I,M,V>::energies( const V& y)
{
    m_heat    = dg::blas2::dot( one, w3d, y);
    m_entropy = dg::blas2::dot( y, w3d, y);
    //Compute rhs of energy theorem
    double Dpar [] = {0,0};
    if (p.p_diff == "adjoint")    {
        m_ds.symv( y, chi);
    }
    else if (p.p_diff == "non-adjoint")    {
        m_ds.ds( dg::centered, y, chi);
        dg::blas1::pointwiseDot(m_divb, chi, chi); // (Div b) ds T
        m_ds.dss(1., y, 1., chi);                  // ds^2 T
    }
    else if (p.p_diff == "elliptic")    {
        dg::blas2::symv(m_elliptic, y, chi);
        dg::blas1::scal( chi, -1.);
    }
    Dpar[0] = p.nu_parallel*dg::blas2::dot(one, w3d, chi);
    Dpar[1] = p.nu_parallel*dg::blas2::dot(y, w3d, chi);
    m_heat_diff = Dpar[0];
    m_entropy_diff = Dpar[1];
}

template<class G, class I, class Matrix, class container>
void Explicit<G,I,Matrix,container>::operator()( double tt, const container& y, container& yp)
{
    // y := T - 1 or T
    dg::Timer t;
    t.tic();
    dg::blas1::scal(yp, 0.);
    //-----------------------parallel dissi------------------------
    if (p.p_diff == "non-adjoint")    {
        m_ds.dss(p.nu_parallel, y, 1., yp);  // ds^2 T
        m_ds.ds( dg::centered, y, lambda);   // (Div b) ds T
        dg::blas1::pointwiseDot( p.nu_parallel, m_divb, lambda, 1., yp);
    }
    t.toc();
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif
    std::cout << "One rhs took "<<t.diff()<<"s\n";
}

} //namespace heat

