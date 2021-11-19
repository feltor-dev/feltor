#pragma once

#include <exception>
#include "dg/algorithm.h"

namespace polar
{

template< class Geometry, class Matrix, class container>
struct Diffusion
{
    Diffusion( const Geometry& g, double nu): m_nu(nu), m_LaplacianM( g)
    {
    }
    void operator()( double t, const container& x, container& y)
    {
        dg::blas2::gemv( m_LaplacianM, x, y);
        dg::blas1::scal( y, -m_nu);
    }
    const container& weights(){return m_LaplacianM.weights();}
    const container& precond(){return m_LaplacianM.precond();}
  private:
    double m_nu;
    dg::Elliptic<Geometry,Matrix,container> m_LaplacianM;
};

template< class Geometry, class Matrix, class container >
struct Explicit
{
    using Vector = container;

    Explicit( const Geometry& grid, double eps);

    const dg::Elliptic<Geometry, Matrix, container>& lap() const { return
        m_laplaceM;}
    dg::ArakawaX<Geometry, Matrix, container>& arakawa() {return m_arakawa;}
    const container& potential( ) {return m_psi;}
    void operator()(double t,  const Vector& y, Vector& yp);
    container m_psi;
    dg::Elliptic<Geometry, Matrix, container> m_laplaceM;
    dg::ArakawaX<Geometry, Matrix, container> m_arakawa;
    dg::PCG<container> m_pcg;
    dg::Extrapolation<container> m_extra;
    double m_eps;
};


template<class Geometry, class Matrix, class container>
Explicit< Geometry, Matrix, container>::Explicit( const Geometry& g, double eps):
    m_psi( dg::evaluate(dg::zero, g) ),
    m_laplaceM( g),
    m_arakawa( g),
    m_pcg( m_psi, g.size()),
    m_extra( 2, m_psi),
    m_eps(eps)
{
}


template<class Geometry, class Matrix, class container>
void Explicit<Geometry, Matrix, container>::operator()(double t, const Vector& y, Vector& yp)
{
    m_extra.extrapolate( t, m_psi);
    m_pcg.solve(m_laplaceM, m_psi, y,
        m_laplaceM.precond(), m_laplaceM.weights(), m_eps );
    m_extra.update( t, m_psi);
    m_arakawa( y, m_psi, yp); //A(y,psi)-> yp
}

}//namespace polar

