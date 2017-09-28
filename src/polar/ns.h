#pragma once

#include <exception>
#include <cusp/ell_matrix.h>

#include "dg/blas.h"
#include "dg/arakawa.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

namespace polar
{
template< class Geometry, class Matrix, class container>
struct Diffusion
{
    Diffusion( const Geometry& g, double nu): nu_(nu),
        w2d( dg::create::volume( g) ), v2d( dg::create::inv_volume(g) ) ,
        LaplacianM( g, dg::normed)
    { 
    }
    void operator()( const container& x, container& y)
    {
        dg::blas2::gemv( LaplacianM, x, y);
        dg::blas1::scal( y, -nu_);
    }
    const container& weights(){return w2d;}
    const container& inv_weights(){return v2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    dg::Elliptic<Geometry,Matrix,container> LaplacianM;
};

template< class Geometry, class Matrix, class container >
struct Explicit 
{
    typedef container Vector;

    Explicit( const Geometry& grid, double eps);

    const dg::Elliptic<Matrix, container, container>& lap() const { return laplaceM;}
    dg::ArakawaX<Geometry, Matrix, container>& arakawa() {return arakawa_;}
    /**
     * @brief Returns psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return psi is the potential
     */
    const container& potential( ) {return psi;}
    void operator()( const Vector& y, Vector& yp);
  private:
    container psi;
    dg::Elliptic<Geometry, Matrix, container> laplaceM;
    dg::ArakawaX<Geometry, Matrix, container> arakawa_; 
    dg::Invert<container> invert;
};

template<class Geometry, class Matrix, class container>
Explicit< Geometry, Matrix, container>::Explicit( const Geometry& g, double eps): 
    psi( dg::evaluate(dg::zero, g) ),
    laplaceM( g, dg::not_normed),
    arakawa_( g), 
    invert( psi, g.size(), eps)
{
}

template<class Geometry, class Matrix, class container>
void Explicit<Geometry, Matrix, container>::operator()( const Vector& y, Vector& yp)
{
    invert( laplaceM, psi, y);
    arakawa_( y, psi, yp); //A(y,psi)-> yp
}

}//namespace polar

