#pragma once

#include <exception>
#include <cusp/ell_matrix.h>

#include "dg/blas.h"
#include "dg/arakawa.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

namespace dg
{
template< class Geometry, class Matrix, class container>
struct Diffusion
{
    Diffusion( const Geometry& g, double nu): nu_(nu),
        w2d( dg::create::weights( g) ), v2d( dg::create::inv_weights(g) ) ,
        LaplacianM( g, dg::normed)
    { 
    }
    void operator()( const container& x, container& y)
    {
        dg::blas2::gemv( LaplacianM, x, y);
        dg::blas1::scal( y, -nu_);
    }
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    dg::Elliptic<Geometry,Matrix,container> LaplacianM;
};

template< class Geometry, class Matrix, class container >
struct Shu 
{
    typedef container Vector;

    Shu( const Geometry& grid, double eps);

    const Elliptic<Matrix, container, container>& lap() const { return laplaceM;}
    ArakawaX<Geometry, Matrix, container>& arakawa() {return arakawa_;}
    /**
     * @brief Returns psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return psi is the potential
     */
    const container& potential( ) {return psi;}
    void operator()( Vector& y, Vector& yp);
  private:
    container psi, w2d, v2d;
    Elliptic<Geometry, Matrix, container> laplaceM;
    ArakawaX<Geometry, Matrix, container> arakawa_; 
    Invert<container> invert;

    Geometry grid;
    DVec tmp;
};


template<class Geometry, class Matrix, class container>
Shu< Geometry, Matrix, container>::Shu( const Geometry& g, double eps): 
    psi( evaluate(dg::zero, g) ),
    w2d( create::weights( g)), v2d( create::inv_weights(g)),  
    laplaceM( g, not_normed),
    arakawa_( g), 
    invert( psi, g.size(), eps),
    grid(g)
{
}

template<class Geometry, class Matrix, class container>
void Shu<Geometry, Matrix, container>::operator()( Vector& y, Vector& yp)
{
    tmp = y;
    geo::multiplyPerpVolume(tmp, grid);
    invert( laplaceM, psi, tmp, w2d, v2d);
    arakawa_( y, psi, yp); //A(y,psi)-> yp
}

}//namespace dg

