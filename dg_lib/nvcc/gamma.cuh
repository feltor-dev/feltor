#ifndef _DG_GAMMA_
#define _DG_GAMMA_


#include "blas.h"

namespace dg{


template< class Matrix, typename Post>
struct Gamma
{
    Gamma( const Matrix& laplaceM, const Post& p, double tau, double mu):p_(p), laplaceM_(laplaceM), tau_(tau), mu_(mu){}
    template <class Vector>
    void symv( const Vector& x, Vector& y) const
    {
        blas2::symv( laplaceM_, x, y);
        blas1::axpby( 1., x, 0.5*tau_*mu_, y);
        blas2::symv( p_, y,  y);
    }
  private:
    const Post& p_;
    const Matrix& laplaceM_;
    double tau_, mu_;
};


template< class M, class T>
struct MatrixTraits< Gamma<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

} //namespace dg
#endif//_DG_GAMMA_

