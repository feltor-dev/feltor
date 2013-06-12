#ifndef _DG_GAMMA_
#define _DG_GAMMA_


#include <cassert>
#include "blas.h"

namespace dg{

/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @addtogroup creation
 * Discretization of \f[ (1+\alpha\Delta) \f]
 * can be used in conjugate gradient
 * @tparam Matrix The cusp-matrix class you want to use
 * @tparam Prec The type of preconditioner you want to use
 */
template< class Matrix, typename Prec>
struct Gamma
{
    /**
     * @brief Construct from existing matrices
     *
     * Since memory is small on gpus Gamma can be constructed using an existing laplace operator
     * @param laplaceM negative normalised laplacian
     * @param p preconditioner ( W2D or T2D); makes the matrix symmetric and is the same you later use in conjugate gradients
     * @param alpha prefactor of laplacian
     */
    Gamma( const Matrix& laplaceM, const Prec& p, double alpha):p_(p), laplaceM_(laplaceM), alpha_( alpha){ }
    /**
     * @brief apply operator
     *
     * same as blas2::symv( gamma, x, y);
     * \f[ y = ( 1 + \alpha\Delta) x \f]
     * @tparam Vector The vector class
     * @param x lhs
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM and thus multiplies by -alpha
     */
    template <class Vector>
    void symv( const Vector& x, Vector& y) const
    {
        if( alpha_ != 0);
            blas2::symv( laplaceM_, x, y);
        blas1::axpby( 1., x, -alpha_, y);
        blas2::symv( p_, y,  y);
    }
    double& alpha( ){  return alpha_;}
    double alpha( ) const  {return alpha_;}
  private:
    const Prec& p_;
    const Matrix& laplaceM_;
    double alpha_;
};

///@cond
template< class M, class T>
struct MatrixTraits< Gamma<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

} //namespace dg
#endif//_DG_GAMMA_

