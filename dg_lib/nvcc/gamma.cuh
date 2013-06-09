#ifndef _DG_GAMMA_
#define _DG_GAMMA_


#include <cassert>
#include "blas.h"

namespace dg{


/**
 * @brief Matrix class that represents the gyroaveraging operator Gamma
 *
 * Discretization of \f[ (1-\frac{1}{2}\tau\mu\Delta) \f]
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
     * @param tau temperature
     * @param mu mass 
     */
    Gamma( const Matrix& laplaceM, const Prec& p, double tau, double mu):p_(p), laplaceM_(laplaceM), tau_(tau), mu_(mu){
        assert( tau_ > 0 && mu_ > 0 );
    }
    /**
     * @brief apply operator
     *
     * same as blas2::symv( gamma, x, y);
     * \f[ y = ( 1- \frac{1}{2}\tau\mu\Delta) x \f]
     * @tparam Vector The vector class
     * @param x lhs
     * @param y rhs contains solution
     */
    template <class Vector>
    void symv( const Vector& x, Vector& y) const
    {
        blas2::symv( laplaceM_, x, y);
        blas1::axpby( 1., x, 0.5*tau_*mu_, y);
        blas2::symv( p_, y,  y);
    }
    void set_species( double tau, double mu){ 
        assert( tau_ > 0 && mu > 0);
        tau_ = tau, mu_ = mu;}
  private:
    const Prec& p_;
    const Matrix& laplaceM_;
    double tau_, mu_;
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

