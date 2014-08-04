#pragma once

#include <cassert>

#include "backend/matrix_traits.h"
#include "blas.h"
#include "cg.h"

#ifdef DG_BENCHMARK
#include "backend/timer.cuh"
#endif

namespace dg{

///@addtogroup operators
///@{
/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * Discretization of \f[ (1+\alpha\Delta) \f]
 * can be used by the Invert class
 * @tparam Matrix The cusp-matrix class you want to use
 * @tparam Prec The type of preconditioner you want to use
 */
template< class Matrix, class Vector, class Prec> 
struct Helmholtz
{
    /**\
     * @brief Construct from existing matrices
     *
     * Since memory is small on gpus Helmholtz can be constructed using an existing laplace operator
     * @param laplaceM negative normalised laplacian
     * @param weights ( W2D or W3D); makes the matrix symmetric and is the same you later use in conjugate gradients
     * @param precond ( V2D or V3D); precondtioner you later use in conjugate gradients
     * @param alpha prefactor of laplacian
     * @note only references are stored so make sure the matrix and the vectors are already allocated when using an object otherwise an out of memory error might occur on gpus
     */
    Helmholtz( const Matrix& laplaceM, const Prec& weights, const Prec& precond, double alpha):p_(weights), q_(precond), laplaceM_(laplaceM), alpha_( alpha){
        }
    /**
     * @brief apply operator
     *
     * Computes
     * \f[ y = W( 1 + \alpha\Delta) x \f]
     * to make the matrix symmetric
     * @param x lhs
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM and thus multiplies by -alpha
     */
    void symv( Vector& x, Vector& y) const
    {
        if( alpha_ != 0);
            blas2::symv( laplaceM_, x, y);
        blas1::axpby( 1., x, -alpha_, y);
        blas2::symv( p_, y,  y);
    }
    const Prec& weights()const {return p_;}
    const Prec& precond()const {return q_;}
    double& alpha( ){  return alpha_;}
    double alpha( ) const  {return alpha_;}
  private:
    const Prec& p_, q_;
    const Matrix& laplaceM_;
    double alpha_;
};

/**
 * @brief Matrix class that represents a Helmholtz-type operator that appears in the parallel induction equation
 *
 * Discretization of \f[ (\alpha\Delta + \chi) \f]
 * can be used by the Invert class
 * @tparam Matrix The cusp-matrix class you want to use
 * @tparam container The type of Vector you want to use
 */
template< class Matrix, class Vector, class Precon>
struct Maxwell
{
    /**
     * @brief Construct from existing matrices
     *
     * Since memory is small on gpus Maxwell can be constructed using an existing laplace operator
     * @param laplaceM negative normalised laplacian
     * @param weights weights
     * @param precon preconditioner
     * @param alpha The factor alpha
     */
    Maxwell( const Matrix& laplaceM, const Precon& weights, const Precon& precon,  double alpha=1.): laplaceM_(laplaceM), chi_(weights.size(),1.), temp_(chi_), w2d(weights), v2d(precon),  alpha_(alpha){ }
    /**
     * @brief apply Maxwell operator
     *
     * same as blas2::symv( gamma, x, y);
     * \f[ y = W  ( \chi + \alpha\Delta) x \f]
     * @param x lhs
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM
     */
    void symv( Vector& x, Vector& y) 
    {
        blas1::pointwiseDot( chi_, x, temp_);
        if( alpha_ != 0);
            blas2::symv( laplaceM_, x, y);
        blas1::axpby( 1., temp_, -alpha_, y);
        blas1::pointwiseDot( w2d, y, y);
    }
    const Precon& weights()const {return w2d;}
    const Precon& precond()const {return v2d;}
    double& alpha( ){  return alpha_;}
    double alpha( ) const  {return alpha_;}
    /**
     * @brief Set chi
     *
     * @return reference to internal chi
     */
    Vector& chi(){return chi_;}
  private:
    const Matrix& laplaceM_;
    Vector chi_, temp_;
    const Precon& w2d, v2d;
    double alpha_;
};


///@}
///@cond
template< class M, class V, class P >
struct MatrixTraits< Maxwell<M, V, P> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class M, class V, class P>
struct MatrixTraits< Helmholtz<M, V, P> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class M, class V, class P >
struct MatrixTraits< const Maxwell<M, V, P> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class M, class V, class P>
struct MatrixTraits< const Helmholtz<M, V, P> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond


} //namespace dg

