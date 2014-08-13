#pragma once

#include <cassert>

#include "backend/matrix_traits.h"
#include "blas.h"
#include "elliptic.h"

#ifdef DG_BENCHMARK
#include "backend/timer.cuh"
#endif

/*!@file
 *
 * Contains Helmholtz and Maxwell operators
 */
namespace dg{

///@addtogroup operators
///@{
/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @ingroup operators
 *
 * Discretization of \f[ (\chi+\alpha\Delta) \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
 * Can be used by the Invert class
 * @tparam Matrix The cusp-matrix class you want to use
 * @tparam Vector The Vector class you want to use
 * @tparam Preconditioner The Preconditioner class you want to use
 * @attention The Laplacian in this formula is positive as opposed to the negative sign in the Elliptic operator
 */
template< class Matrix, class Vector, class Preconditioner> 
struct Helmholtz
{
    /**\
     * @brief Construct Helmholtz operator
     *
     * @tparam Grid The Grid class
     * @param grid The grid to use
     * @param alpha Scalar in the above formula
     * @note The default value of \f$\chi\f$ is one
     */
    template<class Grid>
    Helmholtz( const Grid& g, double alpha = 1.):
        laplaceM_(g, not_normed), 
        temp_(dg::evaluate(dg::one, g)), chi_(temp_),
        alpha_(alpha), isSet(false)
    { }
    /**
     * @brief apply operator
     *
     * Computes
     * \f[ y = W( 1 + \alpha\Delta) x \f] to make the matrix symmetric
     * @param x lhs (is constant up to changes in ghost cells)
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM and thus multiplies by -alpha
     */
    void symv( Vector& x, Vector& y) 
    {
        if( isSet)
            blas1::pointwiseDot( chi_, x, temp_);
        if( alpha_ != 0)
            blas2::symv( laplaceM_, x, y);
        if( isSet)
            blas2::symv( 1., laplaceM_.weights(), temp_, -alpha_, y);
        else
            blas2::symv( 1., laplaceM_.weights(), x, -alpha_, y);

    }
    const Preconditioner& weights()const {return laplaceM_.weights();}
    const Preconditioner& precond()const {return laplaceM_.precond();}
    /**
     * @brief Change alpha
     *
     * @return reference to alpha
     */
    double& alpha( ){  return alpha_;}
    /**
     * @brief Access alpha
     *
     * @return alpha
     */
    double alpha( ) const  {return alpha_;}
    /**
     * @brief Set Chi in the above formula
     *
     * @param chi new Vector
     */
    void set_chi( const Vector& chi) {chi_=chi; isSet =true;}
    /**
     * @brief Sets chi back to one
     */
    void reset_chi(){isSet = false;}
    /**
     * @brief Access chi
     *
     * @return chi
     */
    const Vector& chi(){return chi_;}
  private:
    Elliptic<Matrix, Vector, Preconditioner> laplaceM_;
    Vector temp_, chi_;
    double alpha_;
    bool isSet;
};
///@cond
template< class M, class V, class P>
struct MatrixTraits< Helmholtz<M, V, P> >
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

///**
// * @brief Matrix class that represents a Helmholtz-type operator that appears in the parallel induction equation
// *
// * @ingroup operators
// *
// * Discretization of \f[ (\alpha\Delta + \chi) \f]
// * can be used by the Invert class
// * @tparam Matrix The cusp-matrix class you want to use
// * @tparam container The type of Vector you want to use
// * @tparam Preconditioner The Preconditioner type
// */
//template< class Matrix, class Vector, class Preconditioner>
//struct Maxwell
//{
//    /**
//     * @brief Construct from existing matrices
//     *
//     * Since memory is small on gpus Maxwell can be constructed using an existing laplace operator
//     * @param laplaceM negative normalised laplacian
//     * @param weights weights
//     * @param precon preconditioner
//     * @param alpha The factor alpha
//     * @attention The class takes care of the negative sign of laplacianM, so the alpha given is the alpha in the above formula
//     */
//    template <class Grid>
//    Maxwell( const Grid& g, const Matrix& laplaceM, 
//              const Preconditioner& weights, const Preconditioner& precon, double alpha=1.):         
//        laplaceM_(laplaceM), chi_(dg::evaluate(dg::one, grid)), temp_(chi_), w2d(weights), v2d(precon),  alpha_(alpha){ }
//    Maxwell( Elliptic<Matrix, Vector, Preconditioner>& laplaceM, 
//               const Preconditioner& weights, const Preconditioner& precond, double alpha):
//        p_(weights), q_(precond), laplaceM_(laplaceM), alpha_( alpha){
//    /**
//     * @brief apply Maxwell operator
//     *
//     * same as blas2::symv( gamma, x, y);
//     * \f[ y = W  ( \chi + \alpha\Delta) x \f]
//     * @param x lhs
//     * @param y rhs contains solution
//     * @note Takes care of sign in laplaceM
//     */
//    void symv( Vector& x, Vector& y) 
//    {
//        blas1::pointwiseDot( chi_, x, temp_);
//        if( alpha_ != 0);
//            blas2::symv( laplaceM_, x, y);
//        blas1::axpby( 1., temp_, -alpha_, y);
//        blas1::pointwiseDot( w2d, y, y);
//    }
//    const Preconditioner& weights()const {return p_;}
//    const Preconditioner& precond()const {return q_;}
//    double& alpha( ){  return alpha_;}
//    double alpha( ) const  {return alpha_;}
//    /**
//     * @brief Set chi
//     *
//     * @return reference to internal chi
//     */
//    Vector& chi(){return chi_;}
//  private:
//    const Matrix& laplaceM_;
//    Vector chi_, temp_;
//    const Preconditioner& p_, q_;
//    double alpha_;
//};


///@}
///@cond
//template< class M, class V, class P >
//struct MatrixTraits< Maxwell<M, V, P> >
//{
//    typedef double value_type;
//    typedef SelfMadeMatrixTag matrix_category;
//};
//template< class M, class V, class P >
//struct MatrixTraits< const Maxwell<M, V, P> >
//{
//    typedef double value_type;
//    typedef SelfMadeMatrixTag matrix_category;
//};
///@endcond


} //namespace dg

