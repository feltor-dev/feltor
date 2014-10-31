#pragma once

#include <cassert>

#include "backend/matrix_traits.h"
#include "blas.h"
#include "elliptic.h"

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
 * Unnormed discretization of \f[ (\chi+\alpha\Delta) \f]
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
     * @param dir Direction of the Laplace operator
     * @note The default value of \f$\chi\f$ is one
     */
    template<class Grid>
    Helmholtz( const Grid& g, double alpha = 1., direction dir = dg::forward):
        laplaceM_(g, dg::DIR,dg::DIR, not_normed, dir), 
        temp_(dg::evaluate(dg::one, g)), chi_(temp_),
        alpha_(alpha), isSet(false)
    { }
    /**\
     * @brief Construct Helmholtz operator
     *
     * @tparam Grid The Grid class
     * @param grid The grid to use
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param alpha Scalar in the above formula
     * @param dir Direction of the Laplace operator
     * @note The default value of \f$\chi\f$ is one
     */
    template<class Grid>
    Helmholtz( const Grid& g, bc bcx, bc bcy, double alpha = 1., direction dir = dg::forward):
        laplaceM_(g, bcx,bcy,not_normed, dir), 
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
    /**
     * @brief These are the weights that made the operator symmetric
     *
     * @return weights
     */
    const Preconditioner& weights()const {return laplaceM_.weights();}
    /**
     * @brief Preconditioner to use in conjugate gradient solvers
     *
     * multiply result by these coefficients to get the normed result
     * @return Preconditioner
     */
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


} //namespace dg

