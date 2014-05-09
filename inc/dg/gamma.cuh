#pragma once

#include <cassert>

#include "blas.h"
#include "cg.cuh"
#include "grid.cuh"
#include "typedefs.cuh"
#include "weights.cuh"
#include "derivatives.cuh"

#ifdef DG_BENCHMARK
#include "dg/timer.cuh"
#endif

namespace dg{

/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @ingroup utilities
 * Discretization of \f[ (1+\alpha\Delta) \f]
 * can be used by the Invert class
 * @tparam Matrix The cusp-matrix class you want to use
 * @tparam Prec The type of preconditioner you want to use
 */
template< class Matrix,class Vector> 
struct Helmholtz
{
    /**
     * @brief Construct from existing matrices
     *
     * Since memory is small on gpus Helmholtz can be constructed using an existing laplace operator
     * @param laplaceM negative normalised laplacian
     * @param weights ( W2D or T2D); makes the matrix symmetric and is the same you later use in conjugate gradients
     * @param precond ( V2D or S2D); precondtioner you later use in conjugate gradients
     * @param alpha prefactor of laplacian
     */
    Helmholtz( const Matrix& laplaceM, const Vector& weights, const Vector& precond, double alpha):p_(weights), q_(precond), laplaceM_(laplaceM), alpha_( alpha){ }
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
    void symv( const Vector& x, Vector& y) const
    {
        if( alpha_ != 0);
            blas2::symv( laplaceM_, x, y);
        blas1::axpby( 1., x, -alpha_, y);
        blas2::symv( p_, y,  y);
    }
    const Vector& weights()const {return p_;}
    const Vector& precond()const {return q_;}
    double& alpha( ){  return alpha_;}
    double alpha( ) const  {return alpha_;}
  private:
    const Vector& p_, q_;
    const Matrix& laplaceM_;
    double alpha_;
};

/**
 * @brief Matrix class that represents a Helmholtz-type operator that appears in the parallel induction equation
 *
 * @ingroup utilities 
 * Discretization of \f[ (\alpha\Delta + \chi) \f]
 * can be used by the Invert class
 * @tparam Matrix The cusp-matrix class you want to use
 * @tparam container The type of Vector you want to use
 */
template< class Matrix, class Vector>
struct Maxwell
{
    /**
     * @brief Construct from existing matrices
     *
     * Since memory is small on gpus Maxwell can be constructed using an existing laplace operator
     * @param laplaceM negative normalised laplacian
     * @param chi The first chi
     * @param w2d weights
     * @param v2d preconditioner
     * @param alpha The factor alpha
     */
    Maxwell( const Matrix& laplaceM, const Vector& chi, const Vector& w2d, const Vector& v2d,  double alpha=1.): laplaceM_(laplaceM), chi_(chi), w2d(w2d), v2d(v2d),  alpha_(alpha){ }
    /**
     * @brief apply operator
     *
     * same as blas2::symv( gamma, x, y);
     * \f[ y = ( \chi + \alpha\Delta) x \f]
     * @tparam Vector The vector class
     * @param x lhs
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM
     */
    void symv( const Vector& x, Vector& y) const
    {
        Vector temp( chi_.size());
        if( alpha_ != 0);
            blas2::symv( laplaceM_, x, y);
        blas1::pointwiseDot( chi_, x, temp);
        blas1::axpby( 1., temp, -alpha_, y);
        blas1::pointwiseDot( w2d, y, y);
    }
    const Vector& weights()const {return w2d;}
    const Vector& precond()const{return v2d;}
    /**
     * @brief Set chi
     *
     * @return reference to internal chi
     */
    Vector& chi(){return chi_;}
  private:
    const Matrix& laplaceM_;
    const Vector& w2d, v2d;
    Vector chi_;
    double alpha_;
};


/**
 * @brief Package matrix to be used in the Invert class
 *
 * @ingroup utilities
 * @tparam M Matrix class 
 * @tparam V class for weights and Preconditioner
 */
template< class M, class V>
struct ApplyWithWeights
{
    ApplyWithWeights( const M& m, const V& weights, const V& precond):m_(m), w_(weights), p_(precond){}
    void symv( const V& x, V& y) const
    {
        blas2::symv( m_, x, y);
        blas2::symv( p_, y, y);
    }
    const V& weights() const{return w_;}
    const V& precond() const{return p_;}
    private:
    const M& m_;
    const V& w_, p_;
};
/**
 * @brief Package matrix to be used in the Invert class
 *
 * @ingroup utilities
 * @tparam M Matrix class 
 * @tparam V class for weights and Preconditioner
 */
template< class M, class V>
struct ApplyWithoutWeights
{
    ApplyWithoutWeights( const M& m, const V& weights, const V& precond):m_(m), w_(weights), p_(precond){}
    void symv( const V& x, V& y) const { blas2::symv( m_, x, y); }
    const V& weights() const{return w_;}
    const V& precond() const{return p_;}
    private:
    const M& m_;
    const V& w_, p_;
};
///@cond
template< class M, class T>
struct MatrixTraits< Maxwell<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class M, class T>
struct MatrixTraits< Helmholtz<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class M, class T>
struct MatrixTraits< ApplyWithWeights<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class M, class T>
struct MatrixTraits< ApplyWithoutWeights<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

/**
 * @brief Solve a symmetric linear inversion problem using a conjugate gradient method 
 *
 * @ingroup algorithms
 * Solves the Equation \f[ \hat O \phi = \rho \f]
 * for any symmetric operator O. 
 * It uses solutions from the last two calls to 
 * extrapolate a solution for the current call.
 * @tparam container The Vector class to be used
 */
template<class container>
struct Invert
{
    /**
     * @brief Constructor
     *
     * @param copyable Needed to construct the two previous solutions
     * @param max_iter maximum iteration in conjugate gradient
     * @param eps relative error in conjugate gradient
     */
    Invert(const container& copyable,unsigned max_iter, double eps): 
        eps_(eps),
        phi1( copyable.size(), 0.), phi2(phi1), cg( copyable, max_iter) { }
    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = \rho \f]
     * @tparam SymmetricOp Symmetric operator with the SelfMadeMatrixTag
        The functions weights() and precond() need to be callable and return
        weights and the preconditioner for the conjugate gradient method
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     *
     * @return number of iterations used 
     */
    template< class SymmetricOp >
    unsigned operator()( SymmetricOp& op, container& phi, const container& rho)
    {
        assert( &rho != &phi);
        blas1::axpby( 2., phi1, -1.,  phi2, phi);
        dg::blas2::symv( op.weights(), rho, phi2);
#ifdef DG_BENCHMARK
    Timer t;
    t.tic();
#endif //DG_BENCHMARK
        unsigned number = cg( op, phi, phi2, op.precond(), eps_);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations \t"<< number << "\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
        phi1.swap( phi2);
        blas1::axpby( 1., phi, 0, phi1);
        return number;
    }
  private:
    double eps_;
    container phi1, phi2;
    dg::CG< container > cg;
};


} //namespace dg

