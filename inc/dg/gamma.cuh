#ifndef _DG_GAMMA_
#define _DG_GAMMA_


#include <cassert>

#include "blas.h"
#include "cg.cuh"
#include "grid.cuh"
#include "typedefs.cuh"
#include "weights.cuh"
#include "derivatives.cuh"

namespace dg{

/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @ingroup creation
 * Discretization of \f[ (1+\alpha\Delta) \f]
 * can be used in conjugate gradient
 * @tparam Matrix The cusp-matrix class you want to use
 * @tparam Prec The type of preconditioner you want to use
 */
template< class Matrix,class Vector> 
struct GammaInv
{
    /**
     * @brief Construct from existing matrices
     *
     * Since memory is small on gpus GammaInv can be constructed using an existing laplace operator
     * @param laplaceM negative normalised laplacian
     * @param w2d weights ( W2D or T2D); makes the matrix symmetric and is the same you later use in conjugate gradients
     * @param v2d preconditioner ( V2D or S2D); precondtioner you later use in conjugate gradients
     * @param alpha prefactor of laplacian
     */
    GammaInv( const Matrix& laplaceM, const Vector& w2d, const Vector& v2d, double alpha):p_(w2d), q_(v2d), laplaceM_(laplaceM), alpha_( alpha){ }
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
    const Vector& weights(){return p_;}
    const Vector& precond(){return q_;}
    double& alpha( ){  return alpha_;}
    double alpha( ) const  {return alpha_;}
  private:
    const Vector& p_, q_;
    const Matrix& laplaceM_;
    double alpha_;
};

///@cond
template< class M, class T>
struct MatrixTraits< GammaInv<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond
/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @ingroup creation
 * Discretization of \f[ (\alpha\Delta + \chi) \f]
 * can be used in conjugate gradient
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
    const Vector& weights(){return w2d;}
    const Vector& precond(){return v2d;}
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

///@cond
template< class M, class T>
struct MatrixTraits< Maxwell<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

/**
 * @brief Solve a symmetric linear inversion problem using a conjugate gradient method 
 *
 * @tparam container The Vector class to be used
 */
template<class container>
struct Helmholtz2d
{
    Helmholtz2d(const container& copyable,unsigned max_iter, double eps): 
        eps_(eps),
        phi1( copyable.size(), 0.), phi2(phi1), cg( copyable, max_iter)
    {
    }
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
        unsigned number = cg( op, phi, phi2, op.precond(), eps_);
        phi1.swap( phi2);
        blas1::axpby( 1., phi, 0, phi1);
        return number;
    }
  private:
    double eps_;
    container phi1, phi2;
    dg::CG< container > cg;
};
    /*
    void set_alpha( double alpha_new) 
    {
        cusp::coo_matrix<int, double, cusp::host_memory> A = create::laplacianM( g, not_normed), diff;
        cusp::coo_matrix<int, double, cusp::host_memory> weights( g.size(), g.size(), g.size());
        thrust::sequence(weights.row_indices.begin(), weights.row_indices.end()); 
        thrust::sequence(weights.column_indices.begin(), weights.column_indices.end()); 
        thrust::copy( w2d.begin(), w2d.end(), weights.values.begin());
        cusp::blas::scal( A.values, -alpha_new);
        cusp::add( weights, A, diff);
        A_  = diff;
    }
    */


} //namespace dg
#endif//_DG_GAMMA_

