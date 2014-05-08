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
/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @ingroup creation
 * Discretization of \f[ (\Delta + g) \f]
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
     * @param p preconditioner ( W2D or T2D); makes the matrix symmetric and is the same you later use in conjugate gradients
     */
    Maxwell( const Matrix& laplaceM, const Vector& weights):p_(weights), laplaceM_(laplaceM){ }
    /**
     * @brief apply operator
     *
     * same as blas2::symv( gamma, x, y);
     * \f[ y = ( 1 + \alpha\Delta) x \f]
     * @tparam Vector The vector class
     * @param x lhs
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM
     */
    void symv( const Vector& x, Vector& y) const
    {
        if( alpha_ != 0);
            blas2::symv( laplaceM_, x, y);
        blas1::axpby( 1., chi_, -1., y);
        blas2::symv( p_, y,  y);
    }
    Vector& chi(){return chi_;}
  private:
    const Prec& p_;
    const Matrix& laplaceM_;
    Vector chi_;
};

///@cond
template< class M, class T>
struct MatrixTraits< Maxwell<M, T> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

//directly solve the Helmholtz equation (might be more practical than gamma)
template<class container>
struct Helmholtz2d
{
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    typedef dg::DMatrix Matrix;
    //typedef typename Matrix::MemorySpace MemorySpace;
    Helmholtz2d( const Grid2d<double>& g, double alpha, double eps): 
        eps_(eps), g(g),
        w2d( dg::create::w2d(g)),v2d( dg::create::v2d(g)), 
        phi1( g.size(), 0), phi2(phi1), cg( w2d, w2d.size())
    {
        set_alpha( alpha);
    }
    unsigned operator()( const container& rho, container& phi)
    {
        assert( &rho != &phi);
        blas1::axpby( 2., phi1, -1.,  phi2, phi);
        dg::blas2::symv( w2d, rho, phi2);
        unsigned number = cg( A_, phi, phi2, v2d, eps_);
        phi1.swap( phi2);
        blas1::axpby( 1., phi, 0, phi1);
        return number;
    }
    void explicit_step( const container& rho, container& phi)
    {
        dg::blas2::symv( A_, rho, phi);
        dg::blas2::symv( v2d, phi, phi);
    }
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
  private:
    double eps_;
    const Grid2d<double> g;
    const container w2d, v2d;
    container phi1, phi2;
    Matrix A_;
    dg::CG< container > cg;
};


} //namespace dg
#endif//_DG_GAMMA_

