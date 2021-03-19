#pragma once
// #undef BOOST_MATH_MAX_SERIES_ITERATION_POLICY
// #define BOOST_MATH_MAX_SERIES_ITERATION_POLICY 1000000000    
#include <boost/math/special_functions.hpp>

#include "blas.h"
#include "lgmres.h"

//! M_PI is non-standard ... so MSVC complains
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/**
* @brief Classes for square root Matrix-Vector product computation via the Cauchy integral 
*/
namespace dg
{
/**
 * @brief Matrix class that represents the operator in the Caucha square root integral formula, in particular the discretization of \f$ (-w^2 I -A) x \f$  where \f$ A\f$ is matrix and w is a scalar and x is a vector.
 * 
 * @ingroup matrixoperators
 *
 */
template< class Matrix, class Container>
struct SqrtCauchyIntOp
{
    public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    SqrtCauchyIntOp() {}
    /**
     * @brief Construct operator \f$ (-w^2 I -A) \f$ in cauchy formula
     *
     * @param A symmetric or non-symmetric Matrix, e.g.: a not_normed Helmholtz operator or a symmetric or non-symmetric tridiagonal matrix
     * @param copyable a copyable container 
     * @param multiply_weights multiply inverse weights in front of matrix A (important if matrix A is not_normed) 
     */
    SqrtCauchyIntOp( const Matrix& A, const Container& copyable, const bool& multiply_weights )
    { 
        construct(A, copyable, multiply_weights);
    }
    /**
     * @brief Construct operator \f$ (-w^2 I -A) \f$ in cauchy formula
     *
     * @param A symmetric or non-symmetric Matrix, e.g.: a not_normed Helmholtz operator or a symmetric or non-symmetric tridiagonal matrix
     * @param copyable a copyable container 
     * @param multiply_weights multiply inverse weights in front of matrix A (important if matrix A is not_normed) 
     */
    void construct(const Matrix& A, const Container& copyable, const bool& multiply_weights)
    {
        m_A = A;
        m_precond = copyable;
        m_multiply_weights = multiply_weights;
        m_size = copyable.size();
        dg::blas1::scal(m_precond,0.);
        dg::blas1::plus(m_precond,1.0);
        m_weights = m_inv_weights = m_precond;
        m_w=0.0;
    }
    /**
     * @brief Resize vectors (weights, inverse weights and preconditioner)
     *
     * @param new_max new size of vectors
     * 
     */    
    void new_size( unsigned new_max) { 
        m_weights.resize(new_max);
        m_inv_weights.resize(new_max);
        m_precond.resize(new_max);
        m_size = new_max;
        dg::blas1::scal(m_precond,0.);
        dg::blas1::plus(m_precond,1.0);
        m_weights = m_inv_weights = m_precond;
    }
    /**
     * @brief Return the weights
     * 
     * @return the  weights
     */
    const Container& weights()const { return m_weights; }
    /**
     * @brief Return the inverse weights
     *
     * @return the inverse weights
     */
    const Container& inv_weights()const { return m_inv_weights; }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     * 
     * @return the preconditioner
     */
    const Container& precond()const { return m_precond; }
    /**
     * @brief Set w in the Cauchy integral
     *
     * @param w w in the Cauchy integral
     */
    void set_w( const value_type& w) { m_w=w; }   
/**
     * @brief Set the Matrix A
     *
     * @param A Matrix
     */
    void set_A( const Matrix& A) {
        m_A = A;
    }  
    /**
     * @brief Set the weights
     *
     * @param weights new weights
     */
    void set_weights( const Container& weights) {
        m_weights = weights;
    }
    /**
     * @brief Set the inverse weights
     *
     * @param inv_weights new inverse weights
     */
    void set_inv_weights( const Container& inv_weights) {
        m_inv_weights = inv_weights;
    }
    /**
     * @brief Set the precond
     *
     * @param precond new precond
     */
    void set_precond( const Container& precond) {
        m_precond = precond;
    }
    /**
     * @brief Compute operator
     *
     * i.e. \f$ y= W (w^2 I +V A)*x \f$ if weights are multiplied or  \f$ y= (w^2 I +A)*x \f$ otherwise
     * @param x left-hand-side
     * @param y result
     */ 
    void symv( const Container& x, Container& y) 
    {
        dg::blas2::symv(m_A, x, y); // A x
        if (m_multiply_weights == true) dg::blas2::symv(m_w*m_w, m_weights, x, 1., y); //W w^2 x + A x 
        else dg::blas1::axpby(m_w*m_w, x, 1., y); // w^2 x + A x 
    } 
  private:
    Container m_weights, m_inv_weights, m_precond;
    Matrix m_A;
    value_type m_w;
    bool m_multiply_weights;
    unsigned m_size;
};


template< class Matrix, class Container>
struct TensorTraits< SqrtCauchyIntOp< Matrix, Container> >
{
    using value_type  = dg::get_value_type<Container>;
    using tensor_category = dg::SelfMadeMatrixTag;
};


/**
 * @brief Compute the square root matrix - vector product via the Cauchy integral \f[ \sqrt{A} x=  \frac{- 2 K' \sqrt{m}}{\pi N} A \sum_{j=1}^{N} (w_j^2 I -A)^{-1} c_j d_j  x \f]
 * A is the matrix, x is the vector, w is a scalar m is the smallest eigenvalue of A, K' is the conjuated complete  elliptic integral and \f$c_j\f$ and \f$d_j\f$ are the jacobi functions 
 * 
 *This class is based on the approach (method 3) of the paper <a href="https://doi.org/10.1137/070700607" > Computing A alpha log(A), and Related Matrix Functions by Contour Integrals </a>  by N. Hale et al
 * 
 * @ingroup matrixfunctionapproximation
 *
 */
template<class Matrix, class Container>
struct SqrtCauchyInt
{
  public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    SqrtCauchyInt() { }
    /**
     * @brief Construct Rhs operator
     *
     * @param A A symmetric or non-symmetric Matrix
     * @param g The grid to use
     * @param eps Accuarcy for CG solve
     * @param multiply_weights multiply inverse weights in front of matrix A 
     * @param symmetric true = symmetric A / false = non-symmetric A
     */
    SqrtCauchyInt( const Matrix& A, const Container& copyable, value_type eps, const bool& multiply_weights, const bool& symmetric)
    {
        construct(A, copyable, eps, multiply_weights, symmetric);
    }
    /**
     * @brief Construct Rhs operator
     *
     * @param A A symmetric or non-symmetric Matrix
     * @param g The grid to use
     * @param eps Accuarcy for CG solve
     * @param multiply_weights multiply inverse weights in front of matrix A 
     * @param symmetric true = symmetric A / false = non-symmetric A
     */
    void construct(const Matrix& A, const Container& copyable, value_type eps, const bool& multiply_weights, const bool& symmetric) 
    {
        m_helper = m_temp = m_helper3 = copyable;
        m_A = A;
        m_multiply_weights = multiply_weights;
        m_symmetric = symmetric;
        m_eps = eps;
        m_size = m_helper.size();
        m_number = 0;
        m_op.construct(m_A, m_helper, m_multiply_weights);
        if (m_symmetric == true) m_pcg.construct( m_helper, m_size*m_size+1);
        else m_lgmres.construct( m_helper, 300, 100, 10*m_size*m_size);
        m_temp_ex.set_max(1, copyable);
    }
    /**
     * @brief Resize matrix and set A and vectors and set new size
     *
     * @param new_max new size
     */
     void new_size( unsigned new_max) { 
        m_helper.resize(new_max);
        m_temp.resize(new_max);
        m_helper3.resize(new_max);
        if (m_symmetric == true)  m_pcg.construct( m_helper, new_max*new_max+1);
        else m_lgmres.construct( m_helper, 300, 100, 10*new_max*new_max);
        m_op.new_size(new_max);
        m_temp_ex.set_max(1, m_temp);
        m_size = new_max;
    } 
    ///@brief Get the current size of vectors
    ///@return the current vector size
    unsigned get_size() const {return m_size;}
    /**
     * @brief Set the Matrix A
     *
     * @param A Matrix
     */
     void set_A( const Matrix& A) { 
         m_A = A; 
         m_op.set_A(A);
    } 
    /**
     * @brief Set the weights
     *
     * @param weights weights
     */
    void set_weights( const Container& weights) {
        m_op.set_weights(weights);
    }
    /**
     * @brief Set the inverse weights
     *
     * @param inv_weights inverse weights
     */
    void set_inv_weights( const Container& inv_weights) {
        m_op.set_inv_weights(inv_weights);
    }
    /**
     * @brief Set the preconditioner
     *
     * @param precond preconditioner
     */
    void set_precond( const Container& precond) {
        m_op.set_precond(precond);
    }
    /**
     * @brief Compute cauchy integral (including inversion) 
     *
     * i.e. \f[ b=  \frac{- 2 K' \sqrt{m}}{\pi N} V A \sum_{j=1}^{N} (w_j^2 I -V A)^{-1} c_j d_j  x \f]
     * @param y is \f$ y\f$
     * @param b is \f$ b\approx \sqrt{V A} x\f$
     * @note The Jacobi elliptic functions are related to the Mathematica functions via jacobi_cn(k,u ) = JacobiCN_(u,k^2), ... and the complete elliptic integral of the first kind via comp_ellint_1(k) = EllipticK(k^2) 
     */
    void operator()(const Container& x, Container& b, const value_type& minEV, const value_type& maxEV, const unsigned& iter)
    {
        dg::blas1::scal(m_helper3, 0.0);
        value_type s=0.;
        value_type c=0.;
        value_type d=0.;
        value_type w=0.;
        value_type t=0.;
        value_type sqrtminEV = sqrt(minEV);
        const value_type k2 = minEV/maxEV;
        const value_type sqrt1mk2 = sqrt(1.-k2);
        const value_type Ks=std::comp_ellint_1(sqrt1mk2 );
        const value_type fac = 2.* Ks*sqrtminEV/(M_PI*iter);
        for (unsigned j=1; j<iter+1; j++)
        {
            t  = (j-0.5)*Ks/iter; //imaginary part .. 1i missing
            c = 1./boost::math::jacobi_cn(sqrt1mk2, t); 
            s = boost::math::jacobi_sn(sqrt1mk2, t)*c;
            d = boost::math::jacobi_dn(sqrt1mk2, t)*c;
            w = sqrtminEV*s;
            if (m_multiply_weights == true) 
                dg::blas2::symv(c*d, m_op.weights(), x, 0.0 , m_helper); //m_helper = c d x
            else 
                dg::blas1::axpby(c*d, x, 0.0 , m_helper); //m_helper = c d x
            m_op.set_w(w);
            m_temp_ex.extrapolate(t, m_temp);

            if (m_symmetric == true) 
            {
                m_number = m_pcg( m_op, m_temp, m_helper, m_op.inv_weights(), m_op.weights(), m_eps); // m_temp = (w^2 +V A)^(-1) c d x
                if(  m_number == m_pcg.get_max()) throw dg::Fail( m_eps);
            }
            else 
                m_lgmres.solve( m_op, m_temp, m_helper, m_op.inv_weights(), m_op.weights(), m_eps, 1); 
            m_temp_ex.update(t, m_temp);

            dg::blas1::axpby(fac, m_temp, 1.0, m_helper3); // m_helper3 += -fac  (w^2 +V A)^(-1) c d x
        }
        dg::blas2::symv(m_A, m_helper3, b); // - A fac sum (w^2 +V A)^(-1) c d x
        if (m_multiply_weights == true) dg::blas1::pointwiseDot(m_op.inv_weights(),  b, b);  // fac V A (-w^2 I -V A)^(-1) c d x

    }
  private:
    Container m_helper, m_temp, m_helper3;
    Matrix m_A;
    SqrtCauchyIntOp< Matrix, Container> m_op;
    unsigned m_size, m_number;
    bool m_multiply_weights, m_symmetric;
    value_type m_eps;
    dg::CG<Container> m_pcg;
    dg::LGMRES<Container> m_lgmres;
    dg::Extrapolation<Container> m_temp_ex;
};

} //namespace dg
