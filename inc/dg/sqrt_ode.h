#pragma once

#include "blas.h"
#include "cg.h"
#include "lgmres.h"
/**
 * @brief Matrix class that represents the lhs operator of the square root ODE
 *
 * @ingroup matrixoperators
 *
 * discretization of \f[ ((t-1) I -t A)*x \f]
 * where \f[ A\f] is matrix and t is the time and x is a vector.
 */
template< class Matrix, class Container>
struct Lhs
{
    public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Lhs() {}
    /**
     * @brief Construct Lhs operator
     *
     * @param A symmetric or non-symmetric Matrix, e.g.: a not_normed Helmholtz operator or a symmetric or non-symmetric tridiagonal matrix
     * @param copyable a copyable container 
     * @param multiply_weights multiply (inverse) weights in front of matrix A and vectors (important if matrix A is not_normed) 
     */
    Lhs( const Matrix& A, const Container& copyable, const bool& multiply_weights)
    { 
        construct(A, copyable, multiply_weights);
    }
    /**
     * @brief Construct Lhs operator
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
        t = 0.0;
        m_size = copyable.size();
        dg::blas1::scal(m_precond,0.);
        dg::blas1::plus(m_precond,1.0);
        m_weights = m_inv_weights = m_precond;
    }  
    
    /**
     * @brief Resize matrix T and vectors weights
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
    ///@brief Get the current size of vectors
    ///@return the current vector size
    unsigned get_size() const {return m_size;}
    /**
     * @brief Return the weights of the Helmholtz operator
     * 
     * @return the  weights of the Helmholtz operator
     */
    const Container& weights()const { return m_weights; }
    /**
     * @brief Return the inverse weights of the Helmholtz operator
     *
     * @return the inverse weights of the Helmholtz operator
     */
    const Container& inv_weights()const { return m_inv_weights; }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     * 
     * @return the preconditioner of the Helmholtz operator
     */
    const Container& precond()const { return m_precond; }
    /**
     * @brief Set the time t
     *
     * @param time the time
     */
    void set_time( const value_type& time) { t=time; }   
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
     * @brief Compute Lhs term and store in output
     *
     * i.e. \f[ y= W ((t-1) I -t V A)*x \f] if weights are multiplied or \f[ y=  ((t-1) I -t  A)*x \f] otherwise
     * @param x left-hand-side
     * @param y result
     */ 
    void symv( const Container& x, Container& y) 
    {
        dg::blas2::symv(m_A, x, y); 
        if (m_multiply_weights == true) dg::blas2::symv((t-1.), m_weights, x, -t, y); 
        else  dg::blas1::axpby((t-1.), x, -t, y); 
    } 
  private:
    Container m_weights, m_inv_weights, m_precond;
    Matrix m_A;
    value_type t;
    bool m_multiply_weights;
    unsigned m_size;
};


/**
 * @brief Rhs of the square root ODE \f[ \dot{y}= \left[(t-1) I -t A\right]^{-1} *(I - A)/2 * y \f]
 *
 * where \f[ A\f] is the matrix
 * @note Solution of ODE: \f[ y(1) = \sqrt{A} y(0)\f]
 */
template< class Matrix, class Container>
struct Rhs
{
  public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    Rhs() {};

    /**
     * @brief Construct Rhs operator
     *
     * @param A symmetric matrix
     * @param copyable copyable container
     * @param eps Accuarcy for CG solve
     * @param multiply_weights multiply inverse weights in front of matrix A 
     * @param symmetric true = symmetric A / false = non-symmetric A
     */
    Rhs( const Matrix& A,  const Container& copyable,  value_type eps, const bool& multiply_weights, const bool& symmetric)
    {
        construct(A, copyable, eps, multiply_weights, symmetric);
    }
    /**
     * @brief Construct Rhs operator
     *
     * @param A symmetric matrix
     * @param copyable copyable container
     * @param eps Accuarcy for CG solve
     * @param multiply_weights multiply inverse weights in front of matrix A 
     * @param symmetric true = symmetric A / false = non-symmetric A
     */
    void construct(const Matrix& A,  const Container& copyable,  value_type eps, const bool& multiply_weights, const bool& symmetric) 
    {
         m_helper = copyable;
         m_A = A;
         m_multiply_weights = multiply_weights;
         m_symmetric = symmetric;
         m_eps = eps;
         m_size = m_helper.size();
         m_lhs.construct(m_A, m_helper, multiply_weights);
         if (m_symmetric == true) 
         {
             if (m_multiply_weights==true) m_invert.construct( m_helper, m_size*m_size, eps, 1, true, 1.);
             else m_invert.construct( m_helper, m_size*m_size, eps, 1, false, 1.);
         }
         else m_lgmres.construct( m_helper, 30, 10, 100);
    }
    /**
     * @brief Resize matrix and set A and vectors and set new size
     *
     * @param new_max new size
     */
     void new_size( unsigned new_max) { 
        m_helper.resize(new_max);
        if (m_symmetric == true) m_invert.set_size(m_helper, new_max*new_max);
        else m_lgmres.construct( m_helper, 30, 10, 100);
        m_lhs.new_size(new_max);
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
         m_lhs.set_A(A);
    }  
    /**
     * @brief Set the weights
     *
     * @param weights weights
     */
    void set_weights( const Container& weights) {
        m_lhs.set_weights(weights);
    }
    /**
     * @brief Set the inverse weights
     *
     * @param inv_weights inverse weights
     */
    void set_inv_weights( const Container& inv_weights) {
        m_lhs.set_inv_weights(inv_weights);
    }
    /**
     * @brief Set the preconditioner
     *
     * @param precond preconditioner
     */
    void set_precond( const Container& precond) {
        m_lhs.set_precond(precond);
    }
    /**
     * @brief Compute rhs term (including inversion of lhs via cg or lgmres) 
     *
     * i.e. \f[ yp= ((t-1) I -t V A)^{-1} *(I - V A)/2 * y \f] if weights are multiplied or 
     * \f[ yp= ((t-1) I -t  A)^{-1} *(I -  A)/2 * y \f] otherwise
     * @param y  is \f[ y\f]
     * @param yp is \f[ \dot{y}\f]
     * @note Solution of ODE: \f[ y(1) = \sqrt{V A} y(0)\f] if weights are multiplied or  \f[ y(1) = \sqrt{A} y(0)\f] otherwise
     */
    void operator()(value_type t, const Container& y, Container& yp)
    {
        dg::blas2::symv(m_A, y, m_helper);  
        if (m_multiply_weights == true) dg::blas1::pointwiseDot(m_lhs.inv_weights(), m_helper, m_helper); 
        dg::blas1::axpby(0.5, y, -0.5, m_helper); 
        m_lhs.set_time(t);
        if (m_symmetric == true) m_invert( m_lhs, yp, m_helper); 
        else m_lgmres.solve( m_lhs, yp, m_helper, m_lhs.inv_weights(), m_lhs.inv_weights(), m_eps, 1); 
    }
  private:
    Container m_helper;
    Matrix m_A;
    Lhs<Matrix, Container> m_lhs;
    unsigned m_size;
    bool m_multiply_weights, m_symmetric;
    value_type m_eps;
    dg::Invert<Container> m_invert;  
    dg::LGMRES<Container> m_lgmres;

};


template<  class Matrix, class Container>
struct dg::TensorTraits< Lhs< Matrix, Container> >
{
    using value_type  = dg::get_value_type<Container>;
    using tensor_category = dg::SelfMadeMatrixTag;
};

