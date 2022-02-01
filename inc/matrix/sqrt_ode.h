#pragma once

#include "dg/algorithm.h"

/**
* @brief Classes for Matrix function-Vector product computation via the ODE method
*/
namespace dg
{
/**
 * @brief Matrix class that represents the lhs operator of the square root ODE
 *
 * @ingroup matrixoperators
 *
 * discretization of \f$ ((t-1) I -t A)x \f$
 * where \f$ A\f$ is matrix and t is the time and x is a vector.
 */
template< class Matrix, class Container>
struct SqrtODEOp
{
    public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    SqrtODEOp() {}
    /**
     * @brief Construct SqrtODE operator
     *
     * @param A symmetric or non-symmetric Matrix, e.g.: a not_normed Helmholtz operator or a symmetric or non-symmetric tridiagonal matrix
     * @param copyable a copyable container 
     * @param multiply_weights multiply (inverse) weights in front of matrix A and vectors (important if matrix A is not_normed) 
     */
    SqrtODEOp( const Matrix& A, const Container& copyable, const bool& multiply_weights)
    { 
        construct(A, copyable, multiply_weights);
    }
    /**
     * @brief Construct SqrtODEOp operator
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
     * @brief Compute square root ode operator and store in output
     *
     * i.e. \f$ y= W ((t-1) I -t V A)*x \f$ if weights are multiplied or \f$ y=  ((t-1) I -t  A)*x \f$ otherwise
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


template<  class Matrix, class Container>
struct TensorTraits< SqrtODEOp< Matrix, Container> >
{
    using value_type  = dg::get_value_type<Container>;
    using tensor_category = dg::SelfMadeMatrixTag;
};

/**
 * @brief Right hand side of the square root ODE \f[ \dot{y}= \left[(t-1) I -t A\right]^{-1} (I - A)/2  y \f]
 * where \f$ A\f$ is the matrix
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * This class is based on the approach of the paper <a href="https://doi.org/10.1016/S0024-3795(00)00068-9" > Numerical approximation of the product of the square root of a matrix with a vector</a> by E. J. Allen et al
 * 
 * @note Solution of ODE: \f$ y(1) = \sqrt{A} y(0)\f$
 */
template< class Matrix, class Container>
struct SqrtODE
{
  public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    SqrtODE() {};

    /**
     * @brief Construct SqrtOde operator
     *
     * @param A symmetric matrix
     * @param copyable copyable container
     * @param eps Accuarcy for CG solve
     * @param multiply_weights multiply inverse weights in front of matrix A 
     * @param symmetric true = symmetric A / false = non-symmetric A
     */
    SqrtODE( const Matrix& A,  const Container& copyable,  value_type eps, const bool& multiply_weights, const bool& symmetric)
    {
        construct(A, copyable, eps, multiply_weights, symmetric);
    }
    /**
     * @brief Construct SqrtOde operator
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
        m_number = 0;
        m_lhs.construct(m_A, m_helper, multiply_weights);
        if (m_symmetric == true) m_pcg.construct( m_helper, m_size*m_size+1);
        else m_lgmres.construct( m_helper, 30, 10, 100*m_size*m_size);
        m_yp_ex.set_max(3, copyable);
    }
    /**
     * @brief Resize matrix and set A and vectors and set new size
     *
     * @param new_max new size
     */
     void new_size( unsigned new_max) { 
        m_helper.resize(new_max);
        if (m_symmetric == true)  m_pcg.construct( m_helper, new_max*new_max+1);
        else m_lgmres.construct( m_helper, 30, 10, 100*new_max*new_max);
        m_lhs.new_size(new_max);
        m_yp_ex.set_max(3, m_helper);
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
     * i.e. \f[ yp= ((t-1) I -t V A)^{-1} (I - V A)/2  y \f] if weights are multiplied or 
     * \f$ yp= ((t-1) I -t  A)^{-1} (I -  A)/2 * y \f$ otherwise
     * @param t  is time
     * @param y  is \f$ y\f$
     * @param yp is \f$ \dot{y}\f$
     * @note Solution of ODE: \f$ y(1) = \sqrt{V A} y(0)\f$ if weights are multiplied or  \f$ y(1) = \sqrt{A} y(0)\f$ otherwise
     */
    void operator()(value_type t, const Container& y, Container& yp)
    {
        dg::blas2::symv(m_A, y, m_helper);  
        if (m_multiply_weights == true) 
            dg::blas2::symv(0.5, m_lhs.weights(), y, -0.5, m_helper); 
        else
            dg::blas1::axpby(0.5, y, -0.5, m_helper); 
        
        m_lhs.set_time(t);
        m_yp_ex.extrapolate(t, yp);
        if (m_symmetric == true) 
        {
            m_number = m_pcg( m_lhs, yp, m_helper, m_lhs.inv_weights(), m_lhs.weights(), m_eps); 
            if( m_number == m_pcg.get_max()) throw dg::Fail( m_eps);
        }
        else 
            m_lgmres.solve( m_lhs, yp, m_helper, m_lhs.inv_weights(), m_lhs.weights(), m_eps, 1); 
        
        m_yp_ex.update(t, yp);
    }
  private:
    Container m_helper;
    Matrix m_A;
    SqrtODEOp<Matrix, Container> m_lhs;
    unsigned m_size, m_number;
    bool m_multiply_weights, m_symmetric;
    value_type m_eps;
    dg::CG<Container> m_pcg;  
    dg::LGMRES<Container> m_lgmres;
    dg::Extrapolation<Container> m_yp_ex;
};


/**
 * @brief Right hand side of the exponential ODE \f[ \dot{y}= A y \f]
 * where \f$ A\f$ is the matrix
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note Solution of ODE: \f$ y(1) = \exp{A} y(0)\f$
 */
template< class Matrix, class Container>
struct ExpODE
{
  public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ExpODE() {};

    /**
     * @brief Construct ExpOde 
     *
     * @param A matrix
     */
    ExpODE( const Matrix& A)
    {
        construct(A);
    }
    /**
     * @brief Construct SqrtOde operator
     *
     * @param A matrix
     */
    void construct(const Matrix& A) 
    {
        m_A = A;
    }
    /**
     * @brief Set matrix A
     *
     * @param A matrix
     */
    void set_A( const Matrix& A) { 
         m_A = A; 
    }  
    /**
     * @brief Compute rhs term  \f$ yp= A y \f$ 
     * @param t  is time
     * @param y  is \f$ y\f$
     * @param yp is \f[ \dot{y} \f]
     * @note Solution of ODE: \f$ y(1) = \exp{A} y(0)\f$ otherwise
     */
    void operator()(value_type t, const Container& y, Container& yp)
    {
        dg::blas2::symv(m_A, y, yp);
    }
  private:
    Matrix m_A;
};



/**
 * @brief Right hand side of the (zeroth order) modified Bessel function ODE, rewritten as a system of coupled first order ODEs:
 * \f[ \dot{z_0}= z_1 \f]
 * \f[ \dot{z_0}= A^2 z_0 - t^{-1} z_1 \f]
 * where \f$ A\f$ is the matrix and \f[z=(y,\dot{y})\f]
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note Solution of ODE: \f$ y(1) = I_0(A) y(0)\f$
 */
template< class Matrix, class Container>
struct BesselI0ODE
{
  public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    BesselI0ODE() {};

    /**
     * @brief Construct ExpOde 
     *
     * @param A matrix
     */
    BesselI0ODE( const Matrix& A)
    {
        construct(A);
    }
    /**
     * @brief Construct BesselI0 ode
     *
     * @param A matrix
     */
    void construct(const Matrix& A) 
    {
        m_A = A;
    }
    /**
     * @brief Set matrix A
     *
     * @param A matrix
     */
    void set_A( const Matrix& A) { 
         m_A = A; 
    }  
    /**
     * @brief Compute rhs term  
     * \f[ \dot{z_0}= z_1 \f]
     * \f[ \dot{z_1}= A^2 z_0 - t^{-1} z_1 \f] 
     * @param t  is time
     * @param z  is \f[ z = (y, \dot{y}) \f]
     * @param zp is \f[ \dot{z} \f]
     * @note Solution of ODE: \f$ y(1) = \exp{A} y(0)\f$ for initial condition \f$ z(0) = (y(0),0)^T \f$
     */
    void operator()(value_type t, const std::array<Container,2>& z, std::array<Container,2>& zp)
    {
        dg::blas2::symv(m_A, z[0], zp[0]);
        dg::blas2::symv(m_A, zp[0], zp[1]);
        dg::blas1::axpby(-1./t, z[1], 1.0, zp[1]);
        
        dg::blas1::copy(z[0],zp[0]);

    }
  private:
    Matrix m_A;
};



} //namespace dg
