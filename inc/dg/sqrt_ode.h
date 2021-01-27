#pragma once

#include "blas.h"
#include "helmholtz.h"
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

// #include "bicgstabl.h"
// /**
//  * @brief Matrix class that represents the lhs operator of the square root ODE
//  *
//  * @ingroup matrixoperators
//  *
//  * Unnormed discretization of \f[ ((t-1) I -t V A)*x \f]
//  * where \f[ A\f] is the (not normed) helmholtz operator and t is the time.
//  */
// template< class Geometry, class Matrix, class Container>
// struct Lhs
// {
//     public:
//     using geometry_type = Geometry;
//     using matrix_type = Matrix;
//     using container_type = Container;
//     using value_type = dg::get_value_type<Container>;
//     ///@brief empty object ( no memory allocation)
//     Lhs() {}
//     /**
//      * @brief Construct Lhs operator
//      *
//      * @param A Helmholtz operator (not normed and symmetric)
//      * @param g The grid to use
//      */
//     Lhs( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g):   
//         m_helper( dg::( dg::zero, g)),
//         m_A(A)
//     { 
//         dg::assign( dg::create::inv_volume(g),    m_inv_weights);
//         dg::assign( dg::create::volume(g),        m_weights);
//         dg::assign( dg::create::inv_weights(g),   m_precond);
//         t=0.0;
//     }
//     /**
//      * @brief Return the weights of the Helmholtz operator
//      * 
//      * @return the  weights of the Helmholtz operator
//      */
//     const Container& weights()const { return m_weights; }
//     /**
//      * @brief Return the inverse weights of the Helmholtz operator
//      *
//      * @return the inverse weights of the Helmholtz operator
//      */
//     const Container& inv_weights()const { return m_inv_weights; }
//     /**
//      * @brief Return the default preconditioner to use in conjugate gradient
//      * 
//      * @return the preconditioner of the Helmholtz operator
//      */
//     const Container& precond()const { return m_precond; }
//     /**
//      * @brief Set the time t
//      *
//      * @param time the time
//      */
//     void set_time( const value_type& time) { t=time; }   
//     /**
//      * @brief Compute Lhs term and store in output
//      *
//      * i.e. \f[ y= W ((t-1) I -t V A)*x \f]
//      * @param x left-hand-side
//      * @param y result
//      */ 
//     void symv( const Container& x, Container& y) 
//     {
// 
//         dg::blas2::symv(m_A, x, m_helper); //m_helper =  A x
//         dg::blas1::pointwiseDot(m_inv_weights, m_helper, y); //m_helper = V A x ...normed 
//         dg::blas1::axpby((t-1.), x, -t, y); //m_helper = (t-1)x - t V A x 
//         dg::blas1::pointwiseDot(m_weights, y, y); //make  not normed
//     } 
//   private:
//     Container m_helper;
//     Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;
//     dg::Helmholtz<Geometry,  Matrix, Container> m_A;
//     value_type t;
// };
// 
// 
// /**
//  * @brief Rhs of the square root ODE \f[ \dot{y}= \left[(t-1) I -t V A\right]^{-1} *(I - V A)/2 * y \f]
//  *
//  * where \f[ A\f] is the not normed Helmholtz operator and t is the time
//  * @note Solution of ODE: \f[ y(1) = \sqrt{V A} y(0)\f]
//  */
// template<class Geometry, class Matrix, class Container>
// struct Rhs
// {
//   public:
//     using geometry_type = Geometry;
//     using matrix_type = Matrix;
//     using container_type = Container;
//     using value_type = dg::get_value_type<Container>;
//     /**
//      * @brief Construct Rhs operator
//      *
//      * @param A Helmholtz operator (not normed and symmetric)
//      * @param g The grid to use
//      * @param eps Accuarcy for CG solve
//      */
//     Rhs( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type eps):
//          m_helper( dg::evaluate( dg::zero, g)),
//          m_A(A),
//          m_lhs(m_A, g),
//          m_invert( m_helper, g.size(), eps, 1)
//     {}
//     /**
//      * @brief Compute rhs term (including inversion of lhs) 
//      *
//      * i.e. \f[ yp= ((t-1) I -t V A)^{-1} *(I - V A)/2 * y \f]
//      * @param y  is \f[ y\f]
//      * @param yp is \f[ \dot{y}\f]
//      * @note Solution of ODE: \f[ y(1) = \sqrt{V A} y(0)\f]
//      */
//     void operator()(value_type t, const Container& y, Container& yp)
//     {
//         dg::blas2::symv(m_A, y, m_helper);  //m_helper = A y
//         dg::blas1::pointwiseDot(m_lhs.inv_weights(), m_helper, m_helper); //make normed
//         dg::blas1::axpby(0.5, y, -0.5, m_helper); //m_helper = 1/2 y - 1/2 V A y //is normed
//         m_lhs.set_time(t);
//         m_invert( m_lhs, yp, m_helper); // ( (t-1)  - V A t ) yp = (1-V A)/2 y
//     }
//   private:
//     Container m_helper;
//     dg::Helmholtz<Geometry,  Matrix, Container> m_A;
//     Lhs<Geometry, Matrix, Container> m_lhs;
//     dg::Invert<Container> m_invert;
// };
// 
// 
// 
// template< class Matrix, class Container>
// struct LhsT
// {
//     public:
//     using matrix_type = Matrix;
//     using container_type = Container;
//     using value_type = dg::get_value_type<Container>;
//     ///@brief empty object ( no memory allocation)
//     LhsT() {}
//     /**
//      * @brief Construct Lhs operator
//      *
//      * @param T Symmetric tridiagonal matrix
//      * @param copyable to get the size of the vectors
//      */
//     LhsT( const Matrix& T, const Container& copyable)   
//     { 
//         construct(T, copyable);
//     }
//     void construct(const Matrix& T, const Container& copyable) 
//     {
//         m_A =T;
//         m_one = copyable;
//         t = 0.0;
//         m_size = copyable.size();
//         dg::blas1::scal(m_one,0.);
//         dg::blas1::plus(m_one,1.0);
//     }
//     /**
//      * @brief Resize matrix T and vectors weights
//      *
//      * @param new_max new size of vectors
//      * 
//      */    
//     void new_size( unsigned new_max) { 
//         m_one.resize(new_max);
//         m_size = new_max;
//         dg::blas1::scal(m_one,0.);
//         dg::blas1::plus(m_one,1.0);
//     }
//     ///@brief Get the current size of vectors
//     ///@return the current vector size
//     unsigned get_size() const {return m_size;}
//     /**
//      * @brief Return the weights
//      * 
//      * @return the  weights
//      */
//     const Container& weights()const { return m_one; }
//     /**
//      * @brief Return the inverse weights
//      *
//      * @return the inverse weights 
//      */
//     const Container& inv_weights()const { return m_one; }
//     /**
//      * @brief Return the default preconditioner to use in conjugate gradient
//      * 
//      * @return the preconditioner 
//      */
//     const Container& precond()const { return m_one; }
//     /**
//      * @brief Set the Matrix T
//      *
//      * @param T Matrix
//      */
//     void set_T( const Matrix& T) {
//         m_A=T;
//     }  
//     /**
//      * @brief Set the time t
//      *
//      * @param time the time
//      */
//     void set_time( const value_type& time) { t=time; }   
//     /**
//      * @brief Compute Lhs term and store in output
//      *
//      * i.e. \f[ y=  ((t-1) I -t  T)*x \f]
//      * @param x left-hand-side
//      * @param y result
//      */ 
//     void symv( const Container& x, Container& y) 
//     {
//         dg::blas2::symv(m_A, x, y); //y =  A x
//         dg::blas1::axpby((t-1.), x, -t, y); //y = (t-1)x - t  A x 
//     } 
//   private:
//     Matrix m_A;
//     Container m_one;
//     value_type t;
//     unsigned m_size;
// };
// /**
//  * @brief Rhs of the square root ODE \f[ \dot{y}= \left[(t-1) I -t T\right]^{-1} *(I - A)/2 * y \f]
//  *
//  * where \f[ T\f] is a symmetric tridiagonal matrix
//  * @note Solution of ODE: \f[ y(1) = \sqrt{T} y(0)\f]
//  * @note Todo: Compute inverse analytically
//  */
// template<class Matrix, class Container>
// struct RhsT
// {
//   public:
//     using matrix_type = Matrix;
//     using container_type = Container;
//     using value_type = dg::get_value_type<Container>;
//     RhsT() {}
//     /**
//      * @brief Construct Rhs operator
//      *
//      * @param T symmetric matrix
//      * @param copyable copyable container
//      * @param eps Accuarcy for CG solve
//      */
//     RhsT( const Matrix& T,  const Container& copyable,  value_type eps)
//     {
//         construct(T, copyable, eps);
//     }
//     void construct(const Matrix& T,  const Container& copyable,  value_type eps) 
//     {
//          m_helper = copyable;
//          m_A = T;
//          m_lhs.construct(m_A, copyable);
//          m_size = copyable.size();
//          m_invert.construct( m_helper, m_size*m_size, eps, 1, false, 1.); //weights not multiplied on rhs
//     }
//     /**
//      * @brief Resize matrix and set T and vectors and set new size
//      *
//      * @param T Matrix
//      */
//      void new_size( unsigned new_max) { 
//         m_helper.resize(new_max);
//         m_invert.set_size(m_helper, new_max*new_max);
//         m_lhs.new_size(new_max);
//         m_size = new_max;
//     } 
//     ///@brief Get the current size of vectors
//     ///@return the current vector size
//     unsigned get_size() const {return m_size;}
//     /**
//      * @brief Set the Matrix T
//      *
//      * @param T Matrix
//      */
//      void set_T( const Matrix& T) { 
//          m_A=T; 
//          m_lhs.set_T(T);
//     }  
//     /**
//      * @brief Compute rhs term (including inversion of lhs) 
//      *
//      * i.e. \f[ yp= ((t-1) I -t T)^{-1} *(I - T)/2 * y \f]
//      * @param y  is \f[ y\f]
//      * @param yp is \f[ \dot{y}\f]
//      * @note Solution of ODE: \f[ y(1) = \sqrt{T} y(0)\f]
//      */
//     void operator()(value_type t, const Container& y, Container& yp)
//     {
//         dg::blas2::symv(m_A, y, m_helper);  //m_helper = A y //is not normed
//         dg::blas1::axpby(0.5, y, -0.5, m_helper); //m_helper = 1/2 y - 1/2  A y 
//         m_lhs.set_time(t);
//         m_invert( m_lhs, yp, m_helper); // ( (t-1)  - A t ) yp = (1-A)/2 y
//     }
//   private:
//     Container m_helper;
//     Matrix m_A;
//     LhsT<Matrix, Container> m_lhs;
//     unsigned m_size;
//     dg::Invert<Container> m_invert;
// };
// 
// /**
//  * @brief Rhs of the square root ODE \f[ \dot{y}= \left[(t-1) I -t T\right]^{-1} *(I - A)/2 * y \f]
//  *
//  * where \f[ T\f] is a asymmetric tridiagonal matrix
//  * @note Solution of ODE: \f[ y(1) = \sqrt{T} y(0)\f]
//  * @note Todo: Compute inverse analytically
//  */
// template<class Matrix, class Container>
// struct RhsTasym
// {
//   public:
//     using matrix_type = Matrix;
//     using container_type = Container;
//     using value_type = dg::get_value_type<Container>;
//     /**
//      * @brief Construct Rhs operator
//      *
//      * @param T asymmetric matrix
//      * @param copyable copyable container
//      * @param eps Accuarcy for LGMRES solve
//      */
//     RhsTasym( const Matrix& T,  const Container& copyable,  const value_type& eps):
//          m_A(T),
//          m_helper( copyable),
//          m_eps(eps),
//          m_lhs(m_A, copyable),
//          m_lgmres( copyable, 30,10,100) //weights not multiplied on rhs
// //         m_bicgstabl(copyable, 50, 2)
//     {
//     }
//     /**
//      * @brief Set the Matrix T
//      *
//      * @param T Matrix
//      */
//      void set_T( const Matrix& T) { 
//          m_A=T; 
//          m_lhs.set_T(T);
//     }  
//     /**
//      * @brief Compute rhs term (including inversion of lhs) 
//      *
//      * i.e. \f[ yp= ((t-1) I -t T)^{-1} *(I - T)/2 * y \f]
//      * @param y  is \f[ y\f]
//      * @param yp is \f[ \dot{y}\f]
//      * @note Solution of ODE: \f[ y(1) = \sqrt{T} y(0)\f]
//      */
//     void operator()(value_type t, const Container& y, Container& yp)
//     {
//         dg::blas2::symv(m_A, y, m_helper);  //m_helper = A y //is not normed
//         dg::blas1::axpby(0.5, y, -0.5, m_helper); //m_helper = 1/2 y - 1/2  A y 
//         m_lhs.set_time(t);
//         m_lgmres.solve( m_lhs, yp, m_helper, m_lhs.inv_weights(), m_lhs.inv_weights(), m_eps, 1); // ( (t-1)  - A t ) yp = (1-A)/2 y
// //         m_bicgstabl.solve( m_lhs, yp, m_helper, m_lhs.inv_weights(), m_lhs.inv_weights(), m_eps); // ( (t-1)  - A t ) yp = (1-A)/2 y
// 
//     }
//   private:
//     Matrix m_A;
//     Container m_helper;
//     value_type m_eps;
//     LhsT<Matrix, Container> m_lhs;
//     dg::LGMRES<Container> m_lgmres;
// //     dg::BICGSTABl<Container> m_bicgstabl;
// 
// };

// template< class Matrix, class Container>
// struct dg::TensorTraits< LhsT<   Matrix, Container> >
// {
//     using value_type  = dg::get_value_type<Container>;
//     using tensor_category = dg::SelfMadeMatrixTag;
// };
// 
