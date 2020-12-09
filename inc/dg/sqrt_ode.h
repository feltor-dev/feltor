#include "blas.h"
// #include "backend/typedefs.h"
#include "helmholtz.h"


/**
 * @brief Matrix class that represents the lhs operator of the square root ODE
 *
 * @ingroup matrixoperators
 *
 * Unnormed discretization of \f[ ((t-1) I -t V A)*x \f]
 * where \f[ A\f] is a helmholtz operator and t is the time.
 */
template< class Geometry, class Matrix, class Container>
struct Lhs
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Lhs() {}
    /**
     * @brief Construct Lhs operator
     *
     * @param A Helmholtz operator
     * @param g The grid to use
     */
    Lhs( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g):   
        m_helper( dg::evaluate( dg::zero, g)),
        m_A(A)
    { 
        dg::assign( dg::create::inv_volume(g),    m_inv_weights);
        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::create::inv_weights(g),   m_precond);
        t=0.0;
    }
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
    void set_time( const double& time) { t=time; }   
    /**
     * @brief Compute Lhs term and store in output
     *
     * i.e. \f[ y= W ((t-1) I -t V A)*x \f]
     * @param x left-hand-side
     * @param y result
     */ 
    void symv( const Container& x, Container& y) 
    {

        dg::blas2::symv(m_A, x, m_helper); //m_helper =  A x
        dg::blas1::pointwiseDot(m_inv_weights, m_helper, y); //m_helper = V A x ...normed 
        dg::blas1::axpby((t-1.), x, -t, y); //m_helper = (t-1)x - t V A x 
        dg::blas1::pointwiseDot(m_weights, y, y); //make  not normed
    } 
  private:
    Container m_helper;
    Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    double t;
};


/**
 * @brief Rhs of the square root ODE \f[ \dot{y}= \left[(t-1) I -t V A\right]^{-1} *(I - A)/2 * y \f]
 *
 * where \f[ A\f] is a helmholtz operator and t is the time
 * @note Solution of ODE: \f[ y(1) = \sqrt{A} y(0)\f]
 */
template<class Geometry, class Matrix, class Container>
struct Rhs
{
  public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    /**
     * @brief Construct Rhs operator
     *
     * @param A Helmholtz operator
     * @param g The grid to use
     * @param eps Accuarcy for CG solve
     */
    Rhs( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type eps):
         m_helper( dg::evaluate( dg::zero, g)),
         m_A(A),
         m_lhs(m_A, g),
         m_invert( m_helper, g.size(), eps, 1)
    {}
    /**
     * @brief Compute rhs term (including inversion of lhs) 
     *
     * i.e. \f[ yp= ((t-1) I -t V A)^{-1} *(I - A)/2 * y \f]
     * @param y  is \f[ y\f]
     * @param yp is \f[ \dot{y}\f]
     * @note Solution of ODE: \f[ y(1) = \sqrt{A} y(0)\f]
     */
    void operator()(double t, const Container& y, Container& yp)
    {
        dg::blas2::symv(m_A, y, m_helper);  //m_helper = A y
        dg::blas1::pointwiseDot(m_lhs.inv_weights(), m_helper, m_helper); //make normed
        dg::blas1::axpby(0.5, y, -0.5, m_helper); //m_helper = 1/2 y - 1/2 V A y //is normed
        m_lhs.set_time(t);
        m_invert( m_lhs, yp, m_helper); // ( (t-1)  - A t ) yp = (1-A)/2 y
    }
  private:
    Container m_helper;
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    Lhs<Geometry, Matrix, Container> m_lhs;
    dg::Invert<Container> m_invert;
};



template< class Matrix, class Container>
struct LhsT
{
    public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    LhsT() {}
    /**
     * @brief Construct Lhs operator
     *
     * @param T Symmetric tridiagonal matrix
     * @param copyable to get the size of the vectors
     */
    LhsT( const Matrix& T, const Container& copyable):   
        m_A(T),
        m_one(copyable),
        t(0.0)
    { 
        dg::blas1::scal(m_one,0.);
        dg::blas1::plus(m_one,1.0);
    }
    /**
     * @brief Return the weights
     * 
     * @return the  weights
     */
    const Container& weights()const { return m_one; }
    /**
     * @brief Return the inverse weights
     *
     * @return the inverse weights 
     */
    const Container& inv_weights()const { return m_one; }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     * 
     * @return the preconditioner 
     */
    const Container& precond()const { return m_one; }
    /**
     * @brief Set the Matrix T
     *
     * @param T Matrix
     */
    void set_T( const Matrix& T) { m_A=T; }  
    /**
     * @brief Set the time t
     *
     * @param time the time
     */
    void set_time( const double& time) { t=time; }   
    /**
     * @brief Compute Lhs term and store in output
     *
     * i.e. \f[ y=  ((t-1) I -t  T)*x \f]
     * @param x left-hand-side
     * @param y result
     */ 
    void symv( const Container& x, Container& y) 
    {
        dg::blas2::symv(m_A, x, y); //y =  A x
        dg::blas1::axpby((t-1.), x, -t, y); //y = (t-1)x - t  A x 
    } 
  private:
    Matrix m_A;
    Container m_one;
    double t;
};
/**
 * @brief Rhs of the square root ODE \f[ \dot{y}= \left[(t-1) I -t T\right]^{-1} *(I - A)/2 * y \f]
 *
 * where \f[ T\f] is a symmetric tridiagonal matrix
 * @note Solution of ODE: \f[ y(1) = \sqrt{T} y(0)\f]
 * @note Todo: Compute inverse analytically
 */
template<class Matrix, class Container>
struct RhsT
{
  public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    /**
     * @brief Construct Rhs operator
     *
     * @param T Helmholtz operator
     * @param copyable copyable container
     * @param eps Accuarcy for CG solve
     */
    RhsT( const Matrix& T,  const Container& copyable,  value_type eps):
         m_helper( copyable),
         m_A(T),
         m_lhs(m_A, copyable),
         m_invert( m_helper, copyable.size()*copyable.size(), eps, 1, false, 1.) //weights not multiplied on rhs
    {
    }
    /**
     * @brief Set the Matrix T
     *
     * @param T Matrix
     */
     void set_T( const Matrix& T) { 
         m_A=T; 
         m_lhs.set_T(T);
    }  
    /**
     * @brief Compute rhs term (including inversion of lhs) 
     *
     * i.e. \f[ yp= ((t-1) I -t T)^{-1} *(I - T)/2 * y \f]
     * @param y  is \f[ y\f]
     * @param yp is \f[ \dot{y}\f]
     * @note Solution of ODE: \f[ y(1) = \sqrt{T} y(0)\f]
     */
    void operator()(double t, const Container& y, Container& yp)
    {
        dg::blas2::symv(m_A, y, m_helper);  //m_helper = A y //is not normed
        dg::blas1::axpby(0.5, y, -0.5, m_helper); //m_helper = 1/2 y - 1/2  A y 
        m_lhs.set_time(t);
        m_invert( m_lhs, yp, m_helper); // ( (t-1)  - A t ) yp = (1-A)/2 y
    }
  private:
    Container m_helper;
    Matrix m_A;
    LhsT<Matrix, Container> m_lhs;
    dg::Invert<Container> m_invert;

};


template< class Geometry, class Matrix, class Container>
struct dg::TensorTraits< Lhs<Geometry,  Matrix, Container> >
{
    using value_type  = dg::get_value_type<Container>;
    using tensor_category = dg::SelfMadeMatrixTag;
};


template< class Matrix, class Container>
struct dg::TensorTraits< LhsT<   Matrix, Container> >
{
    using value_type  = dg::get_value_type<Container>;
    using tensor_category = dg::SelfMadeMatrixTag;
};

