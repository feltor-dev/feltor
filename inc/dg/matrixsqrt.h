#include "sqrt_ode.h"
#include "adaptive.h"


///@brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve directly via sqrt ODE solve with adaptive ERK class as timestepper
template< class Geometry, class Matrix, class Container>
struct DirectSqrtSolve
{
   public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    DirectSqrtSolve() {}
    /**
     * @brief Construct DirectSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     */
    DirectSqrtSolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG, value_type epsTimerel, value_type epsTimeabs):   
        m_A(A),
        m_rhs(A, g, epsCG),
        m_epsTimerel(epsTimerel),
        m_epsTimeabs(epsTimeabs) 
    { }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x  \f] via sqrt ODE solve
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \f]
     * @return number of timesteps of sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        return dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., x, 1., b, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs);
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    Rhs<Geometry, Matrix, Container> m_rhs;  
    value_type m_epsTimerel, m_epsTimeabs;
};

/*! 
 * @brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[b \approx \sqrt{A} x \approx b \approx ||x||_M V \sqrt{T} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the Lanczos solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T} e_1\f] is computed via the sqrt ODE solve.
 */
template< class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container>
struct KrylovSqrtSolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    KrylovSqrtSolve() {}
    /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     */
    KrylovSqrtSolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, value_type epsTimerel, value_type epsTimeabs, unsigned iter):   
        m_A(A),
        m_epsTimerel(epsTimerel),
        m_epsTimeabs(epsTimeabs),
        m_xnorm(0.),
        m_e1(iter, 0.),
        m_y(iter, 1.),
        m_rhs(m_T, m_e1, epsCG),
        m_lanczos(copyable, iter)
    { 
        m_e1[0]=1.;
    }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f] via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f]
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        //Lanczos solve first         
        m_xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        m_TVpair = m_lanczos(m_A, x, b, m_A.weights(), m_A.inv_weights()); 
        m_T = m_TVpair.first; 
        m_V = m_TVpair.second;   
        //update T
        m_rhs.set_T(m_T);
        
        unsigned counter = dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., m_e1, 1., m_y, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs); // y = T^(1/2) e_1
        dg::blas2::gemv(m_V, m_y, b);
        dg::blas1::scal(b, m_xnorm);             // b = ||x|| V T^(1/2) e_1     
        return counter;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    value_type m_epsTimerel, m_epsTimeabs, m_xnorm;
    Container m_e1, m_y;
    RhsT<DiaMatrix, Container> m_rhs;  
    dg::Lanczos< Container > m_lanczos;
    DiaMatrix m_T; 
    CooMatrix m_V;
    std::pair<DiaMatrix, CooMatrix> m_TVpair; 
};
