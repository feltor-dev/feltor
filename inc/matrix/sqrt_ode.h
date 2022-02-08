#pragma once

#include "dg/algorithm.h"

/**
* @brief Classes for Matrix function-Vector product computation via the ODE method
*/
namespace dg
{

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
     * @param A self-adjoint matrix (stored by reference so needs to live)
     * @param copyable copyable container
     * @param eps Accuracy for CG solve
     * @param symmetric true = self-adjoint A / false = non-self-adjoint A
     */
    SqrtODE( Matrix& A,  const Container& copyable,  value_type eps, bool symmetric)
    {
        m_helper = copyable;
        m_A = A;
        m_symmetric = symmetric;
        m_eps = eps;
        m_size = m_helper.size();
        m_number = 0;
        if (m_symmetric == true) m_pcg.construct( m_helper, m_size*m_size+1);
        else m_lgmres.construct( m_helper, 30, 10, 100*m_size*m_size);
        m_yp_ex.set_max(3, copyable);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = SqrtODE( std::forward<Params>( ps)...);
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
        m_yp_ex.set_max(3, m_helper);
        m_size = new_max;
    }
    ///@brief Get the current size of vectors
    ///@return the current vector size
    unsigned get_size() const {return m_size;}
    /**
     * @brief Compute rhs term (including inversion of lhs via cg or lgmres)
     *
     * i.e. \f[ yp= ((t-1) I -t V A)^{-1} (I - V A)/2  y \f] if weights are
     * multiplied or
     * \f$ yp= ((t-1) I -t  A)^{-1} (I -  A)/2 * y \f$ otherwise
     * @param t  is time
     * @param y  is \f$ y\f$
     * @param yp is \f$ \dot{y}\f$
     * @note Solution of ODE: \f$ y(1) = \sqrt{V A} y(0)\f$ if weights are multiplied or  \f$ y(1) = \sqrt{A} y(0)\f$ otherwise
     */
    void operator()(value_type t, const Container& y, Container& yp)
    {
        dg::blas2::symv(m_A, y, m_helper);
        dg::blas1::axpby(0.5, y, -0.5, m_helper);

        m_yp_ex.extrapolate(t, yp);
        auto lhs = [&t = t, &A = m_A](  const Container& y, Container& y)
        {
            dg::blas2::symv(A, x, y);
            dg::blas1::axpby((t-1.), x, -t, y);
        }
        if (m_symmetric == true)
        {
            m_number = m_pcg( lhs, yp, m_helper, 1., m_A.weights(), m_eps);
            if( m_number == m_pcg.get_max()) throw dg::Fail( m_eps);
        }
        else
            m_lgmres.solve( lhs, yp, m_helper, 1., m_A.weights(), m_eps, 1);

        m_yp_ex.update(t, yp);
    }
  private:
    Container m_helper;
    Matrix& m_A;
    unsigned m_size, m_number;
    bool m_symmetric;
    value_type m_eps;
    dg::CG<Container> m_pcg;
    dg::LGMRES<Container> m_lgmres;
    dg::Extrapolation<Container> m_yp_ex;
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
 * @param A self-adjoint matrix (stored by reference so needs to live)
 * @param copyable copyable container
 * @param eps Accuracy for CG solve
 * @param symmetric true = self-adjoint A / false = non-self-adjoint A
 */
template< class Matrix, class Container>
dg::SqrtODE<Matrix,Container> make_sqrtode( Matrix& A, const Container& copyable, value_type eps, bool symmetric)
{
    return SqrtODE<Matrix,Container>( A, copyable, eps, symmetric);
}

/**
 * @brief Right hand side of the exponential ODE \f[ \dot{y}= A y \f]
 * where \f$ A\f$ is the matrix
 *
 * @ingroup matrixfunctionapproximation
 *
 * @note Solution of ODE: \f$ y(1) = \exp{A} y(0)\f$
 */
auto make_expode( Matrix& A)
{
    return [&]( auto t, const auto& y, auto& yp) mutable
    {
        dg::blas2::symv( A, y, yp);
    };
}

/**
 * @brief Right hand side of the (zeroth order) modified Bessel function ODE, rewritten as a system of coupled first order ODEs:
 * \f[ \dot{z_0}= z_1 \f]
 * \f[ \dot{z_1}= A^2 z_0 - t^{-1} z_1 \f]
 * where \f$ A\f$ is the matrix and \f[z=(y,\dot{y})\f]
 *
 * @ingroup matrixfunctionapproximation
 *
 * @note Solution of ODE: \f$ y(1) = I_0(A) y(0)\f$ for initial condition \f$ z(0) = (y(0),0)^T \f$
 */
template<class Matrix>
auto make_besselI0ode( Matrix& A)
{
    return [&m_A = A]( auto t, const auto& z, auto& zp) mutable
    {
        dg::blas2::symv(m_A, z[0], zp[0]);
        dg::blas2::symv(m_A, zp[0], zp[1]);
        dg::blas1::axpby(-1./t, z[1], 1.0, zp[1]);
        dg::blas1::copy(z[0],zp[0]);

    };
}


} //namespace dg
