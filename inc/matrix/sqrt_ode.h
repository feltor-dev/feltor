#pragma once

#include "dg/algorithm.h"
#include "tridiaginv.h"

/**
* @brief Classes for Matrix function-Vector product computation via the ODE method
*/
namespace dg {
namespace mat {

/**
 * @brief Right hand side of the square root ODE \f[ \dot{y}= \left[(1-t) A +t I\right]^{-1} (I - A)/2  y \f]
 * where \f$ A\f$ is the matrix
 *
 * @ingroup matrixfunctionapproximation
 *
 * This class is based on the approach of the paper <a href="https://doi.org/10.1016/S0024-3795(00)00068-9" > Numerical approximation of the product of the square root of a matrix with a vector</a> by E. J. Allen et al
 *
 * @note Solution of ODE: \f$ y(1) = \frac{1}{\sqrt{A}} y(0)\f$ If \f$ y(0) = A b \f$ then we get \f$ y(1) = \sqrt{A} b\f$
 */
template< class Container>
struct InvSqrtODE
{
  public:
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    InvSqrtODE() {};

    /**
     * @brief Construct SqrtOde operator
     *
     * @param A matrix (stored by reference so needs to live)
     * @param copyable copyable container
     */
    template<class MatrixType>
    InvSqrtODE( MatrixType& A, const Container& copyable)
    {
        m_helper = copyable;
        m_A = [&]( const Container& x, Container& y){ return dg::apply( A, x, y);};
        m_yp_ex.set_max(3, copyable);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = SqrtODE( std::forward<Params>( ps)...);
    }

    const value_type& time() const{ return m_time;}

    auto make_operator() const{
        return [&t = m_time, &A = m_A](  const Container& x, Container& y)
        {
            dg::blas2::symv(A, x, y);
            dg::blas1::axpby( t, x, (1.-t), y);
        };
    }
    template<class MatrixType>
    void set_inverse_operator( const MatrixType& OpInv ) {
        m_Ainv = OpInv;
    }
    /**
     * @brief Compute rhs term
     *
     * i.e. \f[ yp= (tI + (1-t) A)^{-1} (I - A)/2  y \f]
     * @param t  is time
     * @param y  is \f$ y\f$
     * @param yp is \f$ \dot{y}\f$
     * @note Solution of ODE: \f$ y(1) = 1/\sqrt{A} y(0)\f$
     */
    void operator()(value_type t, const Container& y, Container& yp)
    {
        m_time = t;
        dg::blas2::symv(m_A, y, m_helper);
        dg::blas1::axpby(0.5, y, -0.5, m_helper);

        m_yp_ex.extrapolate(t, yp);
        dg::blas2::symv( m_Ainv, m_helper, yp);
        m_yp_ex.update(t, yp);
    }
  private:
    Container m_helper;
    std::function<void(const Container&, Container&)> m_A, m_Ainv;
    value_type m_time;
    dg::Extrapolation<Container> m_yp_ex;
};


/**
 * @brief Right hand side of the square root ODE \f[ \dot{y}= \left[tI + (1-t) A\right]^{-1} (I - A)/2  y \f]
 * where \f$ A\f$ is the matrix and the inverse is computed via a \c dg::PCG solver
 *
 * @ingroup matrixfunctionapproximation
 *
 * This class is based on the approach of the paper <a href="https://doi.org/10.1016/S0024-3795(00)00068-9" > Numerical approximation of the product of the square root of a matrix with a vector</a> by E. J. Allen et al
 *
 * @note Solution of ODE: \f$ y(1) = 1/\sqrt{A} y(0)\f$
 * @param A self-adjoint matrix (stored by reference so needs to live)
 * @param P the preconditioner for the PCG method
 * @param weights Weights for PCG method
 * @param epsCG Accuracy for PCG solve
 */
template< class Matrix, class Preconditioner, class Container>
InvSqrtODE<Container> make_inv_sqrtodeCG( Matrix& A, const Preconditioner& P,
        const Container& weights, dg::get_value_type<Container> epsCG)
{
    InvSqrtODE<Container> sqrtode( A, weights);
    dg::PCG<Container> pcg( weights, 10000);
    auto op = sqrtode.make_operator();
    sqrtode.set_inverse_operator( [ = ]( const auto& x, auto& y) mutable
        {
            pcg.solve( op, y, x, P, weights, epsCG);
        });
    return sqrtode;
}

template< class Matrix, class Container>
InvSqrtODE<Container> make_inv_sqrtodeTri( const Matrix& TH, const Container&
        copyable)
{
    InvSqrtODE<Container> sqrtode( TH, copyable);
    sqrtode.set_inverse_operator( [ =, &TH = TH, &t = sqrtode.time() ]
            ( const auto& x, auto& y) mutable
        {
            Matrix wTH = TH;
            dg::blas1::scal( wTH.values.values, (1-t));
            for( unsigned u=0; u<wTH.num_rows; u++)
                wTH.values( u,1) += t;
            dg::apply( dg::mat::invert( wTH), x, y);
        });
    return sqrtode;
}

/**
 * @brief Right hand side of the exponential ODE \f[ \dot{y}= A y \f]
 * where \f$ A\f$ is the matrix
 *
 * @ingroup matrixfunctionapproximation
 *
 * @note Solution of ODE: \f$ y(1) = \exp{A} y(0)\f$
 */
template<class MatrixType>
auto make_expode( MatrixType& A)
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
template<class MatrixType>
auto make_besselI0ode( MatrixType& A)
{
    return [&m_A = A]( auto t, const auto& z, auto& zp) mutable
    {
        dg::blas2::symv(m_A, z[0], zp[0]);
        dg::blas2::symv(m_A, zp[0], zp[1]);
        dg::blas1::axpby(-1./t, z[1], 1.0, zp[1]);
        dg::blas1::copy(z[0],zp[0]);

    };
}


} //namespace mat
} //namespace dg
