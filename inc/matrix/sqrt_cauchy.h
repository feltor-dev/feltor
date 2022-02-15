#pragma once
// #undef BOOST_MATH_MAX_SERIES_ITERATION_POLICY
// #define BOOST_MATH_MAX_SERIES_ITERATION_POLICY 1000000000
#include <boost/math/special_functions.hpp>

#include "dg/algorithm.h"

//! M_PI is non-standard ... so MSVC complains
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/**
* @brief Classes for square root Matrix-Vector product computation via the Cauchy integral
*/
namespace dg {
namespace mat {

/**
 * @brief Cauchy integral \f$ \sqrt{A} b=  A\frac{ 2 K' \sqrt{m}}{\pi N}  \sum_{j=1}^{N} ( w_j^2 I + A)^{-1} c_j d_j  b \f$
 *
 * A is the matrix, b is the vector, w is a scalar m is the smallest eigenvalue
 * of A, K' is the conjuated complete  elliptic integral and \f$c_j\f$ and
 * \f$d_j\f$ are the jacobi functions
 *
 * @note Actually approximates the inverse square root. If we want the square
 * root we multiply A in the end
 *
 *This class is based on the approach (method 3) of the paper <a href="https://doi.org/10.1137/070700607" > Computing A alpha log(A), and Related Matrix Functions by Contour Integrals </a>  by N. Hale et al
 *
 * @ingroup matrixfunctionapproximation
 *
 */
template<class Container>
struct SqrtCauchyInt
{
  public:
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    SqrtCauchyInt() { }
    /**
     * @brief Construct Rhs operator
     *
     * @param copyable
     */
    SqrtCauchyInt( const Container& copyable)
    {
        m_helper = m_temp = m_helper3 = copyable;
        m_temp_ex.set_max(1, copyable);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = SqrtCauchyInt( std::forward<Params>( ps)...);
    }

    ///The \f$ w\f$ in \f$ ( w^2I + A)^{-1}\f$
    const double& w() const{return m_w;}

    ///The functor returning \f$ (w^2I + A )\f$
    /// A is stored by reference
    /// (use if needed)
    template<class MatrixType>
    auto make_denominator(MatrixType& A) const{
        return [&A=A, &w = m_w] ( const auto& x, auto& y)
        {
            dg::blas2::symv(A, x, y); // A x
            dg::blas1::axpby(w*w, x, 1., y); // ( w^2 + A) x
        };
    }

    /*
     * @brief Cauchy integral \f$ x = \sqrt(A) = \frac{- 2 K' \sqrt{m}}{\pi N} A \sum_{j=1}^{N} (w_j^2 I - A)^{-1} c_j d_j  b \f$
     *
     * @param A A self-adjoint or non-self-adjoint Matrix
     * @param wAinv The operator \f$ (w^2 I - A)^{-1}\f$ (construct with the
     *  help of \c w() and \c make_denominator(A), we provide an initial guess)
     * @param b is input vector
     * @param x contains the result
     * @param minEV minimum Eigenvalue of A
     * @param maxEV maximum Eigenvalue of A
     * @param steps Number of steps to use in the integration
     * @note The Jacobi elliptic functions are related to the Mathematica
     * functions via \c jacobi_cn(k,u ) = JacobiCN_(u,k^2), ... and the complete
     * elliptic integral of the first kind via
     * \c comp_ellint_1(k) = EllipticK(k^2)
     * @param exp If +1 then the sqrt is computed else the inverse sqrt
     */
    template<class MatrixType0, class MatrixType1, class ContainerType0,
        class ContainerType1>
    void operator()(MatrixType0&& A, MatrixType1&& wAinv, const ContainerType0&
            b, ContainerType1& x, value_type minEV, value_type maxEV, unsigned
            steps,  int exp = +1)
    {
        dg::blas1::copy(0., m_helper3);
        value_type s=0.;
        value_type c=0.;
        value_type d=0.;
        m_w=0.;
        value_type t=0.;
        value_type sqrtminEV = std::sqrt(minEV);
        const value_type k2 = minEV/maxEV;
        const value_type sqrt1mk2 = std::sqrt(1.-k2);
        const value_type Ks=boost::math::ellint_1(sqrt1mk2 );
        const value_type fac = 2.* Ks*sqrtminEV/(M_PI*steps);
        for (unsigned j=1; j<steps+1; j++)
        {
            t  = (j-0.5)*Ks/steps; //imaginary part .. 1i missing
            c = 1./boost::math::jacobi_cn(sqrt1mk2, t);
            s = boost::math::jacobi_sn(sqrt1mk2, t)*c;
            d = boost::math::jacobi_dn(sqrt1mk2, t)*c;
            m_w = sqrtminEV*s;
            dg::blas1::axpby(c*d, b, 0.0 , m_helper); //m_helper = c d b
            m_temp_ex.extrapolate(t, m_temp);
            dg::blas2::symv( std::forward<MatrixType1>(wAinv), m_helper, m_temp);
            m_temp_ex.update(t, m_temp);

            dg::blas1::axpby(fac, m_temp, 1.0, m_helper3); // m_helper3 += fac  (w^2 + A )^(-1) c d x
        }
        if( exp > 0)
            dg::blas2::symv(A, m_helper3, x);
        else
            dg::blas1::copy( m_helper3, x);
    }

  private:
    Container m_helper, m_temp, m_helper3;
    value_type m_w;
    dg::Extrapolation<Container> m_temp_ex;
};

} //namespace mat
} //namespace dg
