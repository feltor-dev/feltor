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
* @brief Classes for square root Matrix-Vector product computation via the %Cauchy integral
*/
namespace dg {
namespace mat {

/**
 * @brief %Cauchy integral \f$ \sqrt{A} b=  A\frac{ 2 K' \sqrt{m}}{\pi N}  \sum_{j=1}^{N} ( w_j^2 I + A)^{-1} c_j d_j  b \f$
 *
 * A is the matrix, b is the vector, w is a scalar m is the smallest eigenvalue
 * of A, K' is the conjuated complete  elliptic integral and \f$c_j\f$ and
 * \f$d_j\f$ are the jacobi functions
 *
 * @note If we leave away the first A on the right hand side we approximate the
 *  inverse square root.
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

    /**
     * @brief %Cauchy integral \f$ x = \sqrt{A}b = A \frac{ 2 K' \sqrt{m}}{\pi N}  \sum_{j=1}^{N} (w_j^2 I + A)^{-1} c_j d_j  b \f$
     *
     * @note The Eigenvalues can be estimated from a few lanczos iterations (which
     *  is at least more reliable than doing it semi-analytically)
     * @code{.cpp}
     *  dg::mat::UniversalLanczos lanczos( A.weights(), 20);
     *  auto T = lanczos.tridiag( A, A.weights(), A.weights());
     *  auto EVs = dg::mat::compute_extreme_EV( T);
     * @endcode
     * @param A A self-adjoint or non-self-adjoint Matrix
     * @param wAinv The operator \f$ (w^2 I + A)^{-1}\f$ (construct with the
     *  help of \c w() and \c make_denominator(A), we provide an initial guess)
     * @param b is input vector
     * @param x contains the result
     * @param EVs {minimum Eigenvalue of A, maximum Eigenvalue of A}
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
            b, ContainerType1& x, std::array<value_type,2> EVs, unsigned
            steps,  int exp = +1)
    {
        dg::blas1::copy(0., m_helper3);
        value_type s=0.;
        value_type c=0.;
        value_type d=0.;
        m_w=0.;
        value_type t=0.;
        value_type minEV = EVs[0], maxEV = EVs[1];
        value_type sqrtminEV = std::sqrt(minEV);
        const value_type k2 = minEV/maxEV;
        const value_type sqrt1mk2 = std::sqrt(1.-k2);
        const value_type Ks=boost::math::ellint_1(sqrt1mk2 );
        const value_type fac = 2.* Ks*sqrtminEV/(M_PI*steps);
        for (unsigned j=1; j<steps+1; j++)
        {
            t  = (j-0.5)*Ks/steps; //imaginary part .. 1i missing
            // approx 5e-6 s each evaluation of 3 boost fcts
            c = 1./boost::math::jacobi_cn(sqrt1mk2, t);
            s = boost::math::jacobi_sn(sqrt1mk2, t)*c;
            d = boost::math::jacobi_dn(sqrt1mk2, t)*c;
            m_w = sqrtminEV*s;
            dg::blas1::axpby(c*d, b, 0.0 , m_helper); //m_helper = c d b
            dg::blas2::symv( std::forward<MatrixType1>(wAinv), m_helper, m_temp);

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
};

/** @brief Shortcut for \f$b \approx \sqrt{A} x \f$ solve directly via sqrt %Cauchy combined with PCG inversions
 * @ingroup matrixfunctionapproximation
*/
template< class Container>
struct DirectSqrtCauchy
{
   public:
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    DirectSqrtCauchy() {}
    /**
     * @brief Construct DirectSqrtCauchy
     *
     * @note The Eigenvalues can be estimated from a few lanczos iterations (which
     *  is at least more reliable than doing it semi-analytically)
     * @code{.cpp}
     *  dg::mat::UniversalLanczos lanczos( A.weights(), 20);
     *  auto T = lanczos.tridiag( A, A.weights(), A.weights());
     *  auto EVs = dg::mat::compute_extreme_EV( T);
     * @endcode
     * @param A The matrix (stored by reference)
     * @param weights
     * @param epsCG
     * @param iterCauchy maximum number of %Cauchy iterations
     * @param EVs {minimum Eigenvalue of A, maximum Eigenvalue of A}
     * @param exp if < 0 then the inverse sqrt is computed, else the sqrt
     */
    template<class MatrixType>
    DirectSqrtCauchy(
            MatrixType& A,
            const Container& weights,
            value_type epsCG,
            unsigned iterCauchy,
            std::array<value_type,2> EVs, int exp)
    {
        m_pcg.construct( weights, 10000);
        Container m_temp = weights;
        m_iterCauchy = iterCauchy;
        m_cauchysqrtint.construct(weights);
        m_EVs = EVs;
        m_A = [&]( const Container& x, Container& y){
            dg::blas2::symv( A, x, y);
        };
        m_op = m_cauchysqrtint.make_denominator(A);
        m_wAinv = [&, eps = epsCG, w = weights]
            ( const Container& x, Container& y){
                m_pcg.solve( m_op, y, x, 1., w, eps);
        };
        m_exp = exp;
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = DirectSqrtCauchy( std::forward<Params>( ps)...);
    }
    /**
     * @brief Compute \f$x \approx \sqrt{A} b \f$ via sqrt %Cauchy integral solve
     *
     * @param b input vector
     * @param x output vector. Is approximating \f$x \approx \sqrt{A} b  \f$
     * @return number of integration steps of sqrt cauchy solve
     */
    unsigned operator()(const Container& b, Container& x)
    {
        m_cauchysqrtint(m_A, m_wAinv, b, x, m_EVs, m_iterCauchy, m_exp);
        return m_iterCauchy;
    }
  private:
    unsigned m_iterCauchy;
    std::function<void ( const Container&, Container&)> m_A, m_wAinv, m_op;
    dg::PCG<Container> m_pcg;
    dg::mat::SqrtCauchyInt<Container> m_cauchysqrtint;
    std::array<value_type,2> m_EVs;
    int m_exp;
};


} //namespace mat
} //namespace dg
