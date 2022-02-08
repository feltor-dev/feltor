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
namespace dg
{

/**
 * @brief Compute the square root matrix - vector product via the Cauchy integral \f[ \sqrt{A} x=  \frac{- 2 K' \sqrt{m}}{\pi N} A \sum_{j=1}^{N} (w_j^2 I -A)^{-1} c_j d_j  x \f]
 * A is the matrix, x is the vector, w is a scalar m is the smallest eigenvalue of A, K' is the conjuated complete  elliptic integral and \f$c_j\f$ and \f$d_j\f$ are the jacobi functions
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
     * @param eps Accuracy for CG solve
     * @param symmetric true = selfadjoint A / false = non-selfadjoint A
     */
    SqrtCauchyInt( const Container& copyable, value_type eps, bool symmetric)
    {
        m_helper = m_temp = m_helper3 = copyable;
        m_A = A;
        m_symmetric = symmetric;
        m_eps = eps;
        m_size = m_helper.size();
        m_number = 0;
        if (m_symmetric == true) m_pcg.construct( m_helper, m_size*m_size+1);
        else m_lgmres.construct( m_helper, 300, 100, 10*m_size*m_size);
        m_temp_ex.set_max(1, copyable);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = SqrtCauchyInt( std::forward<Params>( ps)...);
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
        m_temp_ex.set_max(1, m_temp);
        m_size = new_max;
    }
    ///@brief Get the current size of vectors
    ///@return the current vector size
    unsigned get_size() const {return m_size;}

    /**
     * @brief Compute cauchy integral (including inversion)
     *
     * i.e. \f[ b=  \frac{- 2 K' \sqrt{m}}{\pi N} V A \sum_{j=1}^{N} (w_j^2 I -V A)^{-1} c_j d_j  x \f]
     * @param A A self-adjoint or non-self-adjoint Matrix
     * @param x is \f$ x\f$
     * @param b is \f$ b\approx \sqrt{V A} x\f$
     * @param weights  that define the scalar product
     * @param minEV
     * @param maxEV
     * @param steps Number of steps to use in the integration
     * @note The Jacobi elliptic functions are related to the Mathematica
     * functions via \c jacobi_cn(k,u ) = JacobiCN_(u,k^2), ... and the complete
     * elliptic integral of the first kind via
     * \c comp_ellint_1(k) = EllipticK(k^2)
     */
    template<class MatrixType, class ContainerType0, class ContainerType1,
        class ContainerType2>
    void operator()(MatrixType& A, const ContainerType0& x, ContainerType1& b,
            const ContainerType2& weights, value_type minEV, value_type maxEV,
            unsigned steps)
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
        const value_type Ks=boost::math::ellint_1(sqrt1mk2 );
        const value_type fac = 2.* Ks*sqrtminEV/(M_PI*iter);
        auto op = [&A=A, &w = w] ( const auto& x, auto& y)
        {
            dg::blas2::symv(A, x, y); // A x
            dg::blas1::axpby(w*w, x, 1., y); // w^2 x + A x
        }
        for (unsigned j=1; j<iter+1; j++)
        {
            t  = (j-0.5)*Ks/iter; //imaginary part .. 1i missing
            c = 1./boost::math::jacobi_cn(sqrt1mk2, t);
            s = boost::math::jacobi_sn(sqrt1mk2, t)*c;
            d = boost::math::jacobi_dn(sqrt1mk2, t)*c;
            w = sqrtminEV*s;
            // op knows w as a reference...
            dg::blas1::axpby(c*d, x, 0.0 , m_helper); //m_helper = c d x
            m_temp_ex.extrapolate(t, m_temp);

            if (m_symmetric == true)
            {
                m_number = m_pcg( op, m_temp, m_helper, 1., weights, m_eps);
                // m_temp = (w^2 +V A)^(-1) c d x
            }
            else
                m_lgmres.solve( op, m_temp, m_helper, 1., weights, m_eps, 1);
            m_temp_ex.update(t, m_temp);

            dg::blas1::axpby(fac, m_temp, 1.0, m_helper3); // m_helper3 += -fac  (w^2 +V A)^(-1) c d x
        }
        dg::blas2::symv(A, m_helper3, b); // - A fac sum (w^2 +V A)^(-1) c d x

    }
  private:
    Container m_helper, m_temp, m_helper3;
    unsigned m_size, m_number;
    bool m_symmetric;
    value_type m_eps;
    dg::CG<Container> m_pcg;
    dg::LGMRES<Container> m_lgmres;
    dg::Extrapolation<Container> m_temp_ex;
};

} //namespace dg
