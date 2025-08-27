#pragma once
#include <boost/math/special_functions.hpp>
#include "../dg/backend/config.h" // DG_DEVICE
namespace dg {
namespace mat {
/**
 * @brief \f$ f(x) = I_0 (x)\f$ with \f$I_0\f$ the zeroth order modified Bessel function
 *
 * @tparam T value type
 */
template < class T = double>
struct BESSELI0
{
    BESSELI0( ) {}
    /**
     * @brief return \f$ f(x) = I_0 (x)\f$ with \f$I_0\f$ the zeroth order modified Bessel function
     *
     * @param x x
     *
     * @return \f$ I_0 (x)\f$
     */
    T operator() ( T x) const
    {
        return boost::math::cyl_bessel_i(0, x);
    }
};

/**
 * @brief \f$ f(x) = J_n (x)\f$ with \f$J_n\f$ the n-th order modified Bessel function
 *
 * @tparam T value type
 */
template< class T = double >
struct BesselJ
{
    BesselJ(unsigned n): m_n(n) {}

    T operator() ( T x) const
    {
        return boost::math::cyl_bessel_j(m_n, x);
    }
    private:
    	unsigned m_n;
};

/**
 * @brief \f$ f(x) = L_n (x)\f$ with \f$L_n\f$ the n-th order Laguerre polynomial
 *
 * @tparam T value type
 */
template< class T = double >
struct LaguerreL
{
    LaguerreL(unsigned n): m_n(n) {}

    T operator() ( T x) const
    {
        return boost::math::laguerre(m_n, x);
    }
    private:
        unsigned m_n;
};


/**
 * @brief \f$ f(x) = \Gamma_0 (x) := I_0 (x) exp(x) \f$ with \f$I_0\f$ the zeroth order modified Bessel function
 *
 * @tparam T value type
 */
template < class T = double>
struct GAMMA0
{
    GAMMA0( ) {}
    /**
     * @brief return \f$ f(x) = I_0 (x) exp(x) \f$ with \f$I_0\f$ the zeroth order modified Bessel function
     *
     * @param x x
     *
     * @return \f$ \Gamma_0 (x)\f$
     */
    T operator() ( T x) const
    {
        return exp(x)*boost::math::cyl_bessel_i(0, x);
    }
};


/**
 * @brief \f$ f(x) = (\exp(x) - 1)/x\f$
 *
 * Accurate evaluation close to and at 0
 * @tparam T value type
 */
template<class T>
T phi1( T x){
    if ( fabs( x) < 1e-16 )
        return 1.;
    if ( fabs(x) < 1)
        return expm1( x)/x;
    return (exp( x) - 1)/x;
}

/**
 * @brief \f$ f(x) = (\exp(x) - x - 1)/x^2\f$
 *
 * Accurate evaluation close to and at 0
 * @tparam T value type
 */
template<class T>
T phi2( T x){
    if ( fabs( x) < 1e-16 )
        return 1./2;
    if ( fabs(x) < 1e-2)
        return 1./2.+x*(1./6.+x*(1./24. + x/120.));
    return ((exp( x) - 1)/x - 1)/x;
}

/**
 * @brief \f$ f(x) = (\exp(x) - x^2/2 -x- 1)/x^3\f$
 *
 * Accurate evaluation close to and at 0
 * @tparam T value type
 */
template<class T>
T phi3( T x){
    if ( fabs( x) < 1e-16 )
        return 1./6.;
    if ( fabs(x) < 1e-2)
        return 1./6. + x*(1./24.+x*(1./120. +x/720.));
    return (((exp( x) - 1)/x - 1)/x-1./2.)/x;
}
/**
 * @brief \f$ f(x) = (\exp(x) - x^3/6 - x^2/2 -x- 1)/x^4\f$
 *
 * Accurate evaluation close to and at 0
 * @tparam T value type
 */
template<class T>
T phi4( T x){
    if ( fabs( x) < 1e-16 )
        return 1./24.;
    if ( fabs(x) < 1e-2)
        return 1./24. + x*(1./120.+x*(1./720. + x/5040));
    return ((((exp( x) - 1)/x - 1)/x-1./2.)/x-1./6.)/x;
}

/**
 * @brief \f$ f(x) = (-a*x)^n/n! exp(a*x) \f$
 *
 * @tparam T value type (can be complex)
 */
template<class T = double>
struct GyrolagK
{
    /**
     * @brief Construct
     * @param n order cannot be greater than 12
     * @param a prefactor of x
     */
    GyrolagK(unsigned n, T a): m_n (n), m_a(a) {}

    DG_DEVICE
    T operator()(T x) const {
        // Factorials overflow int32 for n>12
        static constexpr unsigned fact[] = {1, 1, 2, 6, 24, 120, 720,
            5040, 40320, 362880, 3628800, 39916800, 479001600};

        if( m_n == 0)
            return exp(x*m_a); // faster to evaluate than tgamma and pow ...
        if( m_n == 1)
            return (-x*m_a)*exp( x*m_a);
        if( m_n == 2)
            return 0.5*(-x*m_a)*(-x*m_a)*exp( x*m_a);
        if( m_n == 3)
            return (-x*m_a)*(-x*m_a)*(-x*m_a)*exp( x*m_a)/6.0;
        return pow(-x*m_a,m_n)/fact[m_n]*exp(x*m_a);
    }
    DG_DEVICE
    T operator()(T x, T y) const { return this->operator()(x*y); }

    private:
    unsigned m_n;
    T m_a;
};

/**
 * @brief \f$ f(x) = \frac{n}{x}+a \f$
 *
 * @tparam T value type (can be complex)
 */
template<class T = double>
struct DLnGyrolagK
{
    DLnGyrolagK(unsigned n, T a): m_n (n), m_a(a) {}
    T  DG_DEVICE operator()(T x) const {
        return (T)m_n/x+m_a;
    }
    T  DG_DEVICE operator()(T x, T y) const { return this->operator()(x*y); }

    private:
    unsigned m_n;
    T m_a;
};
/**
 * @brief \f$ f(x) = -\frac{n}{x^2}  \f$
 *
 * @tparam T value type (can be complex)
 */
template<class T = double>
struct DDLnGyrolagK
{
    DDLnGyrolagK(unsigned n, T a): m_n (n), m_a(a) {}
    T  DG_DEVICE operator()(T x) const {
        return -(T)m_n/x/x;
    }
    T  DG_DEVICE operator()(T x, T y) const { return this->operator()(x*y); }

    private:
    unsigned m_n;
    T m_a;
};

}//namespace mat
}//namespace dg
