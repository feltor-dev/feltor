#pragma once
#include <boost/math/special_functions.hpp>
namespace dg
{
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
}//namespace dg
