#pragma once

#include "magnetic_field.h"

namespace dg{
namespace geo{
namespace toroidal{

///@addtogroup toroidal
///@{
/**
 * @brief constant \f$\psi_p = 1\f$
 * @return
 */
inline CylindricalFunctorsLvl2 createPsip( )
{
    CylindricalFunctorsLvl2 psip( Constant(1), Constant(0), Constant(0),Constant(0), Constant(0), Constant(0));
    return psip;
}
/**
 * @brief constant \f$ I = 1\f$
 * @return
 */
inline CylindricalFunctorsLvl1 createIpol( )
{
    CylindricalFunctorsLvl1 ipol( Constant(1), Constant(0), Constant(0));
    return ipol;
}

///@}

}//namespace toroidal
namespace circular{

///@addtogroup circular
///@{
/**
 * @brief \f[ \psi_p = 1- \left(\frac{R-R_0}{a}\right)^2 - \left( \frac{Z}{b}\right)^2 \f]
 * gives ellipsoid flux surfaces
 */
struct Psip : public aCylindricalFunctor<Psip>
{ /**
     * @brief Construct from major radius
     * @param R0 the major radius
     * @param a the length of R semi-axis
     * @param b the length of Z semi-axis
     */
    Psip( double R0, double a, double b): m_R0(R0), m_a(a), m_b(b) { }
    double do_compute(double R, double Z) const
    {
        return 1. - (R-m_R0)*(R-m_R0)/m_a/m_a - Z*Z/m_b/m_b;
    }
  private:
    double m_R0, m_a, m_b;
};
/// @brief \f[ -2(R-R_0)/a^2 \f]
struct PsipR : public aCylindricalFunctor<PsipR>
{ /**
     * @brief Construct from major radius
     * @param R0 the major radius
     * @param a the length of R semi-axis
     */
    PsipR( double R0, double a): m_R0(R0), m_a(a) { }
    double do_compute(double R, double Z) const
    {
        return -2*(R-m_R0)/m_a/m_a;
    }
  private:
    double m_R0, m_a;
};
///@brief \f[ -2Z/b^2 \f]
struct PsipZ : public aCylindricalFunctor<PsipZ>
{
    PsipZ( double b): m_b(b) { }
    double do_compute(double R, double Z) const
    {
        return -2*Z/m_b/m_b;
    }
  private:
    double m_b;
};

/**
 * @brief \f$ \psi_p = 1- \left(\frac{R-R_0}{a}\right)^2 - \left( \frac{Z}{b}\right)^2 \f$
 * gives ellipsoid flux surfaces
 * @param R0 the major radius
 * @param a the length of R semi-axis
 * @param b the length of Z semi-axis
 * @return
 */
inline CylindricalFunctorsLvl2 createPsip( double R0, double a , double b )
{
    return CylindricalFunctorsLvl2( Psip(R0, a, b), PsipR(R0, a), PsipZ(b),
        Constant(-2./a/a), Constant(0), Constant(-2./b/b));
}
/**
 * @brief constant \f$ I = I_0\f$
 * @return
 */
inline CylindricalFunctorsLvl1 createIpol( double I0 )
{
    CylindricalFunctorsLvl1 ipol( Constant(I0), Constant(0), Constant(0));
    return ipol;
}
///@}
}//namespace circular

/**
 * @brief Create a Toroidal Magnetic field
 *
 * \f[ \psi_p(R,Z) = 1, \quad I(\psi_p) = 1\f]
 * @param R0 the major radius
 * @return A magnetic field object
 * @ingroup toroidal
 * @note The solovev field can also be made to model a todoidal slab field
 * @note Chooses elongation=a=1, triangularity=0 and description as "none"
 */
inline dg::geo::TokamakMagneticField createToroidalField( double R0)
{
    MagneticFieldParameters params = { 1., 1., 0.,
            equilibrium::circular, modifier::none, description::none};
    return TokamakMagneticField( R0, toroidal::createPsip(), toroidal::createIpol(), params);
}
/**
 * @brief \f$ \psi_p = 1- \left(\frac{R-R_0}{a}\right)^2 - \left( \frac{Z}{b}\right)^2,\quad I(\psi_p) = I_0 \f$
 *
 * Create a Magnetic field with ellipsoid flux surfaces and constant current
 * @param R0 the major radius
 * @param I0 the current
 * @param a the length of R semi-axis
 * @param b the length of Z semi-axis
 * @return A magnetic field object
 * @ingroup circular
 * @note Chooses elongation=a=1, triangularity=0 and description as standardO
 */
inline dg::geo::TokamakMagneticField createCircularField( double R0, double I0, double a = 1, double b = 1)
{
    MagneticFieldParameters params = { a, 1., 0.,
            equilibrium::circular, modifier::none, description::standardO};
    return TokamakMagneticField( R0, circular::createPsip(R0, a, b), circular::createIpol(I0), params);
}

}//namespace geo
}//namespace dg
