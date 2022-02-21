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
static inline CylindricalFunctorsLvl2 createPsip( )
{
    CylindricalFunctorsLvl2 psip( Constant(1), Constant(0), Constant(0),Constant(0), Constant(0), Constant(0));
    return psip;
}
/**
 * @brief constant \f$ I = 1\f$
 * @return
 */
static inline CylindricalFunctorsLvl1 createIpol( )
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
 * @brief \f[ \psi_p = \frac{1}{2}\left((R-R_0)^2 + Z^2 -1\right) \f]
 * gives circular flux surfaces
 */
struct Psip : public aCylindricalFunctor<Psip>
{ /**
     * @brief Construct from major radius
     * @param R0 the major radius
     */
    Psip( double R0): m_R0(R0) { }
    double do_compute(double R, double Z) const
    {
        return 0.5*((R-m_R0)*(R-m_R0) + Z*Z - 1.0);
    }
  private:
    double m_R0;
};
/// @brief \f[ R-R_0 \f]
struct PsipR : public aCylindricalFunctor<PsipR>
{ /**
     * @brief Construct from major radius
     * @param R0 the major radius
     */
    PsipR( double R0): m_R0(R0) { }
    double do_compute(double R, double Z) const
    {
        return R-m_R0;
    }
  private:
    double m_R0;
};
///@brief \f[ Z \f]
struct PsipZ : public aCylindricalFunctor<PsipZ>
{
    double do_compute(double R, double Z) const
    {
        return Z;
    }
  private:
};

/**
 * @brief circular \f$\psi_p = \frac{1}{2}\left((R-R_0)^2 + Z^2 -1 \right)\f$
 * @return
 */
static inline CylindricalFunctorsLvl2 createPsip( double R0 )
{
    return CylindricalFunctorsLvl2( Psip(R0), PsipR(R0), PsipZ(),
        Constant(1), Constant(0), Constant(1));
}
/**
 * @brief constant \f$ I = I_0\f$
 * @return
 */
static inline CylindricalFunctorsLvl1 createIpol( double I0 )
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
static inline dg::geo::TokamakMagneticField createToroidalField( double R0)
{
    MagneticFieldParameters params = { 1., 1., 0.,
            equilibrium::circular, modifier::none, description::none};
    return TokamakMagneticField( R0, toroidal::createPsip(), toroidal::createIpol(), params);
}
/**
 * @brief Create a Magnetic field with circular flux surfaces and constant current

 * \f[ \psi_p(R,Z) = \frac{1}{2}\left((R-R_0)^2 + Z^2 - 1\right), \quad I(\psi_p) = I_0 \f]
 * @param R0 the major radius
 * @param I0 the current
 * @return A magnetic field object
 * @ingroup circular
 * @note Chooses elongation=a=1, triangularity=0 and description as standardO
 */
static inline dg::geo::TokamakMagneticField createCircularField( double R0, double I0)
{
    MagneticFieldParameters params = { 1., 1., 0.,
            equilibrium::circular, modifier::none, description::standardO};
    return TokamakMagneticField( R0, circular::createPsip(R0), circular::createIpol(I0), params);
}

}//namespace geo
}//namespace dg
