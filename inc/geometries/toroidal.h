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
BinaryFunctorsLvl2 createPsip( )
{
    BinaryFunctorsLvl2 psip( new Constant(1), new Constant(0), new Constant(0),new Constant(0), new Constant(0), new Constant(0));
    return psip;
}
/**
 * @brief constant \f$ I = 1\f$
 * @return 
 */
BinaryFunctorsLvl1 createIpol( )
{
    BinaryFunctorsLvl1 ipol( new Constant(1), new Constant(0), new Constant(0));
    return ipol;
}

/**
 * @brief Models a slab toroidal field 
 *
 * \f$ B=\frac{R_0}{R}\f$, \f$ \psi_p = 1\f$ and \f$ I = 1\f$.
 @note The solovev field can also be made to model a todoidal slab field
 */
TokamakMagneticField createMagField( double R0)
{
    return TokamakMagneticField( R0, createPsip(), createIpol());
}
///@}

}//namespace toroidal
namespace circular{

///@addtogroup circular
///@{
/**
 * @brief \f[ \psi_p = \frac{1}{2}\left((R-R_0)^2 + Z^2 \right) \f]
 * gives circular flux surfaces
 */
struct Psip : public aCloneableBinaryFunctor<Psip>
{ /**
     * @brief Construct from major radius
     * @param R0 the major radius
     */
    Psip( double R0): m_R0(R0) { }
  private:
    double do_compute(double R, double Z) const
    {    
        return 0.5*((R-m_R0)*(R-m_R0) + Z*Z);
    }
    double m_R0;
};
/// @brief \f[ R-R_0 \f]
struct PsipR : public aCloneableBinaryFunctor<PsipR>
{ /**
     * @brief Construct from major radius
     * @param R0 the major radius
     */
    PsipR( double R0): m_R0(R0) { }
  private:
    double do_compute(double R, double Z) const
    {    
        return R-m_R0;
    }
    double m_R0;
};
///@brief \f[ Z \f]
struct PsipZ : public aCloneableBinaryFunctor<PsipZ>
{ 
  private:
    double do_compute(double R, double Z) const
    {    
        return Z;
    }
};

/**
 * @brief circular \f$\psi_p = \frac{1}{2}\left((R-R_0)^2 + Z^2 \right)\f$
 * @return 
 */
BinaryFunctorsLvl2 createPsip( double R0 )
{
    BinaryFunctorsLvl2 psip( new Psip(R0), new PsipR(R0), new PsipZ(),new Constant(1), new Constant(0), new Constant(1));
    return psip;
}
/**
 * @brief constant \f$ I = I_0\f$
 * @return 
 */
BinaryFunctorsLvl1 createIpol( double I0 )
{
    BinaryFunctorsLvl1 ipol( new Constant(I0), new Constant(0), new Constant(0));
    return ipol;
}
///@}
}//namespace circular

/**
 * @brief Create a Toroidal Magnetic field
 * @param R0 the major radius
 * @return A magnetic field object
 * @ingroup geom
 * @note The solovev field can also be made to model a todoidal slab field
 */
TokamakMagneticField createToroidalField( double R0)
{
    return TokamakMagneticField( R0, toroidal::createPsip(), toroidal::createIpol());
}
/**
 * @brief Create a Magnetic field with circular flux surfaces and constant current
 * @param R0 the major radius
 * @param I0 the current
 * @return A magnetic field object
 * @ingroup geom
 */
TokamakMagneticField createCircularField( double R0, double I0)
{
    return TokamakMagneticField( R0, circular::createPsip(R0), circular::createIpol(I0));
}

}//namespace geo
}//namespace dg
