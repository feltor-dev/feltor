#pragma once

#include "magnetic_field.h"

namespace dg{
namespace geo{
namespace toroidal{

BinaryFunctorsLvl2 createPsip( )
{
    BinaryFunctorsLvl2 psip( new Constant(1), new Constant(0), new Constant(0),new Constant(0), new Constant(0), new Constant(0));
    return psip;
}
BinaryFunctorsLvl1 createIpol( )
{
    BinaryFunctorsLvl1 ipol( new Constant(1), new Constant(0), new Constant(0))
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

}//namespace toroidal

TokamakMagneticField createToroidalField( double R0)
{
    return TokamakMagneticField( R0, toiroidal::createPsip(), toiroidal::createIpol());
}
}//namespace geo
}//namespace dg
