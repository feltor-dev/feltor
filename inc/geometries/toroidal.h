#pragma once

#include "functors.h"
#include "magnetic_field.h"


namespace dg{
namespace geo{
namespace toroidal{

/**
 * @brief Models a slab toroidal field (models aTokamakMagneticField)
 *
 * \f$ B=\frac{R_0}{R}\f$, \f$ \psi_p = 1\f$ and \f$ I = 1\f$.
 @note The solovev field can also be made to model a todoidal slab field
 */
struct MagneticField
{
    MagneticField(double R0): R_0(R0), psip(1), psipR(0), psipZ(0), psipRR(0), psipRZ(0), psipZZ(0), laplacePsip(0), ipol(1), ipolR(0), ipolZ(0){}
    double R_0;
    dg::CONSTANT psip;
    dg::CONSTANT psipR;
    dg::CONSTANT psipZ;
    dg::CONSTANT psipRR;
    dg::CONSTANT psipRZ;
    dg::CONSTANT psipZZ;
    dg::CONSTANT laplacePsip;
    dg::CONSTANT ipol;
    dg::CONSTANT ipolR;
    dg::CONSTANT ipolZ;
};

}//namespace toroidal
}//namespace geo
}//namespace dg
