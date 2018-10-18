#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "guenther_parameters.h"
#include "magnetic_field.h"

//TODO somebody document the functions as in solovev/geometry.h

/*!@file
 *
 * MagneticField objects
 */
namespace dg
{
namespace geo
{
/**
 * @brief Contains the Guenther type flux functions
 */
namespace guenther
{
///@addtogroup guenther
///@{


/**
 * @brief \f[\cos(\pi(R-R_0)/2)\cos(\pi Z/2)\f]
 */
struct Psip : public aCloneableBinaryFunctor<Psip>
{
    Psip(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {
        return cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi\sin(\pi(R-R_0)/2)\cos(\pi Z/2)/2\f]
 */
struct PsipR : public aCloneableBinaryFunctor<PsipR>
{
    PsipR(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {
        return -M_PI*0.5*sin(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi^2\cos(\pi(R-R_0)/2)\cos(\pi Z/2)/4\f]
 */
struct PsipRR : public aCloneableBinaryFunctor<PsipRR>
{
    PsipRR(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {
        return -M_PI*M_PI*0.25*cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi\cos(\pi(R-R_0)/2)\sin(\pi Z/2)/2\f]
 */
struct PsipZ : public aCloneableBinaryFunctor<PsipZ>

{
    PsipZ(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {
        return -M_PI*0.5*cos(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi^2\cos(\pi(R-R_0)/2)\cos(\pi Z/2)/4\f]
 */
struct PsipZZ : public aCloneableBinaryFunctor<PsipZZ>
{
    PsipZZ(double R_0 ):   R_0(R_0){}
  private:
    double do_compute(double R, double Z) const
    {
        return -M_PI*M_PI*0.25*cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[ \pi^2\sin(\pi(R-R_0)/2)\sin(\pi Z/2)/4\f]
 */
struct PsipRZ : public aCloneableBinaryFunctor<PsipRZ>
{
    PsipRZ(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {
        return M_PI*M_PI*0.25*sin(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5);
    }
    double R_0;
};

/**
 * @brief \f[ I_0\f]
 */
struct Ipol : public aCloneableBinaryFunctor<Ipol>
{
    Ipol( double I_0):   I_0(I_0) {}
    private:
    double do_compute(double R, double Z) const { return I_0; }
    double I_0;
};
/**
 * @brief \f[0\f]
 */
struct IpolR : public aCloneableBinaryFunctor<IpolR>
{
    IpolR(  ) {}
    private:
    double do_compute(double R, double Z) const { return 0; }
};
/**
 * @brief \f[0\f]
 */
struct IpolZ : public aCloneableBinaryFunctor<IpolZ>
{
    IpolZ(  ) {}
    private:
    double do_compute(double R, double Z) const { return 0; }
};

static inline BinaryFunctorsLvl2 createPsip( double R_0)
{
    BinaryFunctorsLvl2 psip( new Psip(R_0), new PsipR(R_0), new PsipZ(R_0),new PsipRR(R_0), new PsipRZ(R_0), new PsipZZ(R_0));
    return psip;
}
static inline BinaryFunctorsLvl1 createIpol( double I_0)
{
    BinaryFunctorsLvl1 ipol( new Ipol(I_0), new IpolR(), new IpolZ());
    return ipol;
}
static inline TokamakMagneticField createMagField( double R_0, double I_0)
{
    return TokamakMagneticField( R_0, createPsip(R_0), createIpol(I_0));
}
///@}
} //namespace guenther

/**
 * @brief Create a Guenther Magnetic field

 * \f[\psi_p(R,Z) = \cos(\pi(R-R_0)/2)\cos(\pi Z/2),\quad I(\psi_p) = I_0\f]
 * @param R_0 the major radius
 * @param I_0 the current
 * @return A magnetic field object
 * @ingroup geom
 */
static inline dg::geo::TokamakMagneticField createGuentherField( double R_0, double I_0)
{
    return TokamakMagneticField( R_0, guenther::createPsip(R_0), guenther::createIpol(I_0));
}
} //namespace geo
}//namespace dg
