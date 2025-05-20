#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "dg/algorithm.h"
#include "solovev_parameters.h"
#include "magnetic_field.h"


/*!@file
 *
 * MagneticField objects
 */
namespace dg
{
namespace geo
{
/**
 * @brief Contains the Cerfon Taylor state type flux functions
 *
 * This is taken from A. J. Cerfon and M. O'Neil: Exact axisymmetric Taylor states for shaped plasmas, Physics of Plasmas 21, 064501 (2014)
 */
namespace taylor
{
///@addtogroup taylor
///@{
typedef dg::geo::solovev::Parameters Parameters; //!< bring Parameters into the taylor namespace

/**
 * @brief \f[ \psi \f]
 *
 * This is taken from A. J. Cerfon and M. O'Neil: Exact axisymmetric Taylor states for shaped plasmas, Physics of Plasmas 21, 064501 (2014)
 */
struct Psip : public aCylindricalFunctor<Psip>
{ /**
     * @brief Construct from given geometric parameters
     *
     * @param gp geometric parameters
     */
    Psip( solovev::Parameters gp): R0_(gp.R_0), c_(gp.c) {
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double do_compute(double R, double Z) const
    {
        double Rn = R/R0_, Zn = Z/R0_;
        double j1_c12 = std::cyl_bessel_j( 1, c_[11]*Rn);
        double y1_c12 = std::cyl_neumann(  1, c_[11]*Rn);
        double j1_cs = std::cyl_bessel_j( 1, cs_*Rn);
        double y1_cs = std::cyl_neumann(  1, cs_*Rn);
        return R0_*(
                   1.0*Rn*j1_c12
               + c_[0]*Rn*y1_c12
               + c_[1]*Rn*j1_cs*cos(c_[10]*Zn)
               + c_[2]*Rn*y1_cs*cos(c_[10]*Zn)
               + c_[3]*cos(c_[11]*sqrt(Rn*Rn+Zn*Zn))
               + c_[4]*cos(c_[11]*Zn)
               + c_[5]*Rn*j1_c12*Zn
               + c_[6]*Rn*y1_c12*Zn
               + c_[7]*Rn*j1_cs*sin(c_[10]*Zn)
               + c_[8]*Rn*y1_cs*sin(c_[10]*Zn)
               + c_[9]*sin(c_[11]*Zn));

    }
  private:
    double R0_, cs_;
    std::vector<double> c_;
};

/**
 * @brief \f[\psi_R\f]
 */
struct PsipR: public aCylindricalFunctor<PsipR>
{
    ///@copydoc Psip::Psip()
    PsipR( solovev::Parameters gp): R0_(gp.R_0), c_(gp.c) {
        cs_=sqrt(c_[11]*c_[11]-c_[10]*c_[10]);

    }
    double do_compute(double R, double Z) const
    {
        double Rn=R/R0_, Zn=Z/R0_;
        double j1_c12R = std::cyl_bessel_j(1, c_[11]*Rn) + c_[11]/2.*Rn*(
                std::cyl_bessel_j(0, c_[11]*Rn) - std::cyl_bessel_j(2,c_[11]*Rn));
        double y1_c12R = std::cyl_neumann(1, c_[11]*Rn) + c_[11]/2.*Rn*(
                std::cyl_neumann(0, c_[11]*Rn) - std::cyl_neumann(2,c_[11]*Rn));
        double j1_csR = std::cyl_bessel_j(1, cs_*Rn) + cs_/2.*Rn*(
                std::cyl_bessel_j(0, cs_*Rn) - std::cyl_bessel_j(2, cs_*Rn));
        double y1_csR = std::cyl_neumann(1, cs_*Rn) + cs_/2.*Rn*(
                std::cyl_neumann(0, cs_*Rn) - std::cyl_neumann(2, cs_*Rn));
        double RZbar = sqrt( Rn*Rn+Zn*Zn);
        double cosR = -c_[11]*Rn/RZbar*sin(c_[11]*RZbar);
        return  (
                   1.0*j1_c12R
               + c_[0]*y1_c12R
               + c_[1]*j1_csR*cos(c_[10]*Zn)
               + c_[2]*y1_csR*cos(c_[10]*Zn)
               + c_[3]*cosR
               + c_[5]*j1_c12R*Zn
               + c_[6]*y1_c12R*Zn
               + c_[7]*j1_csR*sin(c_[10]*Zn)
               + c_[8]*y1_csR*sin(c_[10]*Zn) );
    }
    private:
    double R0_, cs_;
    std::vector<double> c_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\f]
 */
struct PsipRR: public aCylindricalFunctor<PsipRR>
{
    ///@copydoc Psip::Psip()
    PsipRR( solovev::Parameters gp ): R0_(gp.R_0), c_(gp.c) {
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double do_compute(double R, double Z) const
    {
        double Rn=R/R0_, Zn=Z/R0_;
        double j1_c12R = c_[11]*(std::cyl_bessel_j(0, c_[11]*Rn) - Rn*c_[11]*std::cyl_bessel_j(1, c_[11]*Rn));
        double y1_c12R = c_[11]*(std::cyl_neumann( 0, c_[11]*Rn) - Rn*c_[11]*std::cyl_neumann(1, c_[11]*Rn));
        double j1_csR = cs_*(std::cyl_bessel_j(0, cs_*Rn) - Rn*cs_*std::cyl_bessel_j(1, cs_*Rn));
        double y1_csR = cs_*(std::cyl_neumann( 0, cs_*Rn) - Rn*cs_*std::cyl_neumann( 1, cs_*Rn));
        double RZbar = sqrt(Rn*Rn+Zn*Zn);
        double cosR = -c_[11]/(RZbar*RZbar)*(c_[11]*Rn*Rn*cos(c_[11]*RZbar) +Zn*Zn*sin(c_[11]*RZbar)/RZbar);
        return  1./R0_*(
                   1.0*j1_c12R
               + c_[0]*y1_c12R
               + c_[1]*j1_csR*cos(c_[10]*Zn)
               + c_[2]*y1_csR*cos(c_[10]*Zn)
               + c_[3]*cosR
               + c_[5]*j1_c12R*Zn
               + c_[6]*y1_c12R*Zn
               + c_[7]*j1_csR*sin(c_[10]*Zn)
               + c_[8]*y1_csR*sin(c_[10]*Zn) );
    }
  private:
    double R0_, cs_;
    std::vector<double> c_;
};
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\f]
 */
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
    ///@copydoc Psip::Psip()
    PsipZ( solovev::Parameters gp ): R0_(gp.R_0), c_(gp.c) {
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double do_compute(double R, double Z) const
    {
        double Rn = R/R0_, Zn = Z/R0_;
        double j1_c12 = std::cyl_bessel_j( 1, c_[11]*Rn);
        double y1_c12 = std::cyl_neumann(  1, c_[11]*Rn);
        double j1_cs = std::cyl_bessel_j( 1, cs_*Rn);
        double y1_cs = std::cyl_neumann(  1, cs_*Rn);
        return (
               - c_[1]*Rn*j1_cs*c_[10]*sin(c_[10]*Zn)
               - c_[2]*Rn*y1_cs*c_[10]*sin(c_[10]*Zn)
               - c_[3]*c_[11]*Zn/sqrt(Rn*Rn+Zn*Zn)*sin(c_[11]*sqrt(Rn*Rn+Zn*Zn))
               - c_[4]*c_[11]*sin(c_[11]*Zn)
               + c_[5]*Rn*j1_c12
               + c_[6]*Rn*y1_c12
               + c_[7]*Rn*j1_cs*c_[10]*cos(c_[10]*Zn)
               + c_[8]*Rn*y1_cs*c_[10]*cos(c_[10]*Zn)
               + c_[9]*c_[11]*cos(c_[11]*Zn));
    }
    private:
    double R0_,cs_;
    std::vector<double> c_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}\f]
 */
struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    ///@copydoc Psip::Psip()
    PsipZZ( solovev::Parameters gp): R0_(gp.R_0), c_(gp.c) {
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double do_compute(double R, double Z) const
    {
        double Rn = R/R0_, Zn = Z/R0_;
        double j1_cs = std::cyl_bessel_j( 1, cs_*Rn);
        double y1_cs = std::cyl_neumann(  1, cs_*Rn);
        double RZbar = sqrt(Rn*Rn+Zn*Zn);
        double cosZ = -c_[11]/(RZbar*RZbar)*(c_[11]*Zn*Zn*cos(c_[11]*RZbar) +Rn*Rn*sin(c_[11]*RZbar)/RZbar);
        return 1./R0_*(
               - c_[1]*Rn*j1_cs*c_[10]*c_[10]*cos(c_[10]*Zn)
               - c_[2]*Rn*y1_cs*c_[10]*c_[10]*cos(c_[10]*Zn)
               + c_[3]*cosZ
               - c_[4]*c_[11]*c_[11]*cos(c_[11]*Zn)
               - c_[7]*Rn*j1_cs*c_[10]*c_[10]*sin(c_[10]*Zn)
               - c_[8]*Rn*y1_cs*c_[10]*c_[10]*sin(c_[10]*Zn)
               - c_[9]*c_[11]*c_[11]*sin(c_[11]*Zn));
    }
    private:
    double R0_, cs_;
    std::vector<double> c_;
};
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\f]
 */
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
    ///@copydoc Psip::Psip()
    PsipRZ( solovev::Parameters gp ): R0_(gp.R_0), c_(gp.c) {
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double do_compute(double R, double Z) const
    {
        double Rn=R/R0_, Zn=Z/R0_;
        double j1_c12R = std::cyl_bessel_j(1, c_[11]*Rn) + c_[11]/2.*Rn*(
                std::cyl_bessel_j(0, c_[11]*Rn) - std::cyl_bessel_j(2,c_[11]*Rn));
        double y1_c12R = std::cyl_neumann( 1, c_[11]*Rn) + c_[11]/2.*Rn*(
                std::cyl_neumann( 0, c_[11]*Rn) - std::cyl_neumann( 2,c_[11]*Rn));
        double j1_csR = std::cyl_bessel_j(1, cs_*Rn) + cs_/2.*Rn*(
                std::cyl_bessel_j(0, cs_*Rn) - std::cyl_bessel_j(2, cs_*Rn));
        double y1_csR = std::cyl_neumann( 1, cs_*Rn) + cs_/2.*Rn*(
                std::cyl_neumann( 0, cs_*Rn) - std::cyl_neumann(2, cs_*Rn));
        double RZbar = sqrt(Rn*Rn+Zn*Zn);
        double cosRZ = -c_[11]*Rn*Zn/(RZbar*RZbar*RZbar)*( c_[11]*RZbar*cos(c_[11]*RZbar) -sin(c_[11]*RZbar) );
        return  1./R0_*(
               - c_[1]*j1_csR*c_[10]*sin(c_[10]*Zn)
               - c_[2]*y1_csR*c_[10]*sin(c_[10]*Zn)
               + c_[3]*cosRZ
               + c_[5]*j1_c12R
               + c_[6]*y1_c12R
               + c_[7]*j1_csR*c_[10]*cos(c_[10]*Zn)
               + c_[8]*y1_csR*c_[10]*cos(c_[10]*Zn) );
    }
  private:
    double R0_, cs_;
    std::vector<double> c_;
};

/**
 * @brief \f[\hat{I} = c_{12}\psi\f]
 *
   \f[\hat{I}= \sqrt{-2 A \hat{\psi}_p / \hat{R}_0 +1}\f]
 */
struct Ipol: public aCylindricalFunctor<Ipol>
{
    ///@copydoc Psip::Psip()
    Ipol(  solovev::Parameters gp ): c12_(gp.c[11]), psip_(gp) { }
    double do_compute(double R, double Z) const
    {
        return c12_*psip_(R,Z);

    }
  private:
    double c12_;
    Psip psip_;
};
/**
 * @brief \f[\hat I_R\f]
 */
struct IpolR: public aCylindricalFunctor<IpolR>
{
    ///@copydoc Psip::Psip()
    IpolR(  solovev::Parameters gp ): c12_(gp.c[11]), psipR_(gp) { }
    double do_compute(double R, double Z) const
    {
        return c12_*psipR_(R,Z);
    }
  private:
    double c12_;
    PsipR psipR_;
};
/**
 * @brief \f[\hat I_Z\f]
 */
struct IpolZ: public aCylindricalFunctor<IpolZ>
{
    ///@copydoc Psip::Psip()
    IpolZ(  solovev::Parameters gp ): c12_(gp.c[11]), psipZ_(gp) { }
    double do_compute(double R, double Z) const
    {
        return c12_*psipZ_(R,Z);
    }
  private:
    double c12_;
    PsipZ psipZ_;
};

inline CylindricalFunctorsLvl2 createPsip( solovev::Parameters gp)
{
    return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
inline CylindricalFunctorsLvl1 createIpol( solovev::Parameters gp)
{
    return CylindricalFunctorsLvl1( Ipol(gp), IpolR(gp), IpolZ(gp));
}

///@}

} //namespace taylor
/**
 * @brief Create a Taylor Magnetic field

 * Based on \c dg::geo::taylor::Psip(gp) and \c dg::geo::taylor::Ipol(gp)
 * @param gp Solovev parameters
 * @return A magnetic field object
 * @ingroup taylor
 */
inline dg::geo::TokamakMagneticField createTaylorField( dg::geo::solovev::Parameters gp)
{
    MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
            equilibrium::solovev, modifier::none, str2description.at( gp.description)};
    return TokamakMagneticField( gp.R_0, dg::geo::taylor::createPsip(gp), dg::geo::taylor::createIpol(gp), params);
}
} //namespace geo

}//namespace dg

