#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/algorithm.h"
#include "solovev_parameters.h"
#include "magnetic_field.h"
#include "modified.h"


/*!@file
 *
 * MagneticField objects
 */
namespace dg
{
namespace geo
{
/**
 * @brief Contains the solovev state type flux function
 *
 * A.J. Cerfon and J.P. Freidberg: "One size fits all" analytic solutions to the Grad-Shafraonv equation, Physics of Plasmas 17, 032502 (2010)
 */
namespace solovev
{
///@addtogroup solovev
///@{

/**
 * @brief \f[ \hat{\psi}_p  \f]
 *
 * \f[ \hat{\psi}_p(R,Z) =
      \hat{R}_0P_{\psi}\Bigg\{\bar{R}^4/8 + A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
      + \sum_{i=1}^{12} c_i\bar \psi_{pi}(\bar R, \bar Z) \Bigg\}
      =
      \hat{R}_0P_{\psi}\Bigg\{\bar{R}^4/8 + A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
      +c_1+
      c_2 \bar{R}^2 +
      c_3 \left[  \bar{Z}^2-\bar{R}^2  \ln{(\bar{R}   )} \right] +
      c_4 \left[\bar{R}^4 -4 \bar{R}^2  \bar{Z}^2 \right] +
      c_5 \left[3 \bar{R}^4  \ln{(\bar{R}   )}-9 \bar{R}^2  \bar{Z}^2-12 \bar{R}^2  \bar{Z}^2
      \ln{(\bar{R}   )}+2  \bar{Z}^4\right]+
      c_6 \left[ \bar{R}^6 -12 \bar{R}^4  \bar{Z}^2+8 \bar{R}^2  \bar{Z}^4 \right]  +
      c_7 \left[-(15 \bar{R}^6  \ln{(\bar{R}   )})+75 \bar{R}^4  \bar{Z}^2+180 \bar{R}^4
       \bar{Z}^2 \ln{(\bar{R}   )}-140 \bar{R}^2  \bar{Z}^4-120 \bar{R}^2  \bar{Z}^4
      \ln{(\bar{R}   )}+8  \bar{Z}^6 \right] +
      c_8  \bar{Z}+c_9 \bar{R}^2  \bar{Z}+(\bar{R}^4 )/8 +
      c_{10} \left[ \bar{Z}^3-3 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )}\right]+
      c_{11} \left[3 \bar{R}^4  \bar{Z}-4 \bar{R}^2  \bar{Z}^3\right]      +
      c_{12} \left[-(45 \bar{R}^4  \bar{Z})+60 \bar{R}^4  \bar{Z} \ln{(\bar{R}   )}-
      80 \bar{R}^2  \bar{Z}^3 \ln{(\bar{R}   )}+8  \bar{Z}^5 \right]
      \Bigg\} \f]
      with \f$ \bar R := \frac{ R}{R_0} \f$ and \f$\bar Z := \frac{Z}{R_0}\f$
 *
 */
struct Psip: public aCylindricalFunctor<Psip>
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp geometric parameters
     */
    Psip( const Parameters& gp ): m_R0(gp.R_0), mA(gp.A), m_pp(gp.pp), mc(gp.c) {
        m_prev = std::make_shared< std::array<double,3>>(std::array<double,3>{ 0,0,0});
    }
    double do_compute(double R, double Z) const
    {
        // Optimization rationale: The way we compute magnetic field terms
        // through the TokamakMagneticField class and e.g. the BHatR class
        // leads to repeated evaluations of Psip, PsipR, etc. at the same
        // point. We thus let the Psip classes remember the result of the
        // previous call to avoid recomputing the same point. Since the
        // different calls are made through different copies of Psip we need to
        // store the previous results in a shared_ptr such that all copies of
        // Psip have access to it
        if( R == (*m_prev)[0] && Z == (*m_prev)[1])
            return (*m_prev)[2];

        double Rn = R / m_R0, Rn2 = Rn * Rn, lgRn = log(Rn);
        double Zn = Z/m_R0;
        // Copied from Mathematica ...
        (*m_prev)[2] =   m_R0*m_pp*(mc[0] + Rn2 * ((lgRn * mA) / 2. + mc[1] +
            Zn * (mc[8] + Zn *
                (-4 * mc[3] - 9 * mc[4] +
                    Zn * (Zn * (8 * mc[5] - 140 * mc[6]) -
                        4 * mc[10]))) +
            lgRn * (-mc[2] +
                Zn * (-3 * mc[9] +
                    Zn * (-12 * mc[4] +
                        Zn * (-120 * Zn * mc[6] - 80 * mc[11]))
                    )) + Rn2 *
            (0.125 - mA / 8. + mc[3] +
                Rn2 * (mc[5] - 15 * lgRn * mc[6]) +
                Zn * (Zn * (-12 * mc[5] + 75 * mc[6]) +
                    3 * mc[10] - 45 * mc[11]) +
                lgRn * (3 * mc[4] +
                    Zn * (180 * Zn * mc[6] + 60 * mc[11]))))\
            + Zn * (mc[7] + Zn *
                (mc[2] + Zn *
                    (mc[9] +
                        Zn * (2 * mc[4] +
                            Zn * (8 * Zn * mc[6] + 8 * mc[11])))))
                      );
        (*m_prev)[0] = R, (*m_prev)[1] = Z;
        return (*m_prev)[2];
    }
  private:
    double m_R0, mA, m_pp;
    std::vector<double> mc;
    std::shared_ptr<std::array<double,3>> m_prev;
};

/**
 * @brief \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} \f]
 *
 * \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} =
      P_\psi \Bigg\{ 2 c_2 \bar{R} +(\bar{R}^3 )/2+2 c_9 \bar{R}  \bar{Z}
      +c_4 (4 \bar{R}^3 -8 \bar{R}  \bar{Z}^2)+c_{11}
      (12  \bar{R}^3  \bar{Z}-8 \bar{R}  \bar{Z}^3
      +c_6 (6 \bar{R}^5 -48 \bar{R}^3  \bar{Z}^2+16 \bar{R}  \bar{Z}^4)+c_3 (-\bar{R} -2 \bar{R}  \ln{(\bar{R}   )})+
      A ((\bar{R} )/2-(\bar{R}^3 )/2+\bar{R}  \ln{(\bar{R}   )})
      +c_{10} (-3 \bar{R}  \bar{Z}-6 \bar{R}  \bar{Z} \ln{(\bar{R}   )})+c_5 (3 \bar{R}^3 -30 \bar{R}
      \bar{Z}^2+12 \bar{R}^3  \ln{(\bar{R}   )}-24 \bar{R}  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_{12} (-120 \bar{R}^3  \bar{Z}-80 \bar{R}  \bar{Z}^3+240 \bar{R}^3  \bar{Z}
      \ln{(\bar{R}   )}-160 \bar{R}  \bar{Z}^3 \ln{(\bar{R}   )})
      +c_7 (-15 \bar{R}^5 +480 \bar{R}^3  \bar{Z}^2-400 \bar{R}  \bar{Z}^4-90 \bar{R}^5
      \ln{(\bar{R}   )}+720 \bar{R}^3  \bar{Z}^2 \ln{(\bar{R}   )}-240 \bar{R}  \bar{Z}^4
      \ln{(\bar{R}   )})\Bigg\} \f]
      with \f$ \bar R := \frac{ R}{R_0} \f$ and \f$\bar Z := \frac{Z}{R_0}\f$
 */
struct PsipR: public aCylindricalFunctor<PsipR>
{
    ///@copydoc Psip::Psip()
    PsipR( const Parameters& gp ): m_R0(gp.R_0), mA(gp.A), m_pp(gp.pp), mc(gp.c) {
        m_prev = std::make_shared< std::array<double,3>>(std::array<double,3>{ 0,0,0});
    }
    double do_compute(double R, double Z) const
    {
        if( R == (*m_prev)[0] && Z == (*m_prev)[1])
            return (*m_prev)[2];
        double Rn = R / m_R0, Rn2 = Rn * Rn, lgRn = log(Rn);
        double Zn = Z / m_R0;
        // Copied from Mathematica ...
        (*m_prev)[2] = m_pp * (mA * Rn * (0.5 + lgRn - Rn2 / 2.) +
            Rn * (2 * mc[1] - mc[2] -
                8 * Zn * Zn * mc[3] +
                lgRn * (-2 * mc[2] +
                    Zn * (-6 * mc[9] +
                        Zn * (-24 * mc[4] +
                            Zn * (-240 * Zn * mc[6] - 160 * mc[11])
                            ))) +
                Zn * (2 * mc[8] - 3 * mc[9] +
                    Zn * (-30 * mc[4] +
                        Zn * (Zn * (16 * mc[5] - 400 * mc[6]) -
                            8 * mc[10] - 80 * mc[11]))) +
                Rn2 * (0.5 + 4 * mc[3] + 3 * mc[4] +
                    Rn2 * (6 * mc[5] - 15 * mc[6] -
                        90 * lgRn * mc[6]) +
                    Zn * (Zn * (-48 * mc[5] + 480 * mc[6]) +
                        12 * mc[10] - 120 * mc[11]) +
                    lgRn * (12 * mc[4] +
                        Zn * (720 * Zn * mc[6] + 240 * mc[11]))))
            );
        (*m_prev)[0] = R, (*m_prev)[1] = Z;
        return (*m_prev)[2];
    }
  private:
    double m_R0, mA, m_pp;
    std::vector<double> mc;
    std::shared_ptr<std::array<double,3>> m_prev;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\f]
 *
 * \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}=
     \hat{R}_0^{-1} P_\psi \Bigg\{ 2 c_2 +(3 \hat{\bar{R}}^2 )/2+2 c_9  \bar{Z}+c_4 (12 \bar{R}^2 -8  \bar{Z}^2)+c_{11}
      (36 \bar{R}^2  \bar{Z}-8  \bar{Z}^3)
      +c_6 (30 \bar{R}^4 -144 \bar{R}^2  \bar{Z}^2+16  \bar{Z}^4)+c_3 (-3 -2  \ln{(\bar{R}
      )})+
      A ((3 )/2-(3 \bar{R}^2 )/2+ \ln{(\bar{R}   )})
      +c_{10} (-9  \bar{Z}-6
      \bar{Z} \ln{(\bar{R}   )})+c_5 (21 \bar{R}^2 -54
      \bar{Z}^2+36 \bar{R}^2  \ln{(\bar{R}   )}-24  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_{12} (-120 \bar{R}^2  \bar{Z}-240  \bar{Z}^3+720 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )}
      -160  \bar{Z}^3 \ln{(\bar{R}   )})
      + c_7 (-165 \bar{R}^4 +2160 \bar{R}^2  \bar{Z}^2-640  \bar{Z}^4-450 \bar{R}^4  \ln{(\bar{R}   )}+2160 \bar{R}^2  \bar{Z}^2
      \ln{(\bar{R}   )}-240  \bar{Z}^4 \ln{(\bar{R}   )})\Bigg\}\f]
 */
struct PsipRR: public aCylindricalFunctor<PsipRR>
{
    ///@copydoc Psip::Psip()
    PsipRR( const Parameters& gp ): m_R0(gp.R_0), mA(gp.A), m_pp(gp.pp), mc(gp.c) {
        m_prev = std::make_shared< std::array<double,3>>(std::array<double,3>{ 0,0,0});
    }
    double do_compute(double R, double Z) const
    {
        if( R == (*m_prev)[0] && Z == (*m_prev)[1])
            return (*m_prev)[2];
        double Rn = R / m_R0, Rn2 = Rn * Rn, lgRn = log(Rn);
        double Zn = Z / m_R0;
        // Copied from Mathematica ...
        (*m_prev)[2] = m_pp/m_R0 * (mA * (1.5 + lgRn - (3 * Rn2) / 2.) + 2 * mc[1] -
            3 * mc[2] - 8 * Zn * Zn * mc[3] +
            lgRn * (-2 * mc[2] +
                Zn * (-6 * mc[9] +
                    Zn * (-24 * mc[4] +
                        Zn * (-240 * Zn * mc[6] - 160 * mc[11]))))
            + Zn * (2 * mc[8] - 9 * mc[9] +
                Zn * (-54 * mc[4] +
                    Zn * (Zn * (16 * mc[5] - 640 * mc[6]) -
                        8 * mc[10] - 240 * mc[11]))) +
            Rn2 * (1.5 + 12 * mc[3] + 21 * mc[4] +
                Rn2 * (30 * mc[5] - 165 * mc[6] -
                    450 * lgRn * mc[6]) +
                Zn * (Zn * (-144 * mc[5] + 2160 * mc[6]) +
                    36 * mc[10] - 120 * mc[11]) +
                lgRn * (36 * mc[4] +
                    Zn * (2160 * Zn * mc[6] + 720 * mc[11])))
            );
        (*m_prev)[0] = R, (*m_prev)[1] = Z;
        return (*m_prev)[2];
    }
  private:
    double m_R0, mA, m_pp;
    std::vector<double> mc;
    std::shared_ptr<std::array<double,3>> m_prev;
};
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\f]
 *
 * \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}=
      P_\psi \Bigg\{c_8 +c_9 \bar{R}^2 +2 c_3  \bar{Z}-8 c_4 \bar{R}^2  \bar{Z}+c_{11}
      (3 \bar{R}^4 -12 \bar{R}^2  \bar{Z}^2)+c_6 (-24 \bar{R}^4  \bar{Z}+32 \bar{R}^2  \bar{Z}^3)
      +c_{10} (3  \bar{Z}^2-3 \bar{R}^2
      \ln{(\bar{R}   )})+c_5 (-18 \bar{R}^2  \bar{Z}+8  \bar{Z}^3-24 \bar{R}^2  \bar{Z}
      \ln{(\bar{R}   )})
      +c_{12} (-45 \bar{R}^4 +40  \bar{Z}^4+
      60 \bar{R}^4  \ln{(\bar{R}   )}-240 \bar{R}^2  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_7 (150 \bar{R}^4  \bar{Z}-560 \bar{R}^2  \bar{Z}^3+48
      \bar{Z}^5+360 \bar{R}^4  \bar{Z} \ln{(\bar{R}   )}-480 \bar{R}^2  \bar{Z}^3 \ln{(\bar{R}   )})\Bigg\} \f]
 */
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
    ///@copydoc Psip::Psip()
    PsipZ( const Parameters& gp ): m_R0(gp.R_0), m_pp(gp.pp), mc(gp.c) {
        m_prev = std::make_shared< std::array<double,3>>(std::array<double,3>{ 0,0,0});
    }
    double do_compute(double R, double Z) const
    {
        if( R == (*m_prev)[0] && Z == (*m_prev)[1])
            return (*m_prev)[2];
        double Rn = R / m_R0, Rn2 = Rn * Rn, lgRn = log(Rn);
        double Zn = Z / m_R0;
        // Copied from Mathematica ...
        (*m_prev)[2] = m_pp * (mc[7] + Rn2 * (mc[8] - 3 * lgRn * mc[9] +
            Rn2 * (3 * mc[10] - 45 * mc[11] +
                60 * lgRn * mc[11])) +
            Zn * (2 * mc[2] + Rn2 *
                (-8 * mc[3] + (-18 - 24 * lgRn) * mc[4] +
                    Rn2 * (-24 * mc[5] + 150 * mc[6] +
                        360 * lgRn * mc[6])) +
                Zn * (3 * mc[9] +
                    Rn2 * (-12 * mc[10] - 240 * lgRn * mc[11]) +
                    Zn * (8 * mc[4] +
                        Rn2 * (32 * mc[5] - 560 * mc[6] -
                            480 * lgRn * mc[6]) +
                        Zn * (48 * Zn * mc[6] + 40 * mc[11]))))
            );
        (*m_prev)[0] = R, (*m_prev)[1] = Z;
        return (*m_prev)[2];

    }
  private:
    double m_R0, m_pp;
    std::vector<double> mc;
    std::shared_ptr<std::array<double,3>> m_prev;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}\f]

   \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}=
      \hat{R}_0^{-1} P_\psi \Bigg\{2 c_3 -8 c_4 \bar{R}^2 +6 c_{10}  \bar{Z}-24 c_{11}
      \bar{R}^2  \bar{Z}+c_6 (-24 \bar{R}^4 +96 \bar{R}^2  \bar{Z}^2)
      +c_5 (-18 \bar{R}^2 +24  \bar{Z}^2-24 \bar{R}^2  \ln{(\bar{R}   )})+
      c_{12} (160  \bar{Z}^3-480 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )})
      +c_7 (150 \bar{R}^4 -1680 \bar{R}^2  \bar{Z}^2+240  \bar{Z}^4+360 \bar{R}^4
      \ln{(\bar{R}   )}-1440 \bar{R}^2  \bar{Z}^2 \ln{(\bar{R}   )})\Bigg\} \f]
 */
struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    ///@copydoc Psip::Psip()
    PsipZZ( const Parameters& gp): m_R0(gp.R_0), m_pp(gp.pp), mc(gp.c) {
        m_prev = std::make_shared< std::array<double,3>>(std::array<double,3>{ 0,0,0});
    }
    double do_compute(double R, double Z) const
    {
        if( R == (*m_prev)[0] && Z == (*m_prev)[1])
            return (*m_prev)[2];
        double Rn = R / m_R0, Rn2 = Rn * Rn, lgRn = log(Rn);
        double Zn = Z / m_R0;
        // Copied from Mathematica ...
        (*m_prev)[2] = m_pp/m_R0 * (2 * mc[2] + 24 * Zn * Zn * mc[4] +
            Zn * (6 * mc[9] + Zn * Zn *
                (240 * Zn * mc[6] + 160 * mc[11])) +
            Rn2 * (-8 * mc[3] + (-18 - 24 * lgRn) * mc[4] +
                Rn2 * (-24 * mc[5] + 150 * mc[6] +
                    360 * lgRn * mc[6]) +
                Zn * (Zn * (96 * mc[5] - 1680 * mc[6] -
                    1440 * lgRn * mc[6]) - 24 * mc[10] -
                    480 * lgRn * mc[11]))
            );
        (*m_prev)[0] = R, (*m_prev)[1] = Z;
        return (*m_prev)[2];
    }
  private:
    double m_R0, m_pp;
    std::vector<double> mc;
    std::shared_ptr<std::array<double,3>> m_prev;
};
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\f]

  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}=
        \hat{R}_0^{-1} P_\psi \Bigg\{2 c_9 \bar{R} -16 c_4 \bar{R}  \bar{Z}+c_{11}
      (12 \bar{R}^3 -24 \bar{R}  \bar{Z}^2)+c_6 (-96 \bar{R}^3  \bar{Z}+64 \bar{R}  \bar{Z}^3)
      + c_{10} (-3 \bar{R} -6 \bar{R}  \ln{(\bar{R}   )})
      +c_5 (-60 \bar{R}  \bar{Z}-48 \bar{R}  \bar{Z} \ln{(\bar{R}   )})
      +c_{12}  (-120 \bar{R}^3 -240 \bar{R}  \bar{Z}^2+
      240 \bar{R}^3  \ln{(\bar{R}   )}-480 \bar{R}  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_7(960 \bar{R}^3  \bar{Z}-1600 \bar{R}  \bar{Z}^3+1440 \bar{R}^3  \bar{Z} \ln{(\bar{R}
      )}-960 \bar{R}  \bar{Z}^3 \ln{(\bar{R}   )})\Bigg\} \f]
 */
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
    ///@copydoc Psip::Psip()
    PsipRZ( const Parameters& gp ): m_R0(gp.R_0), m_pp(gp.pp), mc(gp.c) {
        m_prev = std::make_shared< std::array<double,3>>(std::array<double,3>{ 0,0,0});
    }
    double do_compute(double R, double Z) const
    {
        if( R == (*m_prev)[0] && Z == (*m_prev)[1])
            return (*m_prev)[2];
        double Rn = R / m_R0, Rn2 = Rn * Rn, lgRn = log(Rn);
        double Zn = Z / m_R0;
        // Copied from Mathematica ...
        (*m_prev)[2] = m_pp/m_R0 * (Rn * (2 * mc[8] - 3 * mc[9] - 6 * lgRn * mc[9] +
            Rn2 * (Zn * (-96 * mc[5] + 960 * mc[6] +
                1440 * lgRn * mc[6]) + 12 * mc[10] -
                120 * mc[11] + 240 * lgRn * mc[11]) +
            Zn * (-16 * mc[3] + (-60 - 48 * lgRn) * mc[4] +
                Zn * (Zn * (64 * mc[5] - 1600 * mc[6] -
                    960 * lgRn * mc[6]) - 24 * mc[10] -
                    240 * mc[11] - 480 * lgRn * mc[11])))
            );
        (*m_prev)[0] = R, (*m_prev)[1] = Z;
        return (*m_prev)[2];
    }
  private:
    double m_R0, m_pp;
    std::vector<double> mc;
    std::shared_ptr<std::array<double,3>> m_prev;
};

/**
 * @brief \f[\hat{I}\f]

    \f[\hat{I}= P_I \sqrt{-2 A \hat{\psi}_p / \hat{R}_0/P_\psi +1}\f]
 */
struct Ipol: public aCylindricalFunctor<Ipol>
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp geometric parameters (for R_0, A, PP and PI)
     * @param psip the flux function to use
     */
    Ipol( const Parameters &gp, const std::function<double(double,double)>& psip ):  m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_pi(gp.pi), m_psip(psip) {
        if( gp.pp == 0.)
            m_pp = 1.; //safety measure to avoid divide by zero errors
    }
    double do_compute(double R, double Z) const
    {
        return m_pi*sqrt(-2.*m_A* m_psip(R,Z) /m_R0/m_pp + 1.);
    }
  private:
    double m_R0, m_A, m_pp, m_pi;
    std::function<double(double,double)> m_psip;
};
/**
 * @brief \f[\hat I_R\f]
 */
struct IpolR: public aCylindricalFunctor<IpolR>
{
    /**
     * @copydoc Ipol::Ipol()
     * @param psipR the R-derivative of the flux function to use
     */
    IpolR( const Parameters& gp, const std::function<double(double,double)>& psip, std::function<double(double,double)> psipR ):
        m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_pi(gp.pi), m_psip(psip), m_psipR(psipR) {
        if( gp.pp == 0.)
            m_pp = 1.; //safety measure to avoid divide by zero errors
    }
    double do_compute(double R, double Z) const
    {
        return -m_pi/sqrt(-2.*m_A* m_psip(R,Z) /m_R0/m_pp + 1.)*(m_A*m_psipR(R,Z)/m_R0/m_pp);
    }
  private:
    double m_R0, m_A, m_pp, m_pi;
    std::function<double(double,double)> m_psip, m_psipR;
};
/**
 * @brief \f[\hat I_Z\f]
 */
struct IpolZ: public aCylindricalFunctor<IpolZ>
{
    /**
     * @copydoc Ipol::Ipol()
     * @param psipZ the Z-derivative of the flux function to use
     */
    IpolZ(  const Parameters& gp, const std::function<double(double,double)>& psip, std::function<double(double,double)> psipZ ):
        m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_pi(gp.pi), m_psip(psip), m_psipZ(psipZ) {
        if( gp.pp == 0.)
            m_pp = 1.; //safety measure to avoid divide by zero errors
    }
    double do_compute(double R, double Z) const
    {
        return -m_pi/sqrt(-2.*m_A* m_psip(R,Z) /m_R0/m_pp + 1.)*(m_A*m_psipZ(R,Z)/m_R0/m_pp);
    }
  private:
    double m_R0, m_A, m_pp, m_pi;
    std::function<double(double,double)> m_psip, m_psipZ;
};

inline dg::geo::CylindricalFunctorsLvl2 createPsip( const Parameters& gp)
{
    return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),
        PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
inline dg::geo::CylindricalFunctorsLvl1 createIpol( const Parameters& gp, const CylindricalFunctorsLvl1& psip)
{
    return CylindricalFunctorsLvl1(
            solovev::Ipol(gp, psip.f()),
            solovev::IpolR(gp,psip.f(), psip.dfx()),
            solovev::IpolZ(gp,psip.f(), psip.dfy()));
}

///@}
///////////////////////////////////////introduce fields into solovev namespace

} //namespace solovev

/**
 * @brief Create a Solovev Magnetic field
 *
 * Based on \c dg::geo::solovev::Psip(gp) and \c dg::geo::solovev::Ipol(gp)
 * @param gp Solovev parameters
 * @return A magnetic field object
 * @ingroup solovev
 */
inline dg::geo::TokamakMagneticField createSolovevField(
    const dg::geo::solovev::Parameters& gp)
{
    MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
            equilibrium::solovev, modifier::none, str2description.at( gp.description)};
    auto psip = solovev::createPsip(gp); // make sure prev is shared
    return TokamakMagneticField( gp.R_0, psip,
        solovev::createIpol(gp, psip), params);
}
/**
 * @brief DEPRECATED Create a modified Solovev Magnetic field
 *
 * Based on \c dg::geo::mod::Psip(gp) and
 * \c dg::geo::solovev::Ipol(gp)
 * We modify psi above a certain value to a constant using the
 * \c dg::IPolynomialHeaviside function (an approximation to the integrated Heaviside
 * function with width alpha), i.e. we replace psi with IPolynomialHeaviside(psi).
 * This subsequently modifies all derivatives of psi and the poloidal
 * current.
 * @param gp Solovev parameters
 * @param psi0 boundary value where psi is modified to a constant psi0
 * @param alpha radius of the transition region where the modification acts (smaller is quicker)
 * @param sign determines which side of Psi to dampen (negative or positive, forwarded to \c dg::IPolynomialHeaviside)
 * @return A magnetic field object
 * @ingroup solovev
 */
inline dg::geo::TokamakMagneticField createModifiedSolovevField(
    const dg::geo::solovev::Parameters& gp, double psi0, double alpha, double sign = -1)
{
    MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
            equilibrium::solovev, modifier::heaviside, str2description.at( gp.description)};
    auto psip = solovev::createPsip(gp); // make sure prev is shared
    return TokamakMagneticField( gp.R_0,
            mod::createPsip( mod::everywhere, psip, psi0, alpha, sign),
        solovev::createIpol( gp, mod::createPsip( mod::everywhere, psip, psi0, alpha, sign)),
        params);
}

} //namespace geo
} //namespace dg

