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
    Psip( Parameters gp ): m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_c(gp.c) {}
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,Zn6,lgRn;
        Rn = R/m_R0; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
        Zn = Z/m_R0; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; Zn6 = Zn3*Zn3;
        lgRn= log(Rn);
        return   m_R0*m_pp*( Rn4/8.+ m_A * ( 1./2.* Rn2* lgRn-(Rn4)/8.)
                      + m_c[0]  //m_c[0] entspricht c_1
              + m_c[1]  *Rn2
              + m_c[2]  *(Zn2 - Rn2 * lgRn )
              + m_c[3]  *(Rn4 - 4.* Rn2*Zn2 )
              + m_c[4]  *(3.* Rn4 * lgRn  -9.*Rn2*Zn2 -12.* Rn2*Zn2 * lgRn + 2.*Zn4)
              + m_c[5]  *(Rn4*Rn2-12.* Rn4*Zn2 +8.* Rn2 *Zn4 )
              + m_c[6]  *(-15.*Rn4*Rn2 * lgRn + 75.* Rn4 *Zn2 + 180.* Rn4*Zn2 * lgRn
                         -140.*Rn2*Zn4 - 120.* Rn2*Zn4 *lgRn + 8.* Zn6 )
              + m_c[7]  *Zn
              + m_c[8]  *Rn2*Zn
                      + m_c[9] *(Zn2*Zn - 3.* Rn2*Zn * lgRn)
              + m_c[10] *( 3. * Rn4*Zn - 4. * Rn2*Zn3)
              + m_c[11] *(-45.* Rn4*Zn + 60.* Rn4*Zn* lgRn - 80.* Rn2*Zn3* lgRn + 8. * Zn5)
                      );
    }
  private:
    double m_R0, m_A, m_pp;
    std::vector<double> m_c;
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
    PsipR( Parameters gp ): m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_c(gp.c) {}
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn3,Rn5,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/m_R0; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;  Rn5 = Rn3*Rn2;
        Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
        lgRn= log(Rn);
        return   m_pp*(Rn3/2. + (Rn/2. - Rn3/2. + Rn*lgRn)* m_A +
        2.* Rn* m_c[1] + (-Rn - 2.* Rn*lgRn)* m_c[2] + (4.*Rn3 - 8.* Rn *Zn2)* m_c[3] +
        (3. *Rn3 - 30.* Rn *Zn2 + 12. *Rn3*lgRn -  24.* Rn *Zn2*lgRn)* m_c[4]
        + (6 *Rn5 - 48 *Rn3 *Zn2 + 16.* Rn *Zn4)*m_c[5]
        + (-15. *Rn5 + 480. *Rn3 *Zn2 - 400.* Rn *Zn4 - 90. *Rn5*lgRn +
            720. *Rn3 *Zn2*lgRn - 240.* Rn *Zn4*lgRn)* m_c[6] +
        2.* Rn *Zn *m_c[8] + (-3. *Rn *Zn - 6.* Rn* Zn*lgRn)* m_c[9] + (12. *Rn3* Zn - 8.* Rn *Zn3)* m_c[10] + (-120. *Rn3* Zn - 80.* Rn *Zn3 + 240. *Rn3* Zn*lgRn -
            160.* Rn *Zn3*lgRn) *m_c[11]
          );
    }
  private:
    double m_R0, m_A, m_pp;
    std::vector<double> m_c;
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
    PsipRR( Parameters gp ): m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_c(gp.c) {}
    double do_compute(double R, double Z) const
    {
       double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
       Rn = R/m_R0; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
       Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
       lgRn= log(Rn);
       return   m_pp/m_R0*( (3.* Rn2)/2. + (3./2. - (3. *Rn2)/2. +lgRn) *m_A +  2.* m_c[1] + (-3. - 2.*lgRn)* m_c[2] + (12. *Rn2 - 8. *Zn2) *m_c[3] +
         (21. *Rn2 - 54. *Zn2 + 36. *Rn2*lgRn - 24. *Zn2*lgRn)* m_c[4]
         + (30. *Rn4 - 144. *Rn2 *Zn2 + 16.*Zn4)*m_c[5] + (-165. *Rn4 + 2160. *Rn2 *Zn2 - 640. *Zn4 - 450. *Rn4*lgRn +
      2160. *Rn2 *Zn2*lgRn - 240. *Zn4*lgRn)* m_c[6] +
      2.* Zn* m_c[8] + (-9. *Zn - 6.* Zn*lgRn) *m_c[9]
 +   (36. *Rn2* Zn - 8. *Zn3) *m_c[10]
 +   (-120. *Rn2* Zn - 240. *Zn3 + 720. *Rn2* Zn*lgRn - 160. *Zn3*lgRn)* m_c[11]);
    }
  private:
    double m_R0, m_A, m_pp;
    std::vector<double> m_c;
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
    PsipZ( Parameters gp ): m_R0(gp.R_0), m_pp(gp.pp), m_c(gp.c) { }
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,lgRn;
        Rn = R/m_R0; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
        Zn = Z/m_R0; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2;
        lgRn= log(Rn);

        return   m_pp*(2.* Zn* m_c[2]
            -  8. *Rn2* Zn* m_c[3] +
              ((-18.)*Rn2 *Zn + 8. *Zn3 - 24. *Rn2* Zn*lgRn) *m_c[4]
            + ((-24.) *Rn4* Zn + 32. *Rn2 *Zn3)* m_c[5]
            + (150. *Rn4* Zn - 560. *Rn2 *Zn3 + 48. *Zn5 + 360. *Rn4* Zn*lgRn - 480. *Rn2 *Zn3*lgRn)* m_c[6]
            + m_c[7]
            + Rn2 * m_c[8]
            + (3. *Zn2 - 3. *Rn2*lgRn)* m_c[9]
            + (3. *Rn4 - 12. *Rn2 *Zn2) *m_c[10]
            + ((-45.)*Rn4 + 40. *Zn4 + 60. *Rn4*lgRn -  240. *Rn2 *Zn2*lgRn)* m_c[11]);

    }
  private:
    double m_R0, m_pp;
    std::vector<double> m_c;
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
    PsipZZ( Parameters gp): m_R0(gp.R_0), m_pp(gp.pp), m_c(gp.c) { }
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/m_R0; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
        Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
        lgRn= log(Rn);
        return   m_pp/m_R0*( 2.* m_c[2] - 8. *Rn2* m_c[3] + (-18. *Rn2 + 24. *Zn2 - 24. *Rn2*lgRn) *m_c[4] + (-24.*Rn4 + 96. *Rn2 *Zn2) *m_c[5]
        + (150. *Rn4 - 1680. *Rn2 *Zn2 + 240. *Zn4 + 360. *Rn4*lgRn - 1440. *Rn2 *Zn2*lgRn)* m_c[6] + 6.* Zn* m_c[9] -  24. *Rn2 *Zn *m_c[10] + (160. *Zn3 - 480. *Rn2* Zn*lgRn) *m_c[11]);
    }
  private:
    double m_R0, m_pp;
    std::vector<double> m_c;
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
    PsipRZ( Parameters gp ): m_R0(gp.R_0), m_pp(gp.pp), m_c(gp.c) { }
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn3,Zn,Zn2,Zn3,lgRn;
        Rn = R/m_R0; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;
        Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn;
        lgRn= log(Rn);
        return   m_pp/m_R0*(
              -16.* Rn* Zn* m_c[3] + (-60.* Rn* Zn - 48.* Rn* Zn*lgRn)* m_c[4] + (-96. *Rn3* Zn + 64.*Rn *Zn3)* m_c[5]
            + (960. *Rn3 *Zn - 1600.* Rn *Zn3 + 1440. *Rn3* Zn*lgRn - 960. *Rn *Zn3*lgRn) *m_c[6] +  2.* Rn* m_c[8] + (-3.* Rn - 6.* Rn*lgRn)* m_c[9]
            + (12. *Rn3 - 24.* Rn *Zn2) *m_c[10] + (-120. *Rn3 - 240. *Rn *Zn2 + 240. *Rn3*lgRn -   480.* Rn *Zn2*lgRn)* m_c[11]
                 );
    }
  private:
    double m_R0, m_pp;
    std::vector<double> m_c;
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
    Ipol( Parameters gp, std::function<double(double,double)> psip ):  m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_pi(gp.pi), m_psip(psip) {
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
    IpolR(  Parameters gp, std::function<double(double,double)> psip, std::function<double(double,double)> psipR ):
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
    IpolZ(  Parameters gp, std::function<double(double,double)> psip, std::function<double(double,double)> psipZ ):
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
    dg::geo::solovev::Parameters gp)
{
    MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
            equilibrium::solovev, modifier::none, str2description.at( gp.description)};
    return TokamakMagneticField( gp.R_0, solovev::createPsip(gp),
        solovev::createIpol(gp, solovev::createPsip(gp)), params);
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
    dg::geo::solovev::Parameters gp, double psi0, double alpha, double sign = -1)
{
    MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
            equilibrium::solovev, modifier::heaviside, str2description.at( gp.description)};
    return TokamakMagneticField( gp.R_0,
            mod::createPsip( mod::everywhere, solovev::createPsip(gp), psi0, alpha, sign),
        solovev::createIpol( gp, mod::createPsip( mod::everywhere, solovev::createPsip(gp), psi0, alpha, sign)),
        params);
}

} //namespace geo
} //namespace dg

