#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/topology/functions.h"
#include "dg/functors.h"
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
      \hat{R}_0\Bigg\{B\bar{R}^4/8 + A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
      + \sum_{i=1}^{12} c_i\bar \psi_{pi}(\bar R, \bar Z) \Bigg\}
      =
      \hat{R}_0\Bigg\{B\bar{R}^4/8 + A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
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
    Psip( Parameters gp ): R_0_(gp.R_0), A_(gp.A), B_(gp.B), c_(gp.c) {}
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,Zn6,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
        Zn = Z/R_0_; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; Zn6 = Zn3*Zn3;
        lgRn= log(Rn);
        return   R_0_*( B_*Rn4/8.+ A_ * ( 1./2.* Rn2* lgRn-(Rn4)/8.)
                      + c_[0]  //c_[0] entspricht c_1
              + c_[1]  *Rn2
              + c_[2]  *(Zn2 - Rn2 * lgRn )
              + c_[3]  *(Rn4 - 4.* Rn2*Zn2 )
              + c_[4]  *(3.* Rn4 * lgRn  -9.*Rn2*Zn2 -12.* Rn2*Zn2 * lgRn + 2.*Zn4)
              + c_[5]  *(Rn4*Rn2-12.* Rn4*Zn2 +8.* Rn2 *Zn4 )
              + c_[6]  *(-15.*Rn4*Rn2 * lgRn + 75.* Rn4 *Zn2 + 180.* Rn4*Zn2 * lgRn
                         -140.*Rn2*Zn4 - 120.* Rn2*Zn4 *lgRn + 8.* Zn6 )
              + c_[7]  *Zn
              + c_[8]  *Rn2*Zn
                      + c_[9] *(Zn2*Zn - 3.* Rn2*Zn * lgRn)
              + c_[10] *( 3. * Rn4*Zn - 4. * Rn2*Zn3)
              + c_[11] *(-45.* Rn4*Zn + 60.* Rn4*Zn* lgRn - 80.* Rn2*Zn3* lgRn + 8. * Zn5)
                      );
    }
  private:
    double R_0_, A_, B_;
    std::vector<double> c_;
};

/**
 * @brief \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} \f]
 *
 * \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} =
      \Bigg\{ 2 c_2 \bar{R} +(\bar{R}^3 )/2+2 c_9 \bar{R}  \bar{Z}
      +c_4 (4 \bar{R}^3 -8 \bar{R}  \bar{Z}^2)+c_{11}
      (12 B \bar{R}^3  \bar{Z}-8 \bar{R}  \bar{Z}^3
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
    PsipR( Parameters gp ): R_0_(gp.R_0), A_(gp.A), B_(gp.B), c_(gp.c) {}
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn3,Rn5,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;  Rn5 = Rn3*Rn2;
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
        lgRn= log(Rn);
        return   (Rn3/2.*B_ + (Rn/2. - Rn3/2. + Rn*lgRn)* A_ +
        2.* Rn* c_[1] + (-Rn - 2.* Rn*lgRn)* c_[2] + (4.*Rn3 - 8.* Rn *Zn2)* c_[3] +
        (3. *Rn3 - 30.* Rn *Zn2 + 12. *Rn3*lgRn -  24.* Rn *Zn2*lgRn)* c_[4]
        + (6 *Rn5 - 48 *Rn3 *Zn2 + 16.* Rn *Zn4)*c_[5]
        + (-15. *Rn5 + 480. *Rn3 *Zn2 - 400.* Rn *Zn4 - 90. *Rn5*lgRn +
            720. *Rn3 *Zn2*lgRn - 240.* Rn *Zn4*lgRn)* c_[6] +
        2.* Rn *Zn *c_[8] + (-3. *Rn *Zn - 6.* Rn* Zn*lgRn)* c_[9] + (12. *Rn3* Zn - 8.* Rn *Zn3)* c_[10] + (-120. *Rn3* Zn - 80.* Rn *Zn3 + 240. *Rn3* Zn*lgRn -
            160.* Rn *Zn3*lgRn) *c_[11]
          );
    }
  private:
    double R_0_, A_, B_;
    std::vector<double> c_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\f]
 *
 * \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}=
     \hat{R}_0^{-1} \Bigg\{ 2 c_2 +B(3 \hat{\bar{R}}^2 )/2+2 c_9  \bar{Z}+c_4 (12 \bar{R}^2 -8  \bar{Z}^2)+c_{11}
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
    PsipRR( Parameters gp ): R_0_(gp.R_0), A_(gp.A), B_(gp.B), c_(gp.c) {}
    double do_compute(double R, double Z) const
    {
       double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
       Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
       Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
       lgRn= log(Rn);
       return   1./R_0_*( (3.* Rn2)/2.*B_ + (3./2. - (3. *Rn2)/2. +lgRn) *A_ +  2.* c_[1] + (-3. - 2.*lgRn)* c_[2] + (12. *Rn2 - 8. *Zn2) *c_[3] +
         (21. *Rn2 - 54. *Zn2 + 36. *Rn2*lgRn - 24. *Zn2*lgRn)* c_[4]
         + (30. *Rn4 - 144. *Rn2 *Zn2 + 16.*Zn4)*c_[5] + (-165. *Rn4 + 2160. *Rn2 *Zn2 - 640. *Zn4 - 450. *Rn4*lgRn +
      2160. *Rn2 *Zn2*lgRn - 240. *Zn4*lgRn)* c_[6] +
      2.* Zn* c_[8] + (-9. *Zn - 6.* Zn*lgRn) *c_[9]
 +   (36. *Rn2* Zn - 8. *Zn3) *c_[10]
 +   (-120. *Rn2* Zn - 240. *Zn3 + 720. *Rn2* Zn*lgRn - 160. *Zn3*lgRn)* c_[11]);
    }
  private:
    double R_0_, A_, B_;
    std::vector<double> c_;
};
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\f]
 *
 * \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}=
      \Bigg\{c_8 +c_9 \bar{R}^2 +2 c_3  \bar{Z}-8 c_4 \bar{R}^2  \bar{Z}+c_{11}
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
    PsipZ( Parameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c) { }
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
        Zn = Z/R_0_; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2;
        lgRn= log(Rn);

        return   (2.* Zn* c_[2]
            -  8. *Rn2* Zn* c_[3] +
              ((-18.)*Rn2 *Zn + 8. *Zn3 - 24. *Rn2* Zn*lgRn) *c_[4]
            + ((-24.) *Rn4* Zn + 32. *Rn2 *Zn3)* c_[5]
            + (150. *Rn4* Zn - 560. *Rn2 *Zn3 + 48. *Zn5 + 360. *Rn4* Zn*lgRn - 480. *Rn2 *Zn3*lgRn)* c_[6]
            + c_[7]
            + Rn2 * c_[8]
            + (3. *Zn2 - 3. *Rn2*lgRn)* c_[9]
            + (3. *Rn4 - 12. *Rn2 *Zn2) *c_[10]
            + ((-45.)*Rn4 + 40. *Zn4 + 60. *Rn4*lgRn -  240. *Rn2 *Zn2*lgRn)* c_[11]);

    }
  private:
    double R_0_, A_;
    std::vector<double> c_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}\f]

   \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}=
      \hat{R}_0^{-1} \Bigg\{2 c_3 -8 c_4 \bar{R}^2 +6 c_{10}  \bar{Z}-24 c_{11}
      \bar{R}^2  \bar{Z}+c_6 (-24 \bar{R}^4 +96 \bar{R}^2  \bar{Z}^2)
      +c_5 (-18 \bar{R}^2 +24  \bar{Z}^2-24 \bar{R}^2  \ln{(\bar{R}   )})+
      c_{12} (160  \bar{Z}^3-480 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )})
      +c_7 (150 \bar{R}^4 -1680 \bar{R}^2  \bar{Z}^2+240  \bar{Z}^4+360 \bar{R}^4
      \ln{(\bar{R}   )}-1440 \bar{R}^2  \bar{Z}^2 \ln{(\bar{R}   )})\Bigg\} \f]
 */
struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    ///@copydoc Psip::Psip()
    PsipZZ( Parameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c) { }
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
        lgRn= log(Rn);
        return   1./R_0_*( 2.* c_[2] - 8. *Rn2* c_[3] + (-18. *Rn2 + 24. *Zn2 - 24. *Rn2*lgRn) *c_[4] + (-24.*Rn4 + 96. *Rn2 *Zn2) *c_[5]
        + (150. *Rn4 - 1680. *Rn2 *Zn2 + 240. *Zn4 + 360. *Rn4*lgRn - 1440. *Rn2 *Zn2*lgRn)* c_[6] + 6.* Zn* c_[9] -  24. *Rn2 *Zn *c_[10] + (160. *Zn3 - 480. *Rn2* Zn*lgRn) *c_[11]);
    }
  private:
    double R_0_, A_;
    std::vector<double> c_;
};
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\f]

  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}=
        \hat{R}_0^{-1} \Bigg\{2 c_9 \bar{R} -16 c_4 \bar{R}  \bar{Z}+c_{11}
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
    PsipRZ( Parameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c) { }
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn3,Zn,Zn2,Zn3,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn;
        lgRn= log(Rn);
        return   1./R_0_*(
              -16.* Rn* Zn* c_[3] + (-60.* Rn* Zn - 48.* Rn* Zn*lgRn)* c_[4] + (-96. *Rn3* Zn + 64.*Rn *Zn3)* c_[5]
            + (960. *Rn3 *Zn - 1600.* Rn *Zn3 + 1440. *Rn3* Zn*lgRn - 960. *Rn *Zn3*lgRn) *c_[6] +  2.* Rn* c_[8] + (-3.* Rn - 6.* Rn*lgRn)* c_[9]
            + (12. *Rn3 - 24.* Rn *Zn2) *c_[10] + (-120. *Rn3 - 240. *Rn *Zn2 + 240. *Rn3*lgRn -   480.* Rn *Zn2*lgRn)* c_[11]
                 );
    }
  private:
    double R_0_, A_;
    std::vector<double> c_;
};

/**
 * @brief \f[\hat{I}\f]

    \f[\hat{I}= \sqrt{-2 A \hat{\psi}_p / \hat{R}_0 +1}\f]
 */
struct Ipol: public aCylindricalFunctor<Ipol>
{
    ///@copydoc Psip::Psip()
    Ipol(  Parameters gp ):  R_0_(gp.R_0), A_(gp.A), psip_(gp) { }
    double do_compute(double R, double Z) const
    {
        //sign before A changed to -
        return sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.);
    }
  private:
    double R_0_, A_;
    Psip psip_;
};
/**
 * @brief \f[\hat I_R\f]
 */
struct IpolR: public aCylindricalFunctor<IpolR>
{
    ///@copydoc Psip::Psip()
    IpolR(  Parameters gp ):  R_0_(gp.R_0), A_(gp.A), psip_(gp), psipR_(gp) { }
    double do_compute(double R, double Z) const
    {
        return -1./sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipR_(R,Z)/R_0_);
    }
  private:
    double R_0_, A_;
    Psip psip_;
    PsipR psipR_;
};
/**
 * @brief \f[\hat I_Z\f]
 */
struct IpolZ: public aCylindricalFunctor<IpolZ>
{
    ///@copydoc Psip::Psip()
    IpolZ(  Parameters gp ):  R_0_(gp.R_0), A_(gp.A), psip_(gp), psipZ_(gp) { }
    double do_compute(double R, double Z) const
    {
        return -1./sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipZ_(R,Z)/R_0_);
    }
  private:
    double R_0_, A_;
    Psip psip_;
    PsipZ psipZ_;
};

static inline dg::geo::CylindricalFunctorsLvl2 createPsip( Parameters gp)
{
    return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),
        PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
static inline dg::geo::CylindricalFunctorsLvl1 createIpol( Parameters gp)
{
    return CylindricalFunctorsLvl1( Ipol(gp), IpolR(gp), IpolZ(gp));
}

static inline dg::geo::TokamakMagneticField createMagField( Parameters gp)
{
    return TokamakMagneticField( gp.R_0, createPsip(gp), createIpol(gp));
}
///@}

///@cond
namespace mod
{

struct Psip: public aCylindricalFunctor<Psip>
{
    Psip( Parameters gp, double psi0, double alpha) :
        m_ipoly( psi0, alpha, -1), m_psip(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        return m_ipoly( psip);
    }
    private:
    dg::IPolynomialHeaviside m_ipoly;
    solovev::Psip m_psip;
};
struct PsipR: public aCylindricalFunctor<PsipR>
{
    PsipR( Parameters gp, double psi0, double alpha) :
        m_poly( psi0, alpha, -1), m_psip(gp), m_psipR(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        return psipR*m_poly( psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    solovev::Psip m_psip;
    solovev::PsipR m_psipR;
};
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
    PsipZ( Parameters gp, double psi0, double alpha) :
        m_poly( psi0, alpha, -1), m_psip(gp), m_psipZ(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipZ = m_psipZ(R,Z);
        return psipZ*m_poly( psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    solovev::Psip m_psip;
    solovev::PsipZ m_psipZ;
};

struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
    PsipZZ( Parameters gp, double psi0, double alpha) :
        m_poly( psi0, alpha, -1), m_dpoly( psi0, alpha, -1), m_psip(gp), m_psipZ(gp), m_psipZZ(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipZ = m_psipZ(R,Z);
        double psipZZ = m_psipZZ(R,Z);
        return psipZZ*m_poly( psip) + psipZ*psipZ*m_dpoly(psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    solovev::Psip m_psip;
    solovev::PsipZ m_psipZ;
    solovev::PsipZZ m_psipZZ;
};
struct PsipRR: public aCylindricalFunctor<PsipRR>
{
    PsipRR( Parameters gp, double psi0, double alpha) :
        m_poly( psi0, alpha, -1), m_dpoly( psi0, alpha, -1), m_psip(gp), m_psipR(gp), m_psipRR(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        double psipRR = m_psipRR(R,Z);
        return psipRR*m_poly( psip) + psipR*psipR*m_dpoly(psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    solovev::Psip m_psip;
    solovev::PsipR m_psipR;
    solovev::PsipRR m_psipRR;
};
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
    PsipRZ( Parameters gp, double psi0, double alpha) :
        m_poly( psi0, alpha, -1), m_dpoly( psi0, alpha, -1), m_psip(gp), m_psipR(gp), m_psipZ(gp), m_psipRZ(gp)
    { }
    double do_compute(double R, double Z) const
    {
        double psip = m_psip(R,Z);
        double psipR = m_psipR(R,Z);
        double psipZ = m_psipZ(R,Z);
        double psipRZ = m_psipRZ(R,Z);
        return psipRZ*m_poly( psip) + psipR*psipZ*m_dpoly(psip);
    }
    private:
    dg::PolynomialHeaviside m_poly;
    dg::DPolynomialHeaviside m_dpoly;
    solovev::Psip m_psip;
    solovev::PsipR m_psipR;
    solovev::PsipZ m_psipZ;
    solovev::PsipRZ m_psipRZ;
};

struct Ipol: public aCylindricalFunctor<Ipol>
{
    Ipol(  Parameters gp, double psi0, double alpha ):  R_0_(gp.R_0), A_(gp.A), psip_(gp, psi0, alpha) { }
    double do_compute(double R, double Z) const
    {
        return sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.);
    }
  private:
    double R_0_, A_;
    mod::Psip psip_;
};
struct IpolR: public aCylindricalFunctor<IpolR>
{
    IpolR(  Parameters gp, double psi0, double alpha ):  R_0_(gp.R_0), A_(gp.A), psip_(gp, psi0, alpha), psipR_(gp, psi0, alpha) { }
    double do_compute(double R, double Z) const
    {
        return -1./sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipR_(R,Z)/R_0_);
    }
  private:
    double R_0_, A_;
    mod::Psip psip_;
    mod::PsipR psipR_;
};
struct IpolZ: public aCylindricalFunctor<IpolZ>
{
    IpolZ(  Parameters gp, double psi0, double alpha ):  R_0_(gp.R_0), A_(gp.A), psip_(gp, psi0, alpha), psipZ_(gp, psi0, alpha) { }
    double do_compute(double R, double Z) const
    {
        return -1./sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipZ_(R,Z)/R_0_);
    }
  private:
    double R_0_, A_;
    mod::Psip psip_;
    mod::PsipZ psipZ_;
};

static inline dg::geo::CylindricalFunctorsLvl2 createPsip( Parameters gp,
    double psi0, double alpha)
{
    return CylindricalFunctorsLvl2( Psip(gp, psi0, alpha), PsipR(gp, psi0,
    alpha), PsipZ(gp, psi0, alpha), PsipRR(gp, psi0, alpha), PsipRZ(gp,
    psi0, alpha), PsipZZ(gp, psi0, alpha));
}
static inline dg::geo::CylindricalFunctorsLvl1 createIpol( Parameters gp,
    double psi0, double alpha)
{
    return CylindricalFunctorsLvl1( Ipol(gp, psi0, alpha), IpolR(gp, psi0,
    alpha), IpolZ(gp, psi0, alpha));
}

} //namespace mod
///@endcond

///////////////////////////////////////introduce fields into solovev namespace


} //namespace solovev

/**
 * @brief Create a Solovev Magnetic field
 *
 * Based on \c dg::geo::solovev::Psip(gp) and \c dg::geo::solovev::Ipol(gp)
 * @param gp Solovev parameters
 * @return A magnetic field object
 * @ingroup geom
 */
static inline dg::geo::TokamakMagneticField createSolovevField(
    dg::geo::solovev::Parameters gp)
{
    return TokamakMagneticField( gp.R_0, solovev::createPsip(gp),
        solovev::createIpol(gp));
}
/**
 * @brief Create a modified Solovev Magnetic field
 *
 * Based on \c dg::geo::solovev::mod::Psip(gp) and
 * \c dg::geo::solovev::mod::Ipol(gp)
 * We modify psi above a certain value to a constant using the
 * \c dg::IPolynomialHeaviside function (an approximation to the integrated Heaviside
 * function with width alpha), i.e. we replace psi with IPolynomialHeaviside(psi).
 * This subsequently modifies all derivatives of psi and the poloidal
 * current.
 * @param gp Solovev parameters
 * @param psi0 above this value psi is modified to a constant psi0
 * @param alpha determines how quickly the modification acts (smaller is quicker)
 * @return A magnetic field object
 * @ingroup geom
 */
static inline dg::geo::TokamakMagneticField createModifiedSolovevField(
    dg::geo::solovev::Parameters gp, double psi0, double alpha)
{
    return TokamakMagneticField( gp.R_0, solovev::mod::createPsip(gp,
        psi0, alpha), solovev::mod::createIpol(gp, psi0, alpha));
}

} //namespace geo
} //namespace dg

