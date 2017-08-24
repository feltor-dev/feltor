#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/backend/functions.h"
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
///@addtogroup geom
///@{

/**
 * @brief \f[ \hat{\psi}_p  \f]
 *
 * \f[ \hat{\psi}_p(R,Z) = 
      \hat{R}_0\Bigg\{\bar{R}^4/8 + A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
      + \sum_{i=1}^{12} c_i\bar \psi_{pi}(\bar R, \bar Z) \Bigg\}
      =
      \hat{R}_0\Bigg\{A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
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
struct Psip: public aCloneableBinaryFunctor<Psip>
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp geometric parameters
     */
    Psip( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c) {}
  private:
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,Zn6,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
        Zn = Z/R_0_; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; Zn6 = Zn3*Zn3;
        lgRn= log(Rn);
        return   R_0_*( c_[12]*Rn4/8.+ A_ * ( 1./2.* Rn2* lgRn-(Rn4)/8.)  //c_[12] is to make fieldlines straight
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
    double psi_horner(double R, double Z) const
    {
        //qampl is missing!!
        const double Rn = R/R_0_, Zn = Z/R_0_;
        const double lgRn = log(Rn);
        double a0,a1, b0,b1, d0, d1;
        a0 = 8.*c_[11]+8*c_[6]*Zn;
        a1 = 2.*c_[4] +Zn*a0;
        a0 = c_[9] + Zn*a1;
        a1 = c_[2] + Zn*a0;
        a0 = c_[7] + Zn*a1;
        a1 = c_[0] + Zn*a0; //////
        b0 = -12.*c_[5] + 75.*c_[6] + 180.*c_[6]*lgRn;
        b1 = 3.*c_[10] - 45.*c_[11]+60.*c_[11]*lgRn + Zn*b0;
        b0 = 1./8.-A_/8. + c_[3] + 3.*c_[4]*lgRn + Rn*Rn*(c_[5] - 15.*c_[6]*lgRn)+Zn*b1;
        b1 = c_[1] + 0.5*A_*lgRn - c_[2]*lgRn + Rn*Rn*b0; 
        b0 = Rn*Rn*b1;//////
        d0 = 8.*c_[5] - 140.*c_[6]-120.*c_[6]*lgRn;
        d1 = -4.*c_[10] - 80.*c_[11]*lgRn + Zn*d0;
        d0 = -4.*c_[3]-9.*c_[4]-12.*c_[4]*lgRn + Zn*d1;
        d1 = c_[8] - 3.*c_[9]*lgRn + Zn*d0;
        d0 = Rn*Rn*Zn*d1; /////

        return  R_0_*(a1 + b0 + d0);
    }
    double R_0_, A_;
    std::vector<double> c_;
};

/**
 * @brief \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} \f]
 *
 * \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} =
      \Bigg\{ 2 c_2 \bar{R} +(\bar{R}^3 )/2+2 c_9 \bar{R}  \bar{Z}
      +c_4 (4 \bar{R}^3 -8 \bar{R}  \bar{Z}^2)+c_{11} 
      (12 \bar{R}^3  \bar{Z}-8 \bar{R}  \bar{Z}^3 
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
struct PsipR: public aCloneableBinaryFunctor<PsipR>
{
    ///@copydoc Psip::Psip()
    PsipR( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c) {}
  private:
    double do_compute(double R, double Z) const
    {    
        double Rn,Rn2,Rn3,Rn5,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;  Rn5 = Rn3*Rn2; 
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
        lgRn= log(Rn);
        return   (Rn3/2.*c_[12] + (Rn/2. - Rn3/2. + Rn*lgRn)* A_ + 
        2.* Rn* c_[1] + (-Rn - 2.* Rn*lgRn)* c_[2] + (4.*Rn3 - 8.* Rn *Zn2)* c_[3] + 
        (3. *Rn3 - 30.* Rn *Zn2 + 12. *Rn3*lgRn -  24.* Rn *Zn2*lgRn)* c_[4]
        + (6 *Rn5 - 48 *Rn3 *Zn2 + 16.* Rn *Zn4)*c_[5]
        + (-15. *Rn5 + 480. *Rn3 *Zn2 - 400.* Rn *Zn4 - 90. *Rn5*lgRn + 
            720. *Rn3 *Zn2*lgRn - 240.* Rn *Zn4*lgRn)* c_[6] + 
        2.* Rn *Zn *c_[8] + (-3. *Rn *Zn - 6.* Rn* Zn*lgRn)* c_[9] + (12. *Rn3* Zn - 8.* Rn *Zn3)* c_[10] + (-120. *Rn3* Zn - 80.* Rn *Zn3 + 240. *Rn3* Zn*lgRn - 
            160.* Rn *Zn3*lgRn) *c_[11]
          );
    }
    double R_0_, A_;
    std::vector<double> c_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\f]
 *
 * \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}=
     \hat{R}_0^{-1} \Bigg\{ 2 c_2 +(3 \hat{\bar{R}}^2 )/2+2 c_9  \bar{Z}+c_4 (12 \bar{R}^2 -8  \bar{Z}^2)+c_{11} 
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
struct PsipRR: public aCloneableBinaryFunctor<PsipRR>
{
    ///@copydoc Psip::Psip()
    PsipRR( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c) {}
  private:
    double do_compute(double R, double Z) const
    {    
       double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
       Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
       Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
       lgRn= log(Rn);
       return   1./R_0_*( (3.* Rn2)/2.*c_[12] + (3./2. - (3. *Rn2)/2. +lgRn) *A_ +  2.* c_[1] + (-3. - 2.*lgRn)* c_[2] + (12. *Rn2 - 8. *Zn2) *c_[3] + 
         (21. *Rn2 - 54. *Zn2 + 36. *Rn2*lgRn - 24. *Zn2*lgRn)* c_[4]
         + (30. *Rn4 - 144. *Rn2 *Zn2 + 16.*Zn4)*c_[5] + (-165. *Rn4 + 2160. *Rn2 *Zn2 - 640. *Zn4 - 450. *Rn4*lgRn + 
      2160. *Rn2 *Zn2*lgRn - 240. *Zn4*lgRn)* c_[6] + 
      2.* Zn* c_[8] + (-9. *Zn - 6.* Zn*lgRn) *c_[9] 
 +   (36. *Rn2* Zn - 8. *Zn3) *c_[10]
 +   (-120. *Rn2* Zn - 240. *Zn3 + 720. *Rn2* Zn*lgRn - 160. *Zn3*lgRn)* c_[11]);
    }
    double R_0_, A_;
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
struct PsipZ: public aCloneableBinaryFunctor<PsipZ>
{
    ///@copydoc Psip::Psip()
    PsipZ( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c) { }
  private:
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
struct PsipZZ: public aCloneableBinaryFunctor<PsipZZ>
{
    ///@copydoc Psip::Psip()
    PsipZZ( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c) { }
  private:
    double do_compute(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2; 
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
        lgRn= log(Rn);
        return   1./R_0_*( 2.* c_[2] - 8. *Rn2* c_[3] + (-18. *Rn2 + 24. *Zn2 - 24. *Rn2*lgRn) *c_[4] + (-24.*Rn4 + 96. *Rn2 *Zn2) *c_[5]
        + (150. *Rn4 - 1680. *Rn2 *Zn2 + 240. *Zn4 + 360. *Rn4*lgRn - 1440. *Rn2 *Zn2*lgRn)* c_[6] + 6.* Zn* c_[9] -  24. *Rn2 *Zn *c_[10] + (160. *Zn3 - 480. *Rn2* Zn*lgRn) *c_[11]);
    }
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
struct PsipRZ: public aCloneableBinaryFunctor<PsipRZ>
{
    ///@copydoc Psip::Psip()
    PsipRZ( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c) { }
  private:
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
    double R_0_, A_;
    std::vector<double> c_;
};

/**
 * @brief \f[\hat{I}\f] 

    \f[\hat{I}= \sqrt{-2 A \hat{\psi}_p / \hat{R}_0 +1}\f] 
 */ 
struct Ipol: public aCloneableBinaryFunctor<Ipol>
{
    ///@copydoc Psip::Psip()
    Ipol(  GeomParameters gp ):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp) { }
  private:
    double do_compute(double R, double Z) const
    {    
        //sign before A changed to -
        return qampl_*sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.);
    }
    double R_0_, A_,qampl_;
    Psip psip_;
};
/**
 * @brief \f[\hat I_R\f]
 */
struct IpolR: public aCloneableBinaryFunctor<IpolR>
{
    ///@copydoc Psip::Psip()
    IpolR(  GeomParameters gp ):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp), psipR_(gp) { }
  private:
    double do_compute(double R, double Z) const
    {    
        return -qampl_/sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipR_(R,Z)/R_0_);
    }
    double R_0_, A_,qampl_;
    Psip psip_;
    PsipR psipR_;
};
/**
 * @brief \f[\hat I_Z\f]
 */
struct IpolZ: public aCloneableBinaryFunctor<IpolZ>
{
    ///@copydoc Psip::Psip()
    IpolZ(  GeomParameters gp ):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp), psipZ_(gp) { }
  private:
    double do_compute(double R, double Z) const
    {    
        return -qampl_/sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipZ_(R,Z)/R_0_);
    }
    double R_0_, A_,qampl_;
    Psip psip_;
    PsipZ psipZ_;
};

BinaryFunctorsLvl2 createPsip( GeomParameters gp)
{
    BinaryFunctorsLvl2 psip( new Psip(gp), new PsipR(gp), new PsipZ(gp),new PsipRR(gp), new PsipRZ(gp), new PsipZZ(gp));
    return psip;
}
BinaryFunctorsLvl1 createIpol( GeomParameters gp)
{
    BinaryFunctorsLvl1 ipol( new Ipol(gp), new IpolR(gp), new IpolZ(gp));
    return ipol;
}

TokamakMagneticField createMagField( GeomParameters gp)
{
    return TokamakMagneticField( gp.R_0, createPsip(gp), createIpol(gp));
}
///@}

///@cond
namespace mod
{

struct Psip: public aCloneableBinaryFunctor<Psip>
{
    Psip( GeomParameters gp): R_X( gp.R_0-1.1*gp.triangularity*gp.a), Z_X(-1.1*gp.elongation*gp.a),
        psip_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp), cauchy_( R_X, Z_X, 50, 50,1.)
    {
        psipZZ_X_ = psipZZ_(R_X, Z_X);
        psipRZ_X_ = psipRZ_(R_X, Z_X);
        psipRR_X_ = psipRR_(R_X, Z_X);
    
    }
    private:
    double do_compute(double R, double Z) const
    {    
        double psip_RZ = psip_(R,Z);
        double Rbar = R - R_X, Zbar = Z - Z_X;
        double psip_2 =  0.5*(- psipZZ_X_*Rbar*Rbar + 2.*psipRZ_X_*Rbar*Zbar - psipRR_X_*Zbar*Zbar) - psip_RZ ; 
        return  psip_RZ + 0.5*psip_2*cauchy_(R,Z);
    }
    double R_X, Z_X; 
    double psipZZ_X_, psipRZ_X_, psipRR_X_;
    solovev::Psip psip_;
    solovev::PsipRR psipRR_;
    solovev::PsipRZ psipRZ_;
    solovev::PsipZZ psipZZ_;
    dg::Cauchy cauchy_;
};
struct PsipR: public aCloneableBinaryFunctor<PsipR>
{
    PsipR( GeomParameters gp): R_X( gp.R_0-1.1*gp.triangularity*gp.a), Z_X(-1.1*gp.elongation*gp.a),
        psip_(gp), psipR_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp), cauchy_( R_X, Z_X, 50, 50,1.)
    {
        psipZZ_X_ = psipZZ_(R_X, Z_X);
        psipRZ_X_ = psipRZ_(R_X, Z_X);
        psipRR_X_ = psipRR_(R_X, Z_X);
    }
    private:
    double do_compute(double R, double Z) const
    {    
        double psipR_RZ = psipR_(R,Z);
        if( !cauchy_.inside(R,Z)) return psipR_RZ;
        double Rbar = R - R_X, Zbar = Z - Z_X;
        double psip_2 =  0.5*(- psipZZ_X_*Rbar*Rbar + 2.*psipRZ_X_*Rbar*Zbar - psipRR_X_*Zbar*Zbar) - psip_(R,Z) ; 
        double psip_2R =  - psipZZ_X_*Rbar + psipRZ_X_*Zbar - psipR_RZ;
        return psipR_RZ + 0.5*(psip_2R*cauchy_(R,Z) + psip_2*cauchy_.dx(R,Z)  );
    }
    double R_X, Z_X; 
    double psipZZ_X_, psipRZ_X_, psipRR_X_;
    solovev::Psip psip_;
    solovev::PsipR psipR_;
    solovev::PsipRR psipRR_;
    solovev::PsipRZ psipRZ_;
    solovev::PsipZZ psipZZ_;
    dg::Cauchy cauchy_;
};
struct PsipZ: public aCloneableBinaryFunctor<PsipZ>
{
    PsipZ( GeomParameters gp): R_X( gp.R_0-1.1*gp.triangularity*gp.a), Z_X(-1.1*gp.elongation*gp.a),
        psip_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp), cauchy_( R_X, Z_X, 50, 50, 1)
    {
        psipZZ_X_ = psipZZ_(R_X, Z_X);
        psipRZ_X_ = psipRZ_(R_X, Z_X);
        psipRR_X_ = psipRR_(R_X, Z_X);
    }
    private:
    double do_compute(double R, double Z) const
    {    
        double psipZ_RZ = psipZ_(R,Z);
        if( !cauchy_.inside(R,Z)) return psipZ_RZ;
        double Rbar = R - R_X, Zbar = Z - Z_X;
        double psip_2 =  0.5*(- psipZZ_X_*Rbar*Rbar + 2.*psipRZ_X_*Rbar*Zbar - psipRR_X_*Zbar*Zbar) - psip_(R,Z) ; 
        double psip_2Z =  - psipRR_X_*Zbar + psipRZ_X_*Rbar - psipZ_RZ;
        return psipZ_RZ + 0.5*(psip_2Z*cauchy_(R,Z) + psip_2*cauchy_.dy(R,Z));
    }
    double R_X, Z_X; 
    double psipZZ_X_, psipRZ_X_, psipRR_X_;
    solovev::Psip psip_;
    solovev::PsipZ psipZ_;
    solovev::PsipRR psipRR_;
    solovev::PsipRZ psipRZ_;
    solovev::PsipZZ psipZZ_;
    dg::Cauchy cauchy_;
};

struct PsipZZ: public aCloneableBinaryFunctor<PsipZZ>
{
    PsipZZ( GeomParameters gp): R_X( gp.R_0-1.1*gp.triangularity*gp.a), Z_X(-1.1*gp.elongation*gp.a),
        psip_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp), cauchy_( R_X, Z_X, 50, 50, 1)
    {
        psipZZ_X_ = psipZZ_(R_X, Z_X);
        psipRZ_X_ = psipRZ_(R_X, Z_X);
        psipRR_X_ = psipRR_(R_X, Z_X);
    }
    private:
    double do_compute(double R, double Z) const
    {    
        double psipZZ_RZ = psipZZ_(R,Z);
        if( !cauchy_.inside(R, Z)) return psipZZ_RZ;
        double Rbar = R - R_X, Zbar = Z - Z_X;
        double psip_2 =  0.5*(- psipZZ_X_*Rbar*Rbar + 2.*psipRZ_X_*Rbar*Zbar - psipRR_X_*Zbar*Zbar) - psip_(R,Z) ; 
        double psip_2Z =  - psipRR_X_*Zbar + psipRZ_X_*Rbar - psipZ_(R,Z);
        double psip_2ZZ =  - psipRR_X_ - psipZZ_RZ;
        return psipZZ_RZ + 0.5*(psip_2ZZ*cauchy_(R,Z) + 2.*cauchy_.dy(R,Z)*psip_2Z +  psip_2*cauchy_.dyy(R,Z));
    }
    double R_X, Z_X; 
    double psipZZ_X_, psipRZ_X_, psipRR_X_;
    solovev::Psip psip_;
    solovev::PsipZ psipZ_;
    solovev::PsipRR psipRR_;
    solovev::PsipRZ psipRZ_;
    solovev::PsipZZ psipZZ_;
    dg::Cauchy cauchy_;
};
struct PsipRR: public aCloneableBinaryFunctor<PsipRR>
{
    PsipRR( GeomParameters gp): R_X( gp.R_0-1.1*gp.triangularity*gp.a), Z_X(-1.1*gp.elongation*gp.a),
        psip_(gp), psipR_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp), cauchy_( R_X, Z_X, 50, 50, 1)
    {
        psipZZ_X_ = psipZZ_(R_X, Z_X);
        psipRZ_X_ = psipRZ_(R_X, Z_X);
        psipRR_X_ = psipRR_(R_X, Z_X);
    }
    private:
    double do_compute(double R, double Z) const
    {    
        double psipRR_RZ = psipRR_(R,Z);
        if( !cauchy_.inside(R,Z)) return psipRR_RZ;
        double Rbar = R - R_X, Zbar = Z - Z_X;
        double psip_2 =  0.5*(- psipZZ_X_*Rbar*Rbar + 2.*psipRZ_X_*Rbar*Zbar - psipRR_X_*Zbar*Zbar) - psip_(R,Z) ; 
        double psip_2R =  - psipZZ_X_*Rbar + psipRZ_X_*Zbar - psipR_(R,Z);
        double psip_2RR =  - psipZZ_X_ - psipRR_RZ;
        return psipRR_RZ + 0.5*(psip_2RR*cauchy_(R,Z) + 2.*cauchy_.dx(R,Z)*psip_2R +  psip_2*cauchy_.dxx(R,Z));
    }
    double R_X, Z_X; 
    double psipZZ_X_, psipRZ_X_, psipRR_X_;
    solovev::Psip psip_;
    solovev::PsipR psipR_;
    solovev::PsipRR psipRR_;
    solovev::PsipRZ psipRZ_;
    solovev::PsipZZ psipZZ_;
    dg::Cauchy cauchy_;
};
struct PsipRZ: public aCloneableBinaryFunctor<PsipRZ>
{
    PsipRZ( GeomParameters gp): R_X( gp.R_0-1.1*gp.triangularity*gp.a), Z_X(-1.1*gp.elongation*gp.a),
        psip_(gp), psipR_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp), cauchy_( R_X, Z_X, 50, 50, 1)
    {
        psipZZ_X_ = psipZZ_(R_X, Z_X);
        psipRZ_X_ = psipRZ_(R_X, Z_X);
        psipRR_X_ = psipRR_(R_X, Z_X);
    }
    private:
    double do_compute(double R, double Z) const
    {    
        double psipRZ_RZ = psipRZ_(R,Z);
        if( !cauchy_.inside(R,Z)) return psipRZ_RZ;
        double Rbar = R - R_X, Zbar = Z - Z_X;
        double psip_2 =  0.5*(- psipZZ_(R_X, Z_X)*Rbar*Rbar + 2.*psipRZ_(R_X, Z_X)*Rbar*Zbar - psipRR_(R_X, Z_X)*Zbar*Zbar) - psip_(R,Z); 
        double psip_2R =  - psipZZ_X_*Rbar + psipRZ_X_*Zbar - psipR_(R,Z);
        double psip_2Z =  - psipRR_X_*Zbar + psipRZ_X_*Rbar - psipZ_(R,Z);
        double psip_2RZ =  - psipRZ_X_ - psipRZ_RZ;
        return psipRZ_RZ + 0.5*(psip_2RZ*cauchy_(R,Z) + cauchy_.dx(R,Z)*psip_2Z + cauchy_.dy(R,Z)*psip_2R  +  psip_2*cauchy_.dxy(R,Z));
    }
    double R_X, Z_X; 
    double psipZZ_X_, psipRZ_X_, psipRR_X_;
    solovev::Psip psip_;
    solovev::PsipR psipR_;
    solovev::PsipZ psipZ_;
    solovev::PsipRR psipRR_;
    solovev::PsipRZ psipRZ_;
    solovev::PsipZZ psipZZ_;
    dg::Cauchy cauchy_;
};

} //namespace mod
///@endcond

///////////////////////////////////////introduce fields into solovev namespace


} //namespace solovev

TokamakMagneticField createSolovevField( solovev::GeomParameters gp)
{
    return TokamakMagneticField( gp.R_0, solovev::createPsip(gp), solovev::createIpol(gp));
}

} //namespace geo
} //namespace dg

