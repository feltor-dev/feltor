#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/backend/functions.h"
#include "solovev_parameters.h"


/*!@file
 *
 * Geometry objects 
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
struct Psip
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp useful geometric parameters
     */
    Psip( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psi_0(gp.psipmaxcut), alpha_( gp.alpha) {}
/**
 * @brief \f$ \hat \psi_p(R,Z) \f$

      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @return \f$ \hat \psi_p(R,Z) \f$
 */
    double operator()(double R, double Z) const
    {    
        return psi_alt( R, Z);
    }
    /**
     * @brief \f$ \psi_p(R,Z,\phi) \equiv \psi_p(R,Z)\f$
     *
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @param phi angle (cylindrical coordinates)
     *
     * @return \f$ \hat \psi_p(R,Z,\phi) \f$
     */
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    /**
     * @brief Show parameters to std::cout
     */
    void display() const
    {
      std::cout << R_0_ <<"  " <<A_ <<"\n";
      std::cout << c_[0] <<"\n";
    }
  private:
    double psi_alt(double R, double Z) const
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
    double R_0_, A_;
    std::vector<double> c_;
    double psi_0;
    double alpha_;
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
struct PsipR
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp useful geometric parameters
     */
    PsipR( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp) {}
/**
 * @brief \f$ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}}(R,Z)  \f$

      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
    * @return \f$ \frac{\partial  \hat{\psi}_p}{ \partial \hat{R}}(R,Z)  \f$
 */ 
    double operator()(double R, double Z) const
    {    
        return psipR_alt( R, Z);
    }
    /**
     * @brief \f$ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}}(R,Z,\phi) \equiv \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}}(R,Z)\f$
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @param phi angle (cylindrical coordinates)
    * @return \f$ \frac{\partial  \hat{\psi}_p}{ \partial \hat{R}}(R,Z,\phi)  \f$
 */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }


    /**
     * @brief Print parameters to std::cout
     */
    void display() const
    {
      std::cout << R_0_ <<"  " <<A_ <<"\n";
      std::cout << c_[0] <<"\n";
    }
  private:
    double psipR_alt(double R, double Z) const
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
    Psip psip_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\f]
 */ 
struct PsipRR
{
    /**
    * @brief Constructor
    *
    * @param gp geometric parameters
    */
    PsipRR( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp), psipR_(gp) {}
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}=
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
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @return value
 */ 
    double operator()(double R, double Z) const
    {    
        //double psipR = psipR_.psipR_alt( R,Z);
        //return psip_.diff_psi_neu( R, Z)* psipRR_alt( R,Z) + psip_.diffdiff_psi_neu( R, Z) *psipR;
        return psipRR_alt( R, Z);
    }
    /**
    * @brief return operator()(R,Z)
    *
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @param phi angle (cylindrical coordinates)
    *
    * @return value
    */
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    /**
    * @brief Display the internal parameters to std::cout
    */
    void display()
    {
      std::cout << R_0_ <<"  " <<A_ <<"\n";
      std::cout << c_[0] <<"\n";
    }
  private:
    double psipRR_alt(double R, double Z) const
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
    Psip psip_;
    PsipR psipR_;
};
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\f]
 */ 
struct PsipZ
{
    PsipZ( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp) { }
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}= 
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
    double psipZ_alt(double R, double Z) const
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
    double operator()(double R, double Z) const
    {    
        //return psip_.diff_psi_neu( R, Z)* psipZ_alt( R,Z);
        return psipZ_alt(R, Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    void display() const
    {
        std::cout << R_0_ <<"  " <<A_ <<"\n";
        std::cout << c_[0] <<"\n";
    }
  private:
    double R_0_, A_;
    std::vector<double> c_;
    Psip psip_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}\f]
 */ 
struct PsipZZ
{
  PsipZZ( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp), psipZ_(gp) { }
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}=
      \hat{R}_0^{-1} \Bigg\{2 c_3 -8 c_4 \bar{R}^2 +6 c_{10}  \bar{Z}-24 c_{11}
      \bar{R}^2  \bar{Z}+c_6 (-24 \bar{R}^4 +96 \bar{R}^2  \bar{Z}^2)
      +c_5 (-18 \bar{R}^2 +24  \bar{Z}^2-24 \bar{R}^2  \ln{(\bar{R}   )})+
      c_{12} (160  \bar{Z}^3-480 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )})
      +c_7 (150 \bar{R}^4 -1680 \bar{R}^2  \bar{Z}^2+240  \bar{Z}^4+360 \bar{R}^4 
      \ln{(\bar{R}   )}-1440 \bar{R}^2  \bar{Z}^2 \ln{(\bar{R}   )})\Bigg\} \f]
 */ 
    double operator()(double R, double Z) const
    {    
        //double psipZ = psipZ_.psipZ_alt(R,Z);
        //return psip_.diff_psi_neu( R, Z)* psipZZ_alt( R,Z) + 
        //       psip_.diffdiff_psi_neu( R, Z) *psipZ*psipZ;
        return psipZZ_alt( R, Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    void display() const
    {
        std::cout << R_0_ <<"  " <<A_ <<"\n";
        std::cout << c_[0] <<"\n";
    }
  private:
    double psipZZ_alt(double R, double Z) const
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
    Psip psip_; 
    PsipZ psipZ_;
};
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\f] 
 */ 
struct PsipRZ
{
    PsipRZ( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp), psipZ_(gp), psipR_(gp) {  }
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}= 
        \hat{R}_0^{-1} \Bigg\{2 c_9 \bar{R} -16 c_4 \bar{R}  \bar{Z}+c_{11} 
      (12 \bar{R}^3 -24 \bar{R}  \bar{Z}^2)+c_6 (-96 \bar{R}^3  \bar{Z}+64 \bar{R}  \bar{Z}^3)
      + c_{10} (-3 \bar{R} -6 \bar{R}  \ln{(\bar{R}   )})
      +c_5 (-60 \bar{R}  \bar{Z}-48 \bar{R}  \bar{Z} \ln{(\bar{R}   )})
      +c_{12}  (-120 \bar{R}^3 -240 \bar{R}  \bar{Z}^2+
      240 \bar{R}^3  \ln{(\bar{R}   )}-480 \bar{R}  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_7(960 \bar{R}^3  \bar{Z}-1600 \bar{R}  \bar{Z}^3+1440 \bar{R}^3  \bar{Z} \ln{(\bar{R} 
      )}-960 \bar{R}  \bar{Z}^3 \ln{(\bar{R}   )})\Bigg\} \f] 
 */ 
    double operator()(double R, double Z) const
    {    
        //return psip_.diff_psi_neu( R, Z)* psipRZ_alt( R,Z) + 
        //       psip_.diffdiff_psi_neu( R, Z) *psipR_.psipR_alt(R,Z)*psipZ_.psipZ_alt(R,Z);
        return psipRZ_alt( R, Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    void display() const
    {
        std::cout << R_0_ <<"  " <<A_ <<"\n";
        std::cout << c_[0] <<"\n";
    }
  private:
    double psipRZ_alt(double R, double Z) const
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
    Psip psip_; 
    PsipZ psipZ_;
    PsipR psipR_;
};
/**
 * @brief \f[\hat{I}\f] 
 */ 
struct Ipol
{
    Ipol(  GeomParameters gp ):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp) { }
    /**
    * @brief \f[\hat{I}= \sqrt{-2 A \hat{\psi}_p / \hat{R}_0 +1}\f] 
    */ 
    double operator()(double R, double Z) const
    {    
        //sign before A changed to -
        return qampl_*sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()( R,Z);
    }
  private:
    double R_0_, A_,qampl_;
    Psip psip_;
};
struct IpolR
{
    IpolR(  GeomParameters gp ):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp), psipR_(gp) { }
    double operator()(double R, double Z) const
    {    
        return -qampl_/sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipR_(R,Z)/R_0_);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()( R,Z);
    }
  private:
    double R_0_, A_,qampl_;
    Psip psip_;
    PsipR psipR_;
};
struct IpolZ
{
    IpolZ(  GeomParameters gp ):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp), psipZ_(gp) { }
    double operator()(double R, double Z) const
    {    
        return -qampl_/sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.)*(A_*psipZ_(R,Z)/R_0_);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()( R,Z);
    }
  private:
    double R_0_, A_,qampl_;
    Psip psip_;
    PsipZ psipZ_;
};

} //namespace dg
