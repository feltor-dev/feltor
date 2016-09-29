#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <boost/math/bessel.hpp>

#include "dg/blas.h"

#include "dg/backend/functions.h"
#include "dg/functors.h"
#include "solovev_parameters.h"


/*!@file
 *
 * Geometry objects 
 */
namespace taylor
{
///@addtogroup geom
///@{

struct Psip
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp useful geometric parameters
     */
    Psip( GeomParameters gp): R0_(gp.R_0), c_(gp.c) {
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
/**
 * @brief \f$ \hat \psi_p(R,Z) \f$

      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @return \f$ \hat \psi_p(R,Z) \f$
 */
    double operator()(double R, double Z) const
    {    
        double Rn = R/R0_, Zn = Z/Z_0_;
        double j1_c12 = cyl_bessel_j( 1, c_[11]*Rn);
        double y1_c12 = cyl_neumann( 1, c_[11]*Rn);
        double j1_cs = cyl_bessel_j( 1, cs_*Rn);
        double y1_cs = cyl_neumann( 1, cs_*Rn);
        return R0_*(    
                   1.0*Rn*j1_c12 
               + c_[0]*Rn*yn_c12 
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
  private:
    double R0_, cs_;
    std::vector<double> c_;
};

struct PsipR
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp useful geometric parameters
     */
    PsipR( GeomParameters gp): R0_(gp.R_0), c_(gp.c) {
        cs_=sqrt(c_[11]*c_[11]-c_[10]*c_[10]);
    
    }
/**
 * @brief \f$ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}}(R,Z)  \f$

      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
    * @return \f$ \frac{\partial  \hat{\psi}_p}{ \partial \hat{R}}(R,Z)  \f$
 */ 
    double operator()(double R, double Z) const
    {    
        double Rn=R/R0_, Zn=Z/Z_0_;
        double j1_c12R = cyl_bessel_j(1, c_[11]*Rn) + c_[11]/2.*Rn*(cyl_bessel_j(0, c_[11]*Rn) - cyl_bessel_j(2,c_[11]*Rn));
        double y1_c12R = cyl_neumann(1, c_[11]*Rn) + c_[11]/2.*Rn*(cyl_neumann(0, c_[11]*Rn) - cyl_neumann(2,c_[11]*Rn));
        double j1_csR = cyl_bessel_j(1, cs_*Rn) + cs_/2.*Rn*(cyl_bessel_j(0, cs_*Rn) - cyl_bessel_j(2, cs_*Rn));
        double y1_csR = cyl_neumann(1, cs_*Rn) + cs_/2.*Rn*(cyl_neumann(0, cs_*Rn) - cyl_neumann(2, cs_*Rn));
        return  R0_*(   
                   1.0*j1_c12R 
               + c_[0]*yn_c12R 
               + c_[1]*j1_csR*cos(c_[10]*Zn)  
               + c_[2]*y1_csR*cos(c_[10]*Zn)  
               - c_[3]*c_[11]*Rn/sqrt(Rn*Rn+Zn*Zn)*sin(c_[11]*sqrt(Rn*Rn+Zn*Zn))  
               + c_[5]*j1_c12R*Zn
               + c_[6]*y1_c12R*Zn
               + c_[7]*j1_csR*sin(c_[10]*Zn)
               + c_[8]*y1_csR*sin(c_[10]*Zn) );
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


  private:
    double R0_, cs_;
    std::vector<double> c_;
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
    PsipRR( GeomParameters gp ): R0_(gp.R_0), c_(gp.c) {
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double operator()(double R, double Z) const
    {    
        double Rn=R/R0_, Zn=Z/Z_0_;
        double j1_c12R = (1./Rn-c_[11]*c_[11]*Rn)*cyl_bessel_j(1, c_[11]*Rn) + c_[11]/2./Rn*(cyl_bessel_j(0, c_[11]*Rn) - cyl_bessel_j(2,c_[11]*Rn));
        double y1_c12R = (1./Rn-c_[11]*c_[11]*Rn)*cyl_neumann( 1, c_[11]*Rn) + c_[11]/2./Rn*(cyl_neumann( 0, c_[11]*Rn) - cyl_neumann( 2,c_[11]*Rn));
        double j1_csR = (1./Rn-cs_*cs_*Rn)*cyl_bessel_j(1, cs_*Rn) + cs_/2./Rn*(cyl_bessel_j(0, cs_*Rn) - cyl_bessel_j(2,cs_*Rn));
        double y1_csR = (1./Rn-cs_*cs_*Rn)*cyl_neumann( 1, cs_*Rn) + cs_/2./Rn*(cyl_neumann( 0, cs_*Rn) - cyl_neumann( 2,cs_*Rn));
        double RZbar = sqrt(Rn*Rn+Zn*Zn);
        double cosR = -c_[11]/(RZbar*RZbar)*(c_[11]*Rn*Rn*cos(c_[11]*RZbar) +Zn*Zn*sin(c_[11]*RZbar)/RZbar);
        return  R0_*(   
                   1.0*j1_c12R 
               + c_[0]*yn_c12R 
               + c_[1]*j1_csR*cos(c_[10]*Zn)  
               + c_[2]*y1_csR*cos(c_[10]*Zn)  
               + c_[3]*cosR;
               + c_[5]*j1_c12R*Zn
               + c_[6]*y1_c12R*Zn
               + c_[7]*j1_csR*sin(c_[10]*Zn)
               + c_[8]*y1_csR*sin(c_[10]*Zn) );
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
  private:
    double R0_, cs_;
    std::vector<double> c_;
};
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\f]
 */ 
struct PsipZ
{
    PsipZ( GeomParameters gp ): R0_(gp.R_0), c_(gp.c) { 
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double operator()(double R, double Z) const
    {    
        double Rn = R/R0_, Zn = Z/Z_0_;
        double j1_c12 = cyl_bessel_j( 1, c_[11]*Rn);
        double y1_c12 = cyl_neumann( 1, c_[11]*Rn);
        double j1_cs = cyl_bessel_j( 1, cs_*Rn);
        double y1_cs = cyl_neumann( 1, cs_*Rn);
        return R0_*(    
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
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    double R0_,cs_; 
    std::vector<double> c_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}\f]
 */ 
struct PsipZZ
{
  PsipZZ( GeomParameters gp): R0_(gp.R_0), c_(gp.c) { 
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double operator()(double R, double Z) const
    {    
        double Rn = R/R0_, Zn = Z/Z_0_;
        double j1_c12 = cyl_bessel_j( 1, c_[11]*Rn);
        double y1_c12 = cyl_neumann( 1, c_[11]*Rn);
        double j1_cs = cyl_bessel_j( 1, cs_*Rn);
        double y1_cs = cyl_neumann( 1, cs_*Rn);
        double RZbar = sqrt(Rn*Rn+Zn*Zn);
        double cosZ = -c_[11]/(RZbar*RZbar)*(c_[11]*Zn*Zn*cos(c_[11]*RZbar) +Rn*Rn*sin(c_[11]*RZbar)/RZbar);
        return R0_*(    
               - c_[1]*Rn*j1_cs*c_[10]*c_[10]*cos(c_[10]*Zn)  
               - c_[2]*Rn*y1_cs*c_[10]*c_[10]*cos(c_[10]*Zn)  
               + c_[3]*cosZ
               - c_[4]*c_[11]*c_[11]*cos(c_[11]*Zn)
               - c_[7]*Rn*j1_cs*c_[10]*c_[10]*sin(c_[10]*Zn)
               - c_[8]*Rn*y1_cs*c_[10]*c_[10]*sin(c_[10]*Zn)
               - c_[9]*c_[11]*c_[11]*sin(c_[11]*Zn));
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    double R0_, cs_;
    std::vector<double> c_;
};
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\f] 
 */ 
struct PsipRZ
{
    PsipRZ( GeomParameters gp ): R0_(gp.R_0), c_(gp.c) { 
        cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
    }
    double operator()(double R, double Z) const
    {    
        double Rn=R/R0_, Zn=Z/Z_0_;
        double j1_c12R = cyl_bessel_j(1, c_[11]*Rn) + c_[11]/2.*Rn*(cyl_bessel_j(0, c_[11]*Rn) - cyl_bessel_j(2,c_[11]*Rn));
        double y1_c12R = cyl_neumann(1, c_[11]*Rn) + c_[11]/2.*Rn*(cyl_neumann(0, c_[11]*Rn) - cyl_neumann(2,c_[11]*Rn));
        double j1_csR = cyl_bessel_j(1, cs_*Rn) + cs_/2.*Rn*(cyl_bessel_j(0, cs_*Rn) - cyl_bessel_j(2, cs_*Rn));
        double y1_csR = cyl_neumann(1, cs_*Rn) + cs_/2.*Rn*(cyl_neumann(0, cs_*Rn) - cyl_neumann(2, cs_*Rn));
        double RZbar = sqrt(Rn*Rn+Zn*Zn);
        double cosRZ = -c_[11]*Rn*Zn/(RZbar*RZbar*RZbar)*(c_[11]*RZbar*cos(c_[11]*RZbar) -sin(c_[11]*RZbar));
        return  R0_*(   
               - c_[1]*j1_csR*c_[10]*sin(c_[10]*Zn)  
               - c_[2]*y1_csR*c_[10]*sin(c_[10]*Zn)  
               + c_[3]*cosRZ
               + c_[5]*j1_c12R
               + c_[6]*y1_c12R
               + c_[7]*j1_csR*c_[10]*cos(c_[10]*Zn)
               + c_[8]*y1_csR*c_[10]*cos(c_[10]*Zn) );
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    double R0_, cs_;
    std::vector<double> c_;
};

struct LaplacePsip
{
    LaplacePsip( GeomParameters gp ): psipRR_(gp), psipZZ_(gp){}
    double operator()(double R, double Z) const
    {    
        return psipRR_(R,Z) + psipZZ_(R,Z);
    }
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    PsipRR psipRR_;
    PsipZZ psipZZ_;
};


/**
 * @brief \f[\hat{I}\f] 
 */ 
struct Ipol
{
    Ipol(  GeomParameters gp ):  R0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp) { }
    /**
    * @brief \f[\hat{I}= \sqrt{-2 A \hat{\psi}_p / \hat{R}_0 +1}\f] 
    */ 
    double operator()(double R, double Z) const
    {    
        //sign before A changed to -
        return qampl_*sqrt(-2.*A_* psip_(R,Z) /R0_ + 1.);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()( R,Z);
    }
  private:
    double R0_, A_,qampl_;
    Psip psip_;
};
struct IpolR
{
    IpolR(  GeomParameters gp ):  R0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp), psipR_(gp) { }
    double operator()(double R, double Z) const
    {    
        return -qampl_/sqrt(-2.*A_* psip_(R,Z) /R0_ + 1.)*(A_*psipR_(R,Z)/R0_);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()( R,Z);
    }
  private:
    double R0_, A_,qampl_;
    Psip psip_;
    PsipR psipR_;
};
struct IpolZ
{
    IpolZ(  GeomParameters gp ):  R0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp), psipZ_(gp) { }
    double operator()(double R, double Z) const
    {    
        return -qampl_/sqrt(-2.*A_* psip_(R,Z) /R0_ + 1.)*(A_*psipZ_(R,Z)/R0_);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()( R,Z);
    }
  private:
    double R0_, A_,qampl_;
    Psip psip_;
    PsipZ psipZ_;
};

} //namespace taylor


#include "fields.h"
