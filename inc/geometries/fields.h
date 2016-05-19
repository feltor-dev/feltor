#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/backend/functions.h"


/*!@file
 *
 * Geometry objects 
 */
namespace solovev
{
///@addtogroup geom
///@{

/**
 * @brief \f[   \hat{B}   \f]
 */ 
struct Bmodule
{
    Bmodule(  GeomParameters gp ):  R_0_(gp.R_0), ipol_(gp), psipR_(gp), psipZ_(gp)  { }
    /**
    * @brief \f[   \hat{B} \f]
    */ 
    double operator()(double R, double Z) const
    {    
        double psipR = psipR_(R,Z), psipZ = psipZ_(R,Z), ipol = ipol_(R,Z);
        return R_0_/R*sqrt(ipol*ipol+psipR*psipR +psipZ*psipZ);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    double R_0_;
    Ipol ipol_;
    PsipR psipR_;
    PsipZ psipZ_;  
};

/**
 * @brief \f[   \frac{1}{\hat{B}}   \f]
 */ 
struct InvB
{
    InvB(  GeomParameters gp ):  R_0_(gp.R_0), ipol_(gp), psipR_(gp), psipZ_(gp)  { }
    /**
    * @brief \f[   \frac{1}{\hat{B}} = 
        \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
        + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
    */ 
    double operator()(double R, double Z) const
    {    
        double psipR = psipR_(R,Z), psipZ = psipZ_(R,Z), ipol = ipol_(R,Z);
        return R/(R_0_*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    double R_0_;
    Ipol ipol_;
    PsipR psipR_;
    PsipZ psipZ_;  
};

/**
 * @brief \f[   \ln{(   \hat{B})}  \f]
 */ 
struct LnB
{
    LnB( GeomParameters gp ):  R_0_(gp.R_0), ipol_(gp), psipR_(gp), psipZ_(gp)  { }
/**
 * @brief \f[   \ln{(   \hat{B})} = \ln{\left[
      \frac{\hat{R}_0}{\hat{R}} \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2} \right] } \f]
 */ 
    double operator()(double R, double Z) const
    {    
        double psipR = psipR_(R,Z), psipZ = psipZ_(R,Z), ipol = ipol_(R,Z);
        return log(R_0_/R*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    double R_0_;
    Ipol ipol_;
    PsipR psipR_;
    PsipZ psipZ_;  
};
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{R}}  \f]
 */ 

struct BR
{
    BR(GeomParameters gp):  R_0_(gp.R_0), ipol_(gp), ipolR_(gp), psipR_(gp), psipRR_(gp),psipZ_(gp) ,psipRZ_(gp), invB_(gp) { }
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{R}} = 
      -\frac{\hat{R}^2\hat{R}_0^{-2} \hat{B}^2+A\hat{R} \hat{R}_0^{-1}   \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)  
      - \hat{R} \left[  \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
      + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)\left( \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\right)\right] }{\hat{R}^3 \hat{R}_0^{-2}\hat{B}} \f]
 */ 
    double operator()(double R, double Z) const
    { 
        double Rn;
        Rn = R/R_0_;
        //sign before A changed to +
        //return -( Rn*Rn/invB_(R,Z)/invB_(R,Z)+ qampl_*qampl_*Rn *A_*psipR_(R,Z) - R  *(psipZ_(R,Z)*psipRZ_(R,Z)+psipR_(R,Z)*psipRR_(R,Z)))/(R*Rn*Rn/invB_(R,Z));
        return -1./R/invB_(R,Z) + invB_(R,Z)/Rn/Rn*(ipol_(R,Z)*ipolR_(R,Z) + psipR_(R,Z)*psipRR_(R,Z) + psipZ_(R,Z)*psipRZ_(R,Z));
    }
      /**
       * @brief == operator()(R,Z)
       */ 
    double operator()(double R, double Z, double phi)const{return operator()(R,Z);}
  private:
    double R_0_;
    Ipol ipol_;
    IpolR ipolR_;
    PsipR psipR_;
    PsipRR psipRR_;
    PsipZ psipZ_;
    PsipRZ psipRZ_;  
    InvB invB_;
};
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}}  \f]
 */ 
struct BZ
{

    BZ( GeomParameters gp):  R_0_(gp.R_0), ipol_(gp), ipolZ_(gp), psipR_(gp),psipZ_(gp), psipZZ_(gp) ,psipRZ_(gp), invB_(gp) { }
    /**
     * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}} = 
     \frac{-A \hat{R}_0^{-1}    \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}   \right)+
     \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
          + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2} \right)}{\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f]
     */ 
    double operator()(double R, double Z) const
    { 
        double Rn;
        Rn = R/R_0_;
        //sign before A changed to -
        //return (-qampl_*qampl_*A_/R_0_*psipZ_(R,Z) + psipR_(R,Z)*psipRZ_(R,Z)+psipZ_(R,Z)*psipZZ_(R,Z))/(Rn*Rn/invB_(R,Z));
        return (invB_(R,Z)/Rn/Rn)*(ipol_(R,Z)*ipolZ_(R,Z) + psipR_(R,Z)*psipRZ_(R,Z) + psipZ_(R,Z)*psipZZ_(R,Z));
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi)const{return operator()(R,Z);}
  private:
    double R_0_;
    Ipol ipol_;
    IpolZ ipolZ_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipZZ psipZZ_;
    PsipRZ psipRZ_;  
    InvB invB_; 
};


/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}  \f]
 */ 
struct CurvatureR
{
    CurvatureR( GeomParameters gp):
        gp_(gp),
        psip_(gp),
        psipR_(gp), psipZ_(gp),
        psipZZ_(gp), psipRZ_(gp),
        ipol_(gp),
        invB_(gp),
        bZ_(gp) { }
    /**
     * @brief \f[ \mathcal{\hat{K}}^{\hat{R}} =-\frac{1}{ \hat{B}^2}  \frac{\partial \hat{B}}{\partial \hat{Z}}  \f]
     */ 
    double operator()( double R, double Z) const
    {
        return -2.*invB_(R,Z)*invB_(R,Z)*bZ_(R,Z); //factor 2 stays under discussion
//         return -ipol_(R,Z)*invB_(R,Z)*invB_(R,Z)*invB_(R,Z)*bZ_(R,Z)*gp_.R_0/R; //factor 2 stays under discussion
    }
    
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return -2.*invB_(R,Z,phi)*invB_(R,Z,phi)*bZ_(R,Z,phi); //factor 2 stays under discussion
//         return -ipol_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*bZ_(R,Z,phi)*gp_.R_0/R; //factor 2 stays under discussion
    }
    private:    
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipZ  psipZ_;
    PsipZZ psipZZ_;
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BZ bZ_;    
};
/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}  \f]
 */ 
struct CurvatureZ
{
    CurvatureZ( GeomParameters gp):
        gp_(gp),
        psip_(gp),
        psipR_(gp),
        psipRR_(gp),
        psipZ_(gp),
        psipRZ_(gp),
        ipol_(gp),
        invB_(gp),
        bR_(gp) { }
 /**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}} =\frac{1}{ \hat{B}^2}   \frac{\partial \hat{B}}{\partial \hat{R}} \f]
 */    
    double operator()( double R, double Z) const
    {
        return 2.*invB_(R,Z)*invB_(R,Z)*bR_(R,Z); //factor 2 stays under discussion
//         return  ipol_(R,Z)*invB_(R,Z)*invB_(R,Z)*invB_(R,Z)*bR_(R,Z)*gp_.R_0/R; //factor 2 stays under discussion
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return 2.*invB_(R,Z,phi)*invB_(R,Z,phi)*bR_(R,Z,phi); //factor 2 stays under discussion
//         return ipol_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*bR_(R,Z,phi)*gp_.R_0/R; //factor 2 stays under discussion
    }
    private:    
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipRR  psipRR_;
    PsipZ  psipZ_;    
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BR bR_;   
};

/**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} \f]
 */ 
struct GradLnB
{
    GradLnB( GeomParameters gp):
        gp_(gp),
        psipR_(gp),
        psipZ_(gp),
        invB_(gp),
        bR_(gp), 
        bZ_(gp) {
    } 
    /**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} = \frac{1}{\hat{R}\hat{B}^2 } \left[ \hat{B}, \hat{\psi}_p\right]_{\hat{R}\hat{Z}} \f]
 */ 
    double operator()( double R, double Z) const
    {
        double invB = invB_(R,Z);
        return gp_.R_0*invB*invB*(bR_(R,Z)*psipZ_(R,Z)-bZ_(R,Z)*psipR_(R,Z))/R ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi)const{return operator()(R,Z);}
    private:
    GeomParameters gp_;
    PsipR  psipR_;
    PsipZ  psipZ_;    
    InvB   invB_;
    BR bR_;
    BZ bZ_;   
};
/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
struct Field
{
    Field( GeomParameters gp):
        gp_(gp),
        psipR_(gp),
        psipZ_(gp),
        ipol_(gp),
        invB_(gp) {
    }
    /**
     * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
     \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
     \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I} \hat{R}_0}  \f]
     */ 
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp) const
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            yp[2][i] =  y[0][i]*y[0][i]/invB_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i])/gp_.R_0;//ds/dphi =  R^2 B/I/R_0_hat
            yp[0][i] =  y[0][i]*psipZ_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i]);               //dR/dphi =  R/I Psip_Z
            yp[1][i] = -y[0][i]*psipR_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i]) ;              //dZ/dphi = -R/I Psip_R
        }
    }
    /**
     * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
     \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
     \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I}  \hat{R}_0}  \f]
     */ 
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double ipol = ipol_(y[0],y[1]);
        yp[2] =  y[0]*y[0]/invB_(y[0],y[1])/ipol/gp_.R_0;       //ds/dphi =  R^2 B/I/R_0_hat
        yp[0] =  y[0]*psipZ_(y[0],y[1])/ipol;              //dR/dphi =  R/I Psip_Z
        yp[1] = -y[0]*psipR_(y[0],y[1])/ipol ;             //dZ/dphi = -R/I Psip_R

    }
    /**
     * @brief \f[   \frac{1}{\hat{B}} = 
      \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
     */ 
    double operator()( double R, double Z) const
    {
        //modified
//          return invB_(R,Z)* invB_(R,Z)*ipol_(R,Z)*gp_.R_0/R;
        return invB_(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z,phi);

//         return invB_(R,Z,phi)*invB_(R,Z,phi)*ipol_(R,Z,phi)*gp_.R_0/R;
    }
    
    private:
    GeomParameters gp_;
    PsipR  psipR_;
    PsipZ  psipZ_;
    Ipol   ipol_;
    InvB   invB_;
   
};

/**
 * @brief Phi component of magnetic field \f$ B_\Phi\f$
*/
struct FieldP
{
    FieldP( GeomParameters gp): R_0(gp.R_0), 
    ipol_(gp){}
    double operator()( double R, double Z, double phi) const
    {
        return R_0*ipol_(R,Z)/R/R;
    }
    
    private:
    double R_0;
    Ipol   ipol_;
   
}; 
/**
 * @brief R component of magnetic field\f$ B_R\f$
 */
struct FieldR
{
    FieldR( GeomParameters gp): psipZ_(gp), R_0(gp.R_0){ }
    double operator()( double R, double Z) const
    {
        return  R_0/R*psipZ_(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return  this->operator()(R,Z);
    }
    private:
    PsipZ  psipZ_;
    double R_0;
   
};
/**
 * @brief Z component of magnetic field \f$ B_Z\f$
 */
struct FieldZ
{
    FieldZ( GeomParameters gp): psipR_(gp), R_0(gp.R_0){ }
    double operator()( double R, double Z) const
    {
        return  -R_0/R*psipR_(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return  this->operator()(R,Z);
    }
    private:
    PsipR  psipR_;
    double R_0;
   
};

/**
 * @brief \f$\theta\f$ component of magnetic field 
 */ 
struct FieldT
{
    FieldT( GeomParameters gp):  R_0_(gp.R_0), fieldR_(gp), fieldZ_(gp){}
  /**
 * @brief \f[  B^{\theta} = 
 * B^R\partial_R\theta + B^Z\partial_Z\theta\f]
 * where \f$ \theta \f$ is the geometrical poloidal angle.
 */ 
  double operator()(double R, double Z) const
  { 
      double r2 = (R-R_0_)*(R-R_0_) + Z*Z;
      return fieldR_(R,Z)*(-Z/r2) + fieldZ_(R,Z)*(R-R_0_)/r2;
  }
    /**
     * @brief == operator()(R,Z)
     */ 
  double operator()(double R, double Z, double phi) const
  { 
      return this->operator()(R,Z);
  }
  private:
    double R_0_;
    FieldR fieldR_;
    FieldZ fieldZ_;

};

namespace flux{

/**
 * @brief y-component of magnetic field 
 */ 
struct FieldY
{
    FieldY( GeomParameters gp): f_psi_(1.), R_0_(gp.R_0),ipol_(gp){}
    void set_f( double f_new){f_psi_=f_new;}
  /**
 * @brief \f[  B^y= 
 * f(\psi) \frac{(\nabla\psi)^2}{R} \f]
 */ 
    double operator()(double R, double Z) const
    { 
        double ipol=ipol_(R, Z);
        return f_psi_*ipol*R_0_/R/R;
        
    }
      /**
       * @brief == operator()(R,Z)
       */ 
    double operator()(double R, double Z, double phi) const
    { 
        return this->operator()(R,Z);
    }
    private:
    double f_psi_, R_0_;
    Ipol ipol_;
};
/**
 * @brief 
 * \f[  d R/d \theta =   B^R/B^\theta \f], 
 * \f[  d Z/d \theta =   B^Z/B^\theta \f],
 * \f[  d y/d \theta =   B^y/B^\theta\f]
 */ 
struct FieldRZYT
{
    FieldRZYT( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp),ipol_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double ipol=ipol_(y[0], y[1]);
        yp[0] =  R_0_/y[0]*psipZ;//fieldR
        yp[1] = -R_0_/y[0]*psipR;//fieldZ
        yp[2] =ipol*R_0_/y[0]/y[0]; //fieldYbar
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + y[1]*y[1];
        double fieldT = yp[0]*(-y[1]/r2) + yp[1]*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
    Ipol ipol_;
};

struct FieldRZYZ
{
    FieldRZYZ( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp),ipol_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double ipol=ipol_(y[0], y[1]);

        yp[0] =  psipZ;//fieldR
        yp[1] = -psipR;//fieldZ
        yp[2] = ipol; //fieldYbar
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
    Ipol ipol_;
};
/**
 * @brief 
 * \f[  d R/d y =   B^R/B^y \f], 
 * \f[  d Z/d y =   B^Z/B^y \f],
 */ 
struct FieldRZY
{
    FieldRZY( GeomParameters gp): f_(1.), R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void set_f(double f){ f_ = f;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] = +psipZ/(psipR*psipR+psipZ*psipZ);//fieldR
        yp[1] = -psipR/(psipR*psipR+psipZ*psipZ);//fieldZ
    }
  private:
    double f_;
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};
/**
 * @brief 
 * \f[  d R/d y   =   B^R/B^y \f], 
 * \f[  d Z/d y   =   B^Z/B^y \f],
 * \f[  d y_R/d y =   .. \f], 
 * \f[  d y_Z/d y =   ..\f],
 */ 
struct FieldRZYRYZY
{
    FieldRZYRYZY( const GeomParameters& gp): psipR_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp), ipol_(gp), ipolR_(gp), ipolZ_(gp)
    { f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psipRR = psipRR_(y[0], y[1]), psipRZ = psipRZ_(y[0],y[1]), psipZZ = psipZZ_(y[0],y[1]);
//         double psip2 = 0.0+1.0*(psipR*psipR+ psipZ*psipZ);
        double ipol=ipol_(y[0], y[1]);
        double ipolR=ipolR_(y[0], y[1]);
        double ipolZ=ipolZ_(y[0], y[1]);

        yp[0] =  (y[0]/ipol/f_)*(psipZ);
        yp[1] =  -(y[0]/ipol/f_)*(psipR);
        //not all terms included in yp[2] and yp[3]
        yp[2] =  (y[0]/ipol/f_)*(-psipRZ*y[2]+ psipRR*y[3])+
         + f_prime_/f_* psipR 
         + ipolR/ipol
         -1./y[0] ;
        yp[3] =  (y[0]/ipol/f_)*( psipRZ*y[3]- psipZZ*y[2])+
         + f_prime_/f_* psipZ 
         + ipolZ/ipol;
    }
  private:
    double f_, f_prime_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipRR psipRR_;
    PsipRZ psipRZ_;
    PsipZZ psipZZ_;
    Ipol ipol_;
    IpolR ipolR_;
    IpolZ ipolZ_;

};
}//namespace flux

namespace conformal{

/**
 * @brief y-component of magnetic field 
 */ 
struct FieldY
{
    FieldY( GeomParameters gp): f_psi_(1.), R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void set_f( double f_new){f_psi_=f_new;}
  /**
 * @brief \f[  B^y= 
 * f(\psi) \frac{(\nabla\psi)^2}{R} \f]
 */ 
    double operator()(double R, double Z) const
    { 
        double psipR = psipR_(R, Z), psipZ = psipZ_(R,Z);
        double psi2 = 0.0+1.0*(psipR*psipR+ psipZ*psipZ);
        return f_psi_*(psi2)*R_0_/R;
        //return (psi2)*R_0_/R/f_psi_;
    }
      /**
       * @brief == operator()(R,Z)
       */ 
    double operator()(double R, double Z, double phi) const
    { 
        return this->operator()(R,Z);
    }
    private:
    double f_psi_, R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZYT
{
    FieldRZYT( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] =  R_0_/y[0]*psipZ;//fieldR
        yp[1] = -R_0_/y[0]*psipR;//fieldZ
        //yp[2] = (1.0))*R_0_/y[0]; //fieldYbar
        //yp[2] = sqrt(psipR*psipR+psipZ*psipZ)*R_0_/y[0]; //fieldYbar
        yp[2] =(psipR*psipR+psipZ*psipZ)*R_0_/y[0]; //fieldYbar
        //yp[2] =1.0*R_0_/y[0]/y[0]; //fieldYbar
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + y[1]*y[1];
        double fieldT = yp[0]*(-y[1]/r2) + yp[1]*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZYZ
{
    FieldRZYZ( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] =  psipZ;//fieldR
        yp[1] = -psipR;//fieldZ
        //yp[2] = (1.0+0.0*(psipR*psipR+psipZ*psipZ)); //fieldYbar
        //yp[2] = sqrt(psipR*psipR+psipZ*psipZ); //fieldYbar
        yp[2] = (psipR*psipR+psipZ*psipZ); //fieldYbar
        //yp[2] = (1.0)/y[0]; //fieldYbar
        //double r2 = (y[0]-R_0_)*(y[0]-R_0_) + y[1]*y[1];
        //double fieldT = yp[0]*(-y[1]/r2) + yp[1]*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZY
{
    FieldRZY( GeomParameters gp): f_(1.), R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void set_f(double f){ f_ = f;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        //yp[0] = +psipZ/f_;//fieldR
        //yp[1] = -psipR/f_;//fieldZ
        //yp[0] = +psipZ/sqrt(psipR*psipR+psipZ*psipZ)/f_;//fieldR
        //yp[1] = -psipR/sqrt(psipR*psipR+psipZ*psipZ)/f_;//fieldZ
        yp[0] = +psipZ/(psipR*psipR+psipZ*psipZ)/f_;//fieldR
        yp[1] = -psipR/(psipR*psipR+psipZ*psipZ)/f_;//fieldZ
        //yp[0] = +psipZ*y[0]/f_;//fieldR
        //yp[1] = -psipR*y[0]/f_;//fieldZ
    }
  private:
    double f_;
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZYRYZY
{
    FieldRZYRYZY( const GeomParameters& gp): psipR_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp){ f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psipRR = psipRR_(y[0], y[1]), psipRZ = psipRZ_(y[0],y[1]), psipZZ = psipZZ_(y[0],y[1]);
        double psip2 = 0.0+1.0*(psipR*psipR+ psipZ*psipZ);

        yp[0] =  psipZ/f_/psip2;
        yp[1] =  -psipR/f_/psip2;
        yp[2] =  (( - psipRZ )*y[2] 
                   +( psipRR )*y[3] )/f_/psip2 
            + f_prime_/f_* psipR + 2.*1.0*(psipR*psipRR + psipZ*psipRZ)/psip2 ;
        yp[3] =  (psipRZ*y[3] - psipZZ*y[2])/f_/psip2 
            + f_prime_/f_* psipZ + 2.*1.0*(psipR*psipRZ + psipZ*psipZZ)/psip2;
        //yp[0] =  psipZ*f_;//psip2;
        //yp[1] =  -psipR*f_;//psip2;
        //yp[2] =  ( (psipZ/y[0] - psipRZ )*y[2] 
        //           +( psipRR - psipR/y[0])*y[3] )*f_/psip2 
        //    - f_prime_/f_* psipR + 2.*0.0*(psipR*psipRR + psipZ*psipRZ)/psip2 -1./y[0];
        //yp[3] =  (psipRZ*y[3] - psipZZ*y[2])*f_/psip2 
        //    - f_prime_/f_* psipZ + 2.*0.0*(psipR*psipRZ + psipZ*psipZZ)/psip2;
    }
  private:
    double f_, f_prime_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipRR psipRR_;
    PsipRZ psipRZ_;
    PsipZZ psipZZ_;
};
}//namespace conformal

namespace orthogonal{

struct FieldRZYT
{
    FieldRZYT( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] =  R_0_/y[0]*psipZ;//fieldR
        yp[1] = -R_0_/y[0]*psipR;//fieldZ
        //yp[2] = (1.0))*R_0_/y[0]; //fieldYbar
        //yp[2] = sqrt(psipR*psipR+psipZ*psipZ)*R_0_/y[0]; //fieldYbar
        yp[2] =(psipR*psipR+psipZ*psipZ)*R_0_/y[0]; //fieldYbar
        //yp[2] =1.0*R_0_/y[0]/y[0]; //fieldYbar
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + y[1]*y[1];
        double fieldT = yp[0]*(-y[1]/r2) + yp[1]*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZYZ
{
    FieldRZYZ( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] =  psipZ;//fieldR
        yp[1] = -psipR;//fieldZ
        //yp[2] = (1.0+0.0*(psipR*psipR+psipZ*psipZ)); //fieldYbar
        //yp[2] = sqrt(psipR*psipR+psipZ*psipZ); //fieldYbar
        yp[2] = (psipR*psipR+psipZ*psipZ); //fieldYbar
        //yp[2] = (1.0)/y[0]; //fieldYbar
        //double r2 = (y[0]-R_0_)*(y[0]-R_0_) + y[1]*y[1];
        //double fieldT = yp[0]*(-y[1]/r2) + yp[1]*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZY
{
    FieldRZY( GeomParameters gp): f_(1.), R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void set_f(double f){ f_ = f;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        //yp[0] = +psipZ/f_;//fieldR
        //yp[1] = -psipR/f_;//fieldZ
        //yp[0] = +psipZ/sqrt(psipR*psipR+psipZ*psipZ)/f_;//fieldR
        //yp[1] = -psipR/sqrt(psipR*psipR+psipZ*psipZ)/f_;//fieldZ
        yp[0] = +psipZ/(psipR*psipR+psipZ*psipZ)/f_;//fieldR
        yp[1] = -psipR/(psipR*psipR+psipZ*psipZ)/f_;//fieldZ
        //yp[0] = +psipZ*y[0]/f_;//fieldR
        //yp[1] = -psipR*y[0]/f_;//fieldZ
    }
  private:
    double f_;
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

}//namespace orthogonal

struct FieldRZ
{
    FieldRZ( GeomParameters gp): psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] =  psipR/psipZ;
        yp[1] = 1.;
    }
  private:
    PsipR psipR_;
    PsipZ psipZ_;
};
struct FieldZR
{
    FieldZR( GeomParameters gp): psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] = 1.;
        yp[1] =  psipZ/psipR;
    }
  private:
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZtau
{
    FieldRZtau( GeomParameters gp): psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psi2 = psipR*psipR+ psipZ*psipZ;
        yp[0] =  psipR/psi2;
        yp[1] =  psipZ/psi2;
    }
  private:
    PsipR psipR_;
    PsipZ psipZ_;
};

struct HessianRZtau
{
    HessianRZtau( GeomParameters gp): norm_(false), quad_(1), psipR_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp){}
    // if true goes into positive Z - direction and X else
    void set_quadrant( int quadrant) {quad_ = quadrant;}
    void set_norm( bool normed) {norm_ = normed;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipRZ = psipRZ_(y[0], y[1]);
        if( psipRZ == 0)
        {
            if(      quad_ == 0) { yp[0] = 1; yp[1] = 0; }
            else if( quad_ == 1) { yp[0] = 0; yp[1] = 1; }
            else if( quad_ == 2) { yp[0] = -1; yp[1] = 0; }
            else if( quad_ == 3) { yp[0] = 0; yp[1] = -1; }
        }
        else
        {
            double psipRR = psipRR_(y[0], y[1]), psipZZ = psipZZ_(y[0],y[1]);
            double T = psipRR + psipZZ; 
            double D = psipZZ*psipRR - psipRZ*psipRZ;
            double L1 = 0.5*T+sqrt( 0.25*T*T-D); // > 0
            double L2 = 0.5*T-sqrt( 0.25*T*T-D); // < 0;  D = L1*L2
            if      ( quad_ == 0){ yp[0] =  L1 - psipZZ; yp[1] = psipRZ;}
            else if ( quad_ == 1){ yp[0] = -psipRZ; yp[1] = psipRR - L2;}
            else if ( quad_ == 2){ yp[0] =  psipZZ - L1; yp[1] = -psipRZ;}
            else if ( quad_ == 3){ yp[0] = +psipRZ; yp[1] = L2 - psipRR;}
        }
        if( norm_) 
        {
            double vgradpsi = yp[0]*psipR_(y[0],y[1]) + yp[1]*psipZ_(y[0],y[1]);
            yp[0] /= vgradpsi, yp[1] /= vgradpsi;
        }
        else
        {
            double norm = sqrt(yp[0]*yp[0]+yp[1]*yp[1]);
            yp[0]/= norm, yp[1]/= norm;
        }

    }
    void newton_iteration( const dg::HVec&y, dg::HVec& yp)
    {
        double psipRZ = psipRZ_(y[0], y[1]);
        double psipRR = psipRR_(y[0], y[1]), psipZZ = psipZZ_(y[0],y[1]);
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0], y[1]);
        double Dinv = 1./(psipZZ*psipRR - psipRZ*psipRZ);
        yp[0] = y[0] - Dinv*(psipZZ*psipR - psipRZ*psipZ);
        yp[1] = y[1] - Dinv*(-psipRZ*psipR + psipRR*psipZ);
    }
  private:
    bool norm_;
    int quad_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipRR psipRR_;
    PsipRZ psipRZ_;
    PsipZZ psipZZ_;
};

struct MinimalCurve
{
    MinimalCurve( GeomParameters gp): norm_(false), psip_(gp), psipR_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp){}
    void set_norm( bool normed) {norm_ = normed;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        //double psipRZ = psipRZ_(y[0], y[1]), psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0], y[1]), psipRR=psipRR_(y[0], y[1]), psipZZ=psipZZ_(y[0], y[1]); 
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0], y[1]);
        yp[0] = y[2];
        yp[1] = y[3];
        //double D2 = psipRR*y[2]*y[2] + 2.*psipRZ*y[2]*y[3] + psipZZ*y[3]*y[3];
        //double grad2 = psipR*psipR+psipZ*psipZ;
        //yp[2] = D2/(1.+grad2) * psipR ;
        //yp[3] = D2/(1.+grad2) * psipZ ;
        if( psip_(y[0], y[1]) < 0)
        {
            yp[2] = -10.*psipR;
            yp[3] = -10.*psipZ;
        }
        else
        {
            yp[2] = 10.*psipR;
            yp[3] = 10.*psipZ;
        }

        if( norm_) 
        {
            double vgradpsi = y[2]*psipR + y[3]*psipZ;
            yp[0] /= vgradpsi, yp[1] /= vgradpsi, yp[2] /= vgradpsi, yp[3] /= vgradpsi;
        }
    }
  private:
    bool norm_;
    Psip psip_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipRR psipRR_;
    PsipRZ psipRZ_;
    PsipZZ psipZZ_;
};

/**
 * @brief R component of magnetic field\f$ b_R\f$
 */
struct BHatR
{
    BHatR( GeomParameters gp): 
        psipZ_(gp), R_0(gp.R_0),
        invB_(gp){ }
    double operator()( double R, double Z, double phi) const
    {
        return  invB_(R,Z)*R_0/R*psipZ_(R,Z);
    }
    private:
    PsipZ  psipZ_;
    double R_0;
    InvB   invB_;

};
/**
 * @brief Z component of magnetic field \f$ b_Z\f$
 */
struct BHatZ
{
    BHatZ( GeomParameters gp): 
        psipR_(gp), R_0(gp.R_0),
        invB_(gp){ }

    double operator()( double R, double Z, double phi) const
    {
        return  -invB_(R,Z)*R_0/R*psipR_(R,Z);
    }
    private:
    PsipR  psipR_;
    double R_0;
    InvB   invB_;

};
/**
 * @brief Phi component of magnetic field \f$ b_\Phi\f$
 */
struct BHatP
{
    BHatP( GeomParameters gp):
        R_0(gp.R_0), 
        ipol_(gp),
        invB_(gp){}
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z)*R_0*ipol_(R,Z)/R/R;
    }
    
    private:
    double R_0;
    Ipol   ipol_;
    InvB   invB_;
  
}; 
///@} 
///@addtogroup profiles
///@{

/**
 * @brief Delta function for poloidal flux \f$ B_Z\f$
     \f[ |\nabla \psi_p|\delta(\psi_p(R,Z)-\psi_0) = \frac{\sqrt{ (\nabla \psi_p)^2}}{\sqrt{2\pi\varepsilon}} \exp\left(-\frac{(\psi_p(R,Z) - \psi_{0})^2}{2\varepsilon} \right)  \f]
 */
struct DeltaFunction
{
    DeltaFunction(GeomParameters gp, double epsilon,double psivalue) :
        psip_(gp),
        psipR_(gp), 
        psipZ_(gp), 
        epsilon_(epsilon),
        psivalue_(psivalue){
    }
    /**
    * @brief Set a new \f$ \varepsilon\f$
    *
    * @param eps new value
    */
    void setepsilon(double eps ){epsilon_ = eps;}
    /**
    * @brief Set a new \f$ \psi_0\f$
    *
    * @param psi_0 new value
    */
    void setpsi(double psi_0 ){psivalue_ = psi_0;}

    /**
     *@brief \f[ \frac{\sqrt{ (\nabla \psi_p)^2}}{\sqrt{2\pi\varepsilon}} \exp\left(-\frac{(\psi_p(R,Z) - \psi_{0})^2}{2\varepsilon} \right)  \f]
     */
    double operator()( double R, double Z) const
    {
        return 1./sqrt(2.*M_PI*epsilon_)*
               exp(-( (psip_(R,Z)-psivalue_)* (psip_(R,Z)-psivalue_))/2./epsilon_)*sqrt(psipR_(R,Z)*psipR_(R,Z) +psipZ_(R,Z)*psipZ_(R,Z));
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return (*this)(R,Z);
    }
    private:
    Psip psip_;
    PsipR  psipR_;
    PsipZ  psipZ_;
    double epsilon_;
    double psivalue_;
};

/**
 * @brief Flux surface average over quantity
 *
 * The average is computed using the formula
 \f[ <f>(\psi_0) = \frac{1}{A} \int dV \delta(\psi_p(R,Z)-\psi_0) |\nabla\psi_p|f(R,Z) \f]
 with \f$ A = \int dV \delta(\psi_p(R,Z)-\psi_0)|\nabla\psi_p|\f$
 * @tparam container  The container class of the vector to average

 */
template <class container = thrust::host_vector<double> >
struct FluxSurfaceAverage
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param gp  geometry parameters
     * @param f container for global safety factor
     */
    FluxSurfaceAverage(const dg::Grid2d<double>& g2d, GeomParameters gp,   const container& f) :
    g2d_(g2d),
    gp_(gp),
    f_(f),
    psip_(gp),
    psipR_(gp),
    psipZ_(gp),
    deltaf_(DeltaFunction(gp,0.0,0.0)),
    w2d_ ( dg::create::weights( g2d_)),
    oneongrid_(dg::evaluate(dg::one,g2d_))              
    {
        dg::HVec psipRog2d  = dg::evaluate( psipR_, g2d_);
        dg::HVec psipZog2d  = dg::evaluate( psipZ_, g2d_);
        double psipRmax = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  0.,     thrust::maximum<double>()  );    
        //double psipRmin = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  psipRmax,thrust::minimum<double>()  );
        double psipZmax = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), 0.,      thrust::maximum<double>()  );    
        //double psipZmin = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), psipZmax,thrust::minimum<double>()  );   
        double deltapsi = fabs(psipZmax/g2d_.Ny()/g2d_.n() +psipRmax/g2d_.Nx()/g2d_.n());
        //deltaf_.setepsilon(deltapsi/4.);
        deltaf_.setepsilon(deltapsi); //macht weniger Zacken
    }
    /**
     * @brief Calculate the Flux Surface Average
     *
     * @param psip0 the actual psi value for q(psi)
     */
    double operator()(double psip0)
    {
        deltaf_.setpsi( psip0);
        container deltafog2d = dg::evaluate( deltaf_, g2d_);    
        double psipcut = dg::blas2::dot( f_,w2d_,deltafog2d); //int deltaf psip
        double vol     = dg::blas2::dot( oneongrid_ , w2d_,deltafog2d); //int deltaf
        double fsa = psipcut/vol;
        return fsa;
    }
    private:
    dg::Grid2d<double> g2d_;
    GeomParameters gp_;    
    container f_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipZ  psipZ_;
    DeltaFunction deltaf_;    
    const container w2d_;
    const container oneongrid_;
};
/**
 * @brief Class for the evaluation of the safety factor q
 * \f[ q(\psi_0) = \frac{1}{2\pi} \int dV |\nabla\psi_p| \delta(\psi_p-\psi_0) \alpha( R,Z) \f]
 * @tparam container 
 *
 */
template <class container = thrust::host_vector<double> >
struct SafetyFactor
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param gp  geometry parameters
     * @param f container for global safety factor
     */
    SafetyFactor(const dg::Grid2d<double>& g2d, GeomParameters gp,   const container& f) :
    g2d_(g2d),
    gp_(gp),
    f_(f), //why not directly use Alpha??
    psip_(gp),
    psipR_(gp),
    psipZ_(gp),
    deltaf_(DeltaFunction(gp,0.0,0.0)),
    w2d_ ( dg::create::weights( g2d_)),
    oneongrid_(dg::evaluate(dg::one,g2d_))              
    {
      dg::HVec psipRog2d  = dg::evaluate( psipR_, g2d_);
      dg::HVec psipZog2d  = dg::evaluate( psipZ_, g2d_);
      double psipRmax = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(), 0.,     thrust::maximum<double>()  );    
      //double psipRmin = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  psipRmax,thrust::minimum<double>()  );
      double psipZmax = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), 0.,      thrust::maximum<double>()  );    
      //double psipZmin = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), psipZmax,thrust::minimum<double>()  );   
      double deltapsi = fabs(psipZmax/g2d_.Ny() +psipRmax/g2d_.Nx());
      //deltaf_.setepsilon(deltapsi/4.);
      deltaf_.setepsilon(4.*deltapsi); //macht weniger Zacken
    }
    /**
     * @brief Calculate the q profile over the function f which has to be the global safety factor
     * \f[ q(\psi_0) = \frac{1}{2\pi} \int dV |\nabla\psi_p| \delta(\psi_p-\psi_0) \alpha( R,Z) \f]
     *
     * @param psip0 the actual psi value for q(psi)
     */
    double operator()(double psip0)
    {
        deltaf_.setpsi( psip0);
        container deltafog2d = dg::evaluate( deltaf_, g2d_);    
        double q = dg::blas2::dot( f_,w2d_,deltafog2d)/(2.*M_PI);
        return q;
    }
    private:
    dg::Grid2d<double> g2d_;
    GeomParameters gp_;    
    container f_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipZ  psipZ_;
    DeltaFunction deltaf_;    
    const container w2d_;
    const container oneongrid_;
};
/**
 * @brief Global safety factor
\f[ \alpha(R,Z) = \frac{|B^\varphi|}{R|B^\eta|} = \frac{I_{pol}(R,Z)}{R|\nabla\psi_p|} \f]
 */
struct Alpha
{
    Alpha( GeomParameters gp):
        psipR_(gp), 
        psipZ_(gp),
        ipol_(gp),
        R_0(gp.R_0){ }

    /**
    * @brief \f[ \frac{ I_{pol}(R,Z)}{R \sqrt{\nabla\psi_p}} \f]
    */
    double operator()( double R, double Z) const
    {
        return (1./R)*(ipol_(R,Z)/sqrt(psipR_(R,Z)*psipR_(R,Z) +psipZ_(R,Z)*psipZ_(R,Z))) ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return operator()(R,Z);
    }
    private:
    PsipR  psipR_;
    PsipZ  psipZ_;
    Ipol   ipol_;
    double R_0;  
};

struct FuncNeu
{
    FuncNeu( GeomParameters gp):psip_(gp){}
    double operator()(double R, double Z, double phi) const {return -psip_(R,Z)*cos(phi);
    }
    private:
    Psip psip_;
};
struct DeriNeu
{
    DeriNeu( GeomParameters gp):psip_(gp), bhat_(gp){}
    double operator()(double R, double Z, double phi) const {return psip_(R,Z)*bhat_(R,Z,phi)*sin(phi);
    }
    private:
    Psip psip_;
    BHatP bhat_;
};

struct FuncDirPer
{
    FuncDirPer( GeomParameters gp, double psi_0, double psi_1):
        R_0_(gp.R_0), psi0_(psi_0), psi1_(psi_1), psip_(gp), psipR_(gp), psipRR_(gp), psipZ_(gp), psipZZ_(gp){}
    double operator()(double R, double Z) const {
        double psip = psip_(R,Z);
        return (psip-psi0_)*(psip-psi1_)*sin(theta(R,Z));
    }
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    double dR( double R, double Z)const
    {
        double psip = psip_(R,Z), psipR = psipR_(R,Z), theta_ = theta(R,Z);
        return (2.*psip*psipR - (psi0_+psi1_)*psipR)*sin(theta_) 
            + (psip-psi0_)*(psip-psi1_)*cos(theta_)*thetaR(R,Z);
    }
    double dRR( double R, double Z)const
    {
        double psip = psip_(R,Z), psipR = psipR_(R,Z), theta_=theta(R,Z), thetaR_=thetaR(R,Z);
        double psipRR = psipRR_(R,Z);
        return (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR)*sin(theta_)
            + (2.*psip*psipR-(psi0_+psi1_)*psipR)*cos(theta_)*thetaR_
            + (2.*psip*psipR-(psi0_+psi1_)*psipR)*cos(theta_)*thetaR_
            + (psip-psi0_)*(psip-psi1_)*(thetaRR(R,Z)*cos(theta_)-sin(theta_)*thetaR_*thetaR_);
            
    }
    double dZ( double R, double Z)const
    {
        double psip = psip_(R,Z), psipZ = psipZ_(R,Z), theta_=theta(R,Z);
        return (2*psip*psipZ - (psi0_+psi1_)*psipZ)*sin(theta_) 
            + (psip-psi0_)*(psip-psi1_)*cos(theta_)*thetaZ(R,Z);
    }
    double dZZ( double R, double Z)const
    {
        double psip = psip_(R,Z), psipZ = psipZ_(R,Z), theta_=theta(R,Z), thetaZ_=thetaZ(R,Z);
        double psipZZ = psipZZ_(R,Z);
        return (2.*(psipZ*psipZ + psip*psipZZ) - (psi0_+psi1_)*psipZZ)*sin(theta_)
            + (2.*psip*psipZ-(psi0_+psi1_)*psipZ)*cos(theta_)*thetaZ_
            + (2.*psip*psipZ-(psi0_+psi1_)*psipZ)*cos(theta_)*thetaZ_
            + (psip-psi0_)*(psip-psi1_)*(-sin(theta_)*thetaZ_*thetaZ_ + cos(theta_)*thetaZZ(R,Z));
    }
    private:
    double theta( double R, double Z) const {
        double dR = R-R_0_;
        if( Z >= 0)
            return acos( dR/sqrt( dR*dR + Z*Z));
        else
            return 2.*M_PI-acos( dR/sqrt( dR*dR + Z*Z));
    }
    double thetaR( double R, double Z) const {
        double dR = R-R_0_;
        return -Z/(dR*dR+Z*Z);
    }
    double thetaZ( double R, double Z) const {
        double dR = R-R_0_;
        return dR/(dR*dR+Z*Z);
    }
    double thetaRR( double R, double Z) const {
        double dR = R-R_0_;
        return 2*Z*dR/(dR*dR+Z*Z)/(dR*dR+Z*Z);
    }
    double thetaZZ( double R, double Z) const { return -thetaRR(R,Z);}
    double R_0_;
    double psi0_, psi1_;
    Psip psip_;
    PsipR psipR_;
    PsipRR psipRR_;
    PsipZ psipZ_;
    PsipZZ psipZZ_;
};

//takes the magnetic field as chi
struct EllipticDirPerM
{
    EllipticDirPerM( GeomParameters gp, double psi_0, double psi_1): func_(gp, psi_0, psi_1), bmod_(gp), br_(gp), bz_(gp) {}
    double operator()(double R, double Z, double phi) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z);
        return -(br*func_.dR(R,Z) + bz*func_.dZ(R,Z) + bmod*( 1./R*func_.dR(R,Z) + func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    private:
    FuncDirPer func_;
    Bmodule bmod_;
    BR br_;
    BZ bz_;
};

struct FuncDirNeu
{
    FuncDirNeu( GeomParameters gp, double psi_0, double psi_1):
        psi0_(psi_0), psi1_(psi_1), psip_(gp), psipR_(gp), psipRR_(gp), psipZ_(gp), psipZZ_(gp){}
    double operator()(double R, double Z, double phi) const {
        double psip = psip_(R,Z);
        return (psip-psi0_)*(psip-psi1_);
    }
    double dR( double R, double Z)const
    {
        double psip = psip_(R,Z), psipR = psipR_(R,Z);
        return (2.*psip*psipR - (psi0_+psi1_)*psipR);
    }
    double dRR( double R, double Z)const
    {
        double psip = psip_(R,Z), psipR = psipR_(R,Z);
        double psipRR = psipRR_(R,Z);
        return (2.*(psipR*psipR + psip*psipRR) - (psi0_+psi1_)*psipRR);
            
    }
    double dZ( double R, double Z)const
    {
        double psip = psip_(R,Z), psipZ = psipZ_(R,Z);
        return (2*psip*psipZ - (psi0_+psi1_)*psipZ);
    }
    double dZZ( double R, double Z)const
    {
        double psip = psip_(R,Z), psipZ = psipZ_(R,Z);
        double psipZZ = psipZZ_(R,Z);
        return (2.*(psipZ*psipZ + psip*psipZZ) - (psi0_+psi1_)*psipZZ);
    }
    private:
    double psi0_, psi1_;
    Psip psip_;
    PsipR psipR_;
    PsipRR psipRR_;
    PsipZ psipZ_;
    PsipZZ psipZZ_;
};
//takes the magnetic field multiplied by (1+0.5sin(theta)) as chi
struct BmodTheta
{
    BmodTheta( GeomParameters gp): R_0_(gp.R_0), bmod_(gp){}
    double operator()(double R,double Z, double phi) const{
        return bmod_(R,Z)*(1.+0.5*sin(theta(R,Z)));
    }
    private:
    double theta( double R, double Z) const {
        double dR = R-R_0_;
        if( Z >= 0)
            return acos( dR/sqrt( dR*dR + Z*Z));
        else
            return 2.*M_PI-acos( dR/sqrt( dR*dR + Z*Z));
    }
    double R_0_;
    Bmodule bmod_;

};
struct EllipticDirNeuM
{
    EllipticDirNeuM( GeomParameters gp, double psi_0, double psi_1): R_0_(gp.R_0), func_(gp, psi_0, psi_1), bmod_(gp), br_(gp), bz_(gp) {}
    double operator()(double R, double Z, double phi) const {
        double bmod = bmod_(R,Z), br = br_(R,Z), bz = bz_(R,Z), theta_ = theta(R,Z);
        double chi = bmod*(1.+0.5*sin(theta_));
        double chiR = br*(1.+0.5*sin(theta_)) + bmod*0.5*cos(theta_)*thetaR(R,Z);
        double chiZ = bz*(1.+0.5*sin(theta_)) + bmod*0.5*cos(theta_)*thetaZ(R,Z);
        //double chi = bmod;
        //double chiR = br;
        //double chiZ = bz;
        return -(chiR*func_.dR(R,Z) + chiZ*func_.dZ(R,Z) + chi*( 1./R*func_.dR(R,Z) + func_.dRR(R,Z) + func_.dZZ(R,Z) ));

    }
    private:
    double theta( double R, double Z) const {
        double dR = R-R_0_;
        if( Z >= 0)
            return acos( dR/sqrt( dR*dR + Z*Z));
        else
            return 2.*M_PI-acos( dR/sqrt( dR*dR + Z*Z));
    }
    double thetaR( double R, double Z) const {
        double dR = R-R_0_;
        return -Z/(dR*dR+Z*Z);
    }
    double thetaZ( double R, double Z) const {
        double dR = R-R_0_;
        return dR/(dR*dR+Z*Z);
    }
    double R_0_;
    FuncDirNeu func_;
    Bmodule bmod_;
    BR br_;
    BZ bz_;
};


///@} 
} //namespace solovev
