#pragma once


#include "dg/backend/functions.h"
#include "dg/functors.h"


/*!@file
 *
 * Geometry objects 
 */
namespace dg
{
namespace fields
{
///@addtogroup geom
///@{

/**
 * @brief \f[   \hat{B}   \f]
 */ 
template<class Collective>
struct Bmodule
{
    Bmodule( const Collective& c, double R0 ):  R_0_(R0), c_(c)  { }
    /**
    * @brief \f[   \hat{B} \f]
    */ 
    double operator()(double R, double Z) const
    {    
        double psipR = c_.psipR(R,Z), psipZ = c_.psipZ(R,Z), ipol = c_.ipol(R,Z);
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
    Collective c_;
};

/**
 * @brief \f[   \frac{1}{\hat{B}}   \f]
 */ 
template<class Collective>
struct InvB
{
    InvB(  const Collective& c, double R0 ):  R_0_(R0), c_(c)  { }
    /**
    * @brief \f[   \frac{1}{\hat{B}} = 
        \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
        + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
    */ 
    double operator()(double R, double Z) const
    {    
        double psipR = c_.psipR(R,Z), psipZ = c_.psipZ(R,Z), ipol = c_.ipol(R,Z);
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
    Collective c_;
};

/**
 * @brief \f[   \ln{(   \hat{B})}  \f]
 */ 
template<class Collective>
struct LnB
{
    LnB(  const Collective& c, double R0 ):  R_0_(R0), c_(c)  { }
/**
 * @brief \f[   \ln{(   \hat{B})} = \ln{\left[
      \frac{\hat{R}_0}{\hat{R}} \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2} \right] } \f]
 */ 
    double operator()(double R, double Z) const
    {    
        double psipR = c_.psipR(R,Z), psipZ = c_.psipZ(R,Z), ipol = c_.ipol(R,Z);
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
    Collective c_;
};
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{R}}  \f]
 */ 

template<class Collective>
struct BR
{
    BR(const Collective& c, double R0):  R_0_(R0), invB_(c, R0), c_(c) { }
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
        return -1./R/invB_(R,Z) + invB_(R,Z)/Rn/Rn*(c_.ipol(R,Z)*c_.ipolR(R,Z) + c_.psipR(R,Z)*c_.psipRR(R,Z) + c_.psipZ(R,Z)*c_.psipRZ(R,Z));
    }
      /**
       * @brief == operator()(R,Z)
       */ 
    double operator()(double R, double Z, double phi)const{return operator()(R,Z);}
  private:
    double R_0_;
    InvB<Collective> invB_;
    Collective c_;
};
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}}  \f]
 */ 
template<class Collective>
struct BZ
{

    BZ(const Collective& c, double R0):  R_0_(R0), c_(c), invB_(c, R0) { }
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
        return (invB_(R,Z)/Rn/Rn)*(c_.ipol(R,Z)*c_.ipolZ(R,Z) + c_.psipR(R,Z)*c_.psipRZ(R,Z) + c_.psipZ(R,Z)*c_.psipZZ(R,Z));
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi)const{return operator()(R,Z);}
  private:
    double R_0_;
    Collective c_;
    InvB<Collective> invB_; 
};
/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} \f]
 */ 
template<class Collective>
struct CurvatureNablaBR
{
    CurvatureNablaBR(const Collective& c, double R0 ): invB_(c, R0), bZ_(c, R0) { }
    /**
     * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} =-\frac{1}{ \hat{B}^2}  \frac{\partial \hat{B}}{\partial \hat{Z}}  \f]
     */ 
    double operator()( double R, double Z) const
    {
        return -invB_(R,Z)*invB_(R,Z)*bZ_(R,Z); 
    }
    
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return -invB_(R,Z,phi)*invB_(R,Z,phi)*bZ_(R,Z,phi); 
    }
    private:    
    InvB<Collective>   invB_;
    BZ<Collective> bZ_;    
};

/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B}  \f]
 */ 
template<class Collective>
struct CurvatureNablaBZ
{
    CurvatureNablaBZ( const Collective& c, double R0): invB_(c, R0), bR_(c, R0) { }
 /**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B} =\frac{1}{ \hat{B}^2}   \frac{\partial \hat{B}}{\partial \hat{R}} \f]
 */    
    double operator()( double R, double Z) const
    {
        return invB_(R,Z)*invB_(R,Z)*bR_(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z,phi)*invB_(R,Z,phi)*bR_(R,Z,phi);
    }
    private:    
    InvB<Collective> invB_;
    BR<Collective> bR_;   
};
    /**
     * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}} \f]
     */ 
template<class Collective>
struct CurvatureKappaR
{
    /**
     * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}} =0  \f]
     */ 
    double operator()( double R, double Z) const
    {
        return  0.;
    }
    
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return 0.;
    }
};
/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\vec{\kappa}}  \f]
 */ 
template<class Collective>
struct CurvatureKappaZ
{
    CurvatureKappaZ( const Collective c, double R0):
        invB_(c, R0) { }
 /**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\vec{\kappa}} = - \frac{1}{\hat{R} \hat{B}} \f]
 */    
    double operator()( double R, double Z) const
    {
        return -invB_(R,Z)/R;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return -invB_(R,Z,phi)/R;
    }
    private:    
    InvB<Collective>   invB_;
};
/**
 * @brief \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  \f]
 */ 
template<class Collective>
struct DivCurvatureKappa
{
    DivCurvatureKappa( const Collective& c, double R0):
        invB_(c, R0),
        bZ_(c, R0){ }
 /**
 * @brief \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  = \frac{1}{\hat{R}  \hat{B}^2 } \partial_{\hat{Z}} \hat{B}\f]
 */    
    double operator()( double R, double Z) const
    {
        return bZ_(R,Z)*invB_(R,Z)*invB_(R,Z)/R;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return  bZ_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)/R;
    }
    private:    
    InvB<Collective>   invB_;
    BZ<Collective> bZ_;    
};
/**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} \f]
 */ 
template<class Collective>
struct GradLnB
{
    GradLnB( const Collective& c, double R0): R_0_(R0), c_(c), invB_(c, R0), bR_(c, R0), bZ_(c, R0) { } 
    /**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} = \frac{1}{\hat{R}\hat{B}^2 } \left[ \hat{B}, \hat{\psi}_p\right]_{\hat{R}\hat{Z}} \f]
 */ 
    double operator()( double R, double Z) const
    {
        double invB = invB_(R,Z);
        return R_0_*invB*invB*(bR_(R,Z)*c_.psipZ(R,Z)-bZ_(R,Z)*c_.psipR(R,Z))/R ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi)const{return operator()(R,Z);}
    private:
    double R_0_;
    Collective c_;
    InvB<Collective>   invB_;
    BR<Collective> bR_;
    BZ<Collective> bZ_;   
};
/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
template<class Collective>
struct Field
{
    Field( const Collective& c, double R0):c_(c), R_0_(R0), invB_(c, R0) { }
    /**
     * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
     \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
     \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I}  \hat{R}_0}  \f]
     */ 
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double ipol = c_.ipol(y[0],y[1]);
        yp[2] =  y[0]*y[0]/invB_(y[0],y[1])/ipol/R_0_;       //ds/dphi =  R^2 B/I/R_0_hat
        yp[0] =  y[0]*c_.psipZ(y[0],y[1])/ipol;              //dR/dphi =  R/I Psip_Z
        yp[1] = -y[0]*c_.psipR(y[0],y[1])/ipol ;             //dZ/dphi = -R/I Psip_R

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
    double error( const dg::HVec& x0, const dg::HVec& x1)
    {
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end){ 
        if ( isnan(end[0]) || isnan(end[1]) || isnan(end[2]) ) 
        {
            return false;
        }
        //if new integrated point outside domain
        if ((1e-5 > end[0]  ) || (1e10 < end[0])  ||(-1e10  > end[1]  ) || (1e10 < end[1])||(-1e10 > end[2]  ) || (1e10 < end[2])  )
        {
            return false;
        }
        return true;
    }
    
    private:
    Collective c_;
    double R_0_;
    InvB<Collective>   invB_;
   
};


//////////////////////////////////////////////////////////////////////////////
/**
 * @brief Phi component of magnetic field \f$ B_\Phi\f$
*/
template<class Collective>
struct FieldP
{
    FieldP( const Collective& c, double R0): R_0(R0), c_(c){}
    double operator()( double R, double Z, double phi) const
    {
        return R_0*c_.ipol(R,Z)/R/R;
    }
    
    private:
    double R_0;
    Collective c_;
}; 

/**
 * @brief R component of magnetic field\f$ B_R\f$
 */
template<class Collective>
struct FieldR
{
    FieldR( const Collective& c, double R0): R_0(R0), c_(c){}
    double operator()( double R, double Z) const
    {
        return  R_0/R*c_.psipZ(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return  this->operator()(R,Z);
    }
    private:
    double R_0;
    Collective c_;
   
};
/**
 * @brief Z component of magnetic field \f$ B_Z\f$
 */
template<class Collective>
struct FieldZ
{
    FieldZ( const Collective& c, double R0): R_0(R0), c_(c){}
    double operator()( double R, double Z) const
    {
        return  -R_0/R*c_.psipR(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return  this->operator()(R,Z);
    }
    private:
    double R_0;
    Collective c_;
   
};

/**
 * @brief \f$\theta\f$ component of magnetic field 
 */ 
template<class Collective>
struct FieldT
{
    FieldT( const Collective& c, double R0):  R_0_(R0), fieldR_(c, R0), fieldZ_(c, R0){}
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
    FieldR<Collective> fieldR_;
    FieldZ<Collective> fieldZ_;

};

//////////////////////////////////////////////////////////////////////////////
/**
 * @brief R component of magnetic field unit vector \f$ b_R\f$
 */
template<class Collective>
struct BHatR
{
    BHatR( const Collective& c, double R0): c_(c), R_0(R0), invB_(c, R0){ }
    double operator()( double R, double Z, double phi) const
    {
        return  invB_(R,Z)*R_0/R*c_.psipZ(R,Z);
    }
    private:
    Collective c_;
    double R_0;
    InvB<Collective>   invB_;

};
/**
 * @brief Z component of magnetic field unit vector \f$ b_Z\f$
 */
template<class Collective>
struct BHatZ
{
    BHatZ( const Collective& c, double R0): c_(c), R_0(R0), invB_(c, R0){ }

    double operator()( double R, double Z, double phi) const
    {
        return  -invB_(R,Z)*R_0/R*c_.psipR(R,Z);
    }
    private:
    Collective c_;
    double R_0;
    InvB<Collective>   invB_;

};
/**
 * @brief Phi component of magnetic field unit vector \f$ b_\Phi\f$
 */
template<class Collective>
struct BHatP
{
    BHatP( const Collective& c, double R0): c_(c), R_0(R0), invB_(c, R0){ }
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z)*R_0*c_.ipol(R,Z)/R/R;
    }
    
    private:
    Collective c_;
    double R_0;
    InvB<Collective>   invB_;
  
}; 
////////////////////////////////////////////for grid generation/////////////////
namespace flux{

/**
 * @brief 
 * \f[  d R/d \theta =   B^R/B^\theta \f], 
 * \f[  d Z/d \theta =   B^Z/B^\theta \f],
 * \f[  d y/d \theta =   B^y/B^\theta\f]
 */ 
template< class PsiR, class PsiZ, class Ipol>
struct FieldRZYT
{
    FieldRZYT( PsiR psiR, PsiZ psiZ, Ipol ipol, double R0, double Z0): R_0_(R0), psipR_(psiR), psipZ_(psiZ),ipol_(ipol){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double ipol=ipol_(y[0], y[1]);
        yp[0] =  psipZ;//fieldR
        yp[1] = -psipR;//fieldZ
        yp[2] =ipol/y[0];
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + (y[1]-Z_0_)*(y[1]-Z_0_);
        double fieldT = psipZ*(y[1]-Z_0_)/r2 + psipR*(y[0]-R_0_)/r2; 
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_, Z_0_;
    PsiR psipR_;
    PsiZ psipZ_;
    Ipol ipol_;
};

template< class PsiR, class PsiZ, class Ipol>
struct FieldRZYZ
{
    FieldRZYZ( PsiR psiR, PsiZ psiZ, Ipol ipol): psipR_(psiR), psipZ_(psiZ), ipol_(ipol){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double ipol=ipol_(y[0], y[1]);
        yp[0] =  psipZ;//fieldR
        yp[1] = -psipR;//fieldZ
        yp[2] =   ipol/y[0]; //fieldYbar
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    PsiR psipR_;
    PsiZ psipZ_;
    Ipol ipol_;
};
/**
 * @brief 
 * \f[  d R/d y =   B^R/B^y \f], 
 * \f[  d Z/d y =   B^Z/B^y \f],
 */ 
template< class PsiR, class PsiZ, class Ipol>
struct FieldRZY
{
    FieldRZY( PsiR psiR, PsiZ psiZ, Ipol ipol): f_(1.), psipR_(psiR), psipZ_(psiZ),ipol_(ipol){}
    void set_f(double f){ f_ = f;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double ipol=ipol_(y[0], y[1]);
        double fnorm = y[0]/ipol/f_;       
        yp[0] =  (psipZ)*fnorm;
        yp[1] = -(psipR)*fnorm;
    }
  private:
    double f_;
    PsiR psipR_;
    PsiZ psipZ_;
    Ipol ipol_;
};
/**
 * @brief 
 * \f[  d R/d y   =   B^R/B^y = \frac{q R}{I} \frac{\partial \psi_p}{\partial Z} \f], 
 * \f[  d Z/d y   =   B^Z/B^y =    -\frac{q R}{I} \frac{\partial \psi_p}{\partial R} \f],
 * \f[  d y_R/d y =  \frac{q( \psi_p) R}{I( \psi_p)}\left[\frac{\partial^2 \psi_p}{\partial R \partial Z} y_R
    -\frac{\partial^2 \psi_p}{\partial^2 R} y_Z)\right] + 
    \frac{\partial \psi_p}{\partial R} \left(\frac{1}{I(\psi_p)} \frac{\partial I(\psi_p)}{\partial \psi_p} -\frac{1}{q(\psi_p)} \frac{\partial q(\psi_p)}{\partial \psi_p}\right)-\frac{1}{R} \f], 
 * \f[  d y_Z/d y =   - \frac{q( \psi_p) R}{I( \psi_p)}\left[\frac{\partial^2 \psi_p}{\partial Z^2} y_R\right)
    -\frac{\partial^2 \psi_p}{\partial R \partial Z} y_Z\right]+ 
    \frac{\partial \psi_p}{\partial Z} \left(\frac{1}{I(\psi_p)} \frac{\partial I(\psi_p)}{\partial \psi_p} -\frac{1}{q(\psi_p)} \frac{\partial q(\psi_p)}{\partial \psi_p}\right)\f],
 */ 
template<class PsiR, class PsiZ, class PsiRR, class PsiRZ, class PsiZZ, class Ipol, class IpolR, class IpolZ>
struct FieldRZYRYZY
{
    FieldRZYRYZY( PsiR psiR, PsiZ psiZ, PsiRR psiRR, PsiRZ psiRZ, PsiZZ psiZZ, Ipol ipol, IpolR ipolR, IpolZ ipolZ): 
        psipR_(psiR), psipZ_(psiZ), psipRR_(psiRR), psipRZ_(psiRZ), psipZZ_(psiZZ), ipol_(ipol), ipolR_(ipolR), ipolZ_(ipolZ){ f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void initialize( double R0, double Z0, double& yR, double& yZ)
    {
        double psipR = psipR_(R0, Z0), psipZ = psipZ_(R0,Z0);
        double psip2 = (psipR*psipR+ psipZ*psipZ);
        double fnorm =R0/ipol_(R0,Z0)/f_; //=Rq/I
        yR = -psipZ_(R0, Z0)/psip2/fnorm;
        yZ = +psipR_(R0, Z0)/psip2/fnorm;
    }
    void derive( double R0, double Z0, double& xR, double& xZ)
    {
        xR = +f_*psipR_(R0, Z0);
        xZ = +f_*psipZ_(R0, Z0);
    }
    
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psipRR = psipRR_(y[0], y[1]), psipRZ = psipRZ_(y[0],y[1]), psipZZ = psipZZ_(y[0],y[1]);
        double ipol=ipol_(y[0], y[1]);
        double ipolR=ipolR_(y[0], y[1]);
        double ipolZ=ipolZ_(y[0], y[1]);
        double fnorm =y[0]/ipol/f_; //=R/(I/q)

        yp[0] = -(psipZ)*fnorm;
        yp[1] = +(psipR)*fnorm;
        yp[2] = (+psipRZ*y[2]- psipRR*y[3])*fnorm + f_prime_/f_*psipR + ipolR/ipol - 1./y[0];
        yp[3] = (-psipRZ*y[3]+ psipZZ*y[2])*fnorm + f_prime_/f_*psipZ + ipolZ/ipol;

    }
  private:
    double f_, f_prime_;
    PsiR psipR_;
    PsiZ psipZ_;
    PsiRR psipRR_;
    PsiRZ psipRZ_;
    PsiZZ psipZZ_;
    Ipol ipol_;
    IpolR ipolR_;
    IpolZ ipolZ_;
};

}//namespace flux
namespace ribeiro{

template< class PsiR, class PsiZ>
struct FieldRZYT
{
    FieldRZYT( PsiR psiR, PsiZ psiZ, double R0, double Z0): R_0_(R0), Z_0_(Z0), psipR_(psiR), psipZ_(psiZ){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psip2 = psipR*psipR+psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] = +psipR;//fieldZ
        //yp[2] = 1; //volume
        //yp[2] = sqrt(psip2); //equalarc
        yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + (y[1]-Z_0_)*(y[1]-Z_0_);
        double fieldT = psipZ*(y[1]-Z_0_)/r2 + psipR*(y[0]-R_0_)/r2; 
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_, Z_0_;
    PsiR psipR_;
    PsiZ psipZ_;
};

template< class PsiR, class PsiZ>
struct FieldRZYZ
{
    FieldRZYZ( PsiR psiR, PsiZ psiZ): psipR_(psiR), psipZ_(psiZ){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psip2 = psipR*psipR+psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] =  psipR;//fieldZ
        //yp[2] = 1.0; //volume
        //yp[2] = sqrt(psip2); //equalarc
        yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    PsiR psipR_;
    PsiZ psipZ_;
};

template <class PsiR, class PsiZ>
struct FieldRZY
{
    FieldRZY( PsiR psiR, PsiZ psiZ): f_(1.), psipR_(psiR), psipZ_(psiZ){}
    void set_f(double f){ f_ = f;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psip2 = psipR*psipR+psipZ*psipZ;
        //yp[0] = +psipZ/f_;//volume 
        //yp[1] = -psipR/f_;//volume 
        //yp[0] = +psipZ/sqrt(psip2)/f_;//equalarc
        //yp[1] = -psipR/sqrt(psip2)/f_;//equalarc
        yp[0] = -psipZ/psip2/f_;//ribeiro
        yp[1] = +psipR/psip2/f_;//ribeiro
        //yp[0] = +psipZ/psip2/sqrt(psip2)/f_;//separatrix
        //yp[1] = -psipR/psip2/sqrt(psip2)/f_;//separatrix
    }
  private:
    double f_;
    PsiR psipR_;
    PsiZ psipZ_;
};


template<class PsiR, class PsiZ, class PsiRR, class PsiRZ, class PsiZZ>
struct FieldRZYRYZY
{
    FieldRZYRYZY( PsiR psiR, PsiZ psiZ, PsiRR psiRR, PsiRZ psiRZ, PsiZZ psiZZ): 
        psipR_(psiR), psipZ_(psiZ), psipRR_(psiRR), psipRZ_(psiRZ), psipZZ_(psiZZ){ f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void initialize( double R0, double Z0, double& yR, double& yZ)
    {
        yR = -f_*psipZ_(R0, Z0);
        yZ = +f_*psipR_(R0, Z0);
    }
    void derive( double R0, double Z0, double& xR, double& xZ)
    {
        xR = +f_*psipR_(R0, Z0);
        xZ = +f_*psipZ_(R0, Z0);
    }

    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psipRR = psipRR_(y[0], y[1]), psipRZ = psipRZ_(y[0],y[1]), psipZZ = psipZZ_(y[0],y[1]);
        double psip2 = (psipR*psipR+ psipZ*psipZ);

        yp[0] =  -psipZ/f_/psip2;
        yp[1] =  +psipR/f_/psip2;
        yp[2] =  ( + psipRZ*y[2] - psipRR*y[3] )/f_/psip2 
            + f_prime_/f_* psipR + 2.*(psipR*psipRR + psipZ*psipRZ)/psip2 ;
        yp[3] =  (-psipRZ*y[3] + psipZZ*y[2])/f_/psip2 
            + f_prime_/f_* psipZ + 2.*(psipR*psipRZ + psipZ*psipZZ)/psip2;
    }
  private:
    double f_, f_prime_;
    PsiR psipR_;
    PsiZ psipZ_;
    PsiRR psipRR_;
    PsiRZ psipRZ_;
    PsiZZ psipZZ_;
};
}//namespace ribeiro
namespace equalarc{


template< class PsiR, class PsiZ>
struct FieldRZYT
{
    FieldRZYT( PsiR psiR, PsiZ psiZ, double R0, double Z0): R_0_(R0), Z_0_(Z0), psipR_(psiR), psipZ_(psiZ){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psip2 = psipR*psipR+psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] = +psipR;//fieldZ
        //yp[2] = 1; //volume
        yp[2] = sqrt(psip2); //equalarc
        //yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + (y[1]-Z_0_)*(y[1]-Z_0_);
        double fieldT = psipZ*(y[1]-Z_0_)/r2 + psipR*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_, Z_0_;
    PsiR psipR_;
    PsiZ psipZ_;
};

template< class PsiR, class PsiZ>
struct FieldRZYZ
{
    FieldRZYZ( PsiR psiR, PsiZ psiZ): psipR_(psiR), psipZ_(psiZ){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psip2 = psipR*psipR+psipZ*psipZ;
        yp[0] = -psipZ;//fieldR
        yp[1] = +psipR;//fieldZ
        //yp[2] = 1.0; //volume
        yp[2] = sqrt(psip2); //equalarc
        //yp[2] = psip2; //ribeiro
        //yp[2] = psip2*sqrt(psip2); //separatrix
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    PsiR psipR_;
    PsiZ psipZ_;
};

template <class PsiR, class PsiZ>
struct FieldRZY
{
    FieldRZY( PsiR psiR, PsiZ psiZ): f_(1.), psipR_(psiR), psipZ_(psiZ){}
    void set_f(double f){ f_ = f;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psip2 = psipR*psipR+psipZ*psipZ;
        //yp[0] = +psipZ/f_;//volume 
        //yp[1] = -psipR/f_;//volume 
        yp[0] = -psipZ/sqrt(psip2)/f_;//equalarc
        yp[1] = +psipR/sqrt(psip2)/f_;//equalarc
        //yp[0] = -psipZ/psip2/f_;//ribeiro
        //yp[1] = +psipR/psip2/f_;//ribeiro
        //yp[0] = +psipZ/psip2/sqrt(psip2)/f_;//separatrix
        //yp[1] = -psipR/psip2/sqrt(psip2)/f_;//separatrix
    }
  private:
    double f_;
    PsiR psipR_;
    PsiZ psipZ_;
};


template<class PsiR, class PsiZ, class PsiRR, class PsiRZ, class PsiZZ>
struct FieldRZYRYZY
{
    FieldRZYRYZY( PsiR psiR, PsiZ psiZ, PsiRR psiRR, PsiRZ psiRZ, PsiZZ psiZZ): 
        psipR_(psiR), psipZ_(psiZ), psipRR_(psiRR), psipRZ_(psiRZ), psipZZ_(psiZZ){ f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void initialize( double R0, double Z0, double& yR, double& yZ)
    {
        double psipR = psipR_(R0, Z0), psipZ = psipZ_(R0,Z0);
        double psip2 = (psipR*psipR+ psipZ*psipZ);
        yR = -f_*psipZ_(R0, Z0)/sqrt(psip2);
        yZ = +f_*psipR_(R0, Z0)/sqrt(psip2);
    }
    void derive( double R0, double Z0, double& xR, double& xZ)
    {
        xR = +f_*psipR_(R0, Z0);
        xZ = +f_*psipZ_(R0, Z0);
    }

    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psipRR = psipRR_(y[0], y[1]), psipRZ = psipRZ_(y[0],y[1]), psipZZ = psipZZ_(y[0],y[1]);
        double psip2 = (psipR*psipR+ psipZ*psipZ);

        yp[0] =  -psipZ/f_/sqrt(psip2);
        yp[1] =  +psipR/f_/sqrt(psip2);
        yp[2] =  ( +psipRZ*y[2] - psipRR *y[3])/f_/sqrt(psip2) 
            + f_prime_/f_* psipR + (psipR*psipRR + psipZ*psipRZ)/psip2 ;
        yp[3] =  ( -psipRZ*y[3] + psipZZ*y[2])/f_/sqrt(psip2)
            + f_prime_/f_* psipZ + (psipR*psipRZ + psipZ*psipZZ)/psip2;
    }
  private:
    double f_, f_prime_;
    PsiR psipR_;
    PsiZ psipZ_;
    PsiRR psipRR_;
    PsiRZ psipRZ_;
    PsiZZ psipZZ_;
};

}//namespace equalarc

template < class PsiR, class PsiZ>
struct FieldRZtau
{
    FieldRZtau( PsiR psiR, PsiZ psiZ): psipR_(psiR), psipZ_(psiZ){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psi2 = psipR*psipR+ psipZ*psipZ;
        yp[0] =  psipR/psi2;
        yp[1] =  psipZ/psi2;
    }
  private:
    PsiR psipR_;
    PsiZ psipZ_;
};

template<class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
struct HessianRZtau
{
    HessianRZtau( PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY): norm_(false), quad_(1), psipR_(psiX), psipZ_(psiY), psipRR_(psiXX), psipRZ_(psiXY), psipZZ_(psiYY){}
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
    PsiX  psipR_;
    PsiY  psipZ_;
    PsiXX psipRR_;
    PsiXY psipRZ_;
    PsiYY psipZZ_;
};

template<class Psi, class PsiX, class PsiY>
struct MinimalCurve
{
    MinimalCurve(Psi psi, PsiX psiX, PsiY psiY): norm_(false), 
        psip_(psi), psipR_(psiX), psipZ_(psiY){}
    void set_norm( bool normed) {norm_ = normed;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0], y[1]);
        yp[0] = y[2];
        yp[1] = y[3];
        //double psipRZ = psipRZ_(y[0], y[1]), psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0], y[1]), psipRR=psipRR_(y[0], y[1]), psipZZ=psipZZ_(y[0], y[1]); 
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
    Psi  psip_;
    PsiX psipR_;
    PsiY psipZ_;
};

///@} 
} //namespace fields
} //namespace dg

