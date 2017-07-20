#pragma once


/*!@file
 *
 * MagneticField objects 
 */
namespace dg
{
namespace geo
{
///@addtogroup magnetic
///@{

/**
 * @brief \f[   |B| = R_0\sqrt{I^2+(\nabla\psi)^2}/R   \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct Bmodule
{
    Bmodule( const MagneticField& c):  R_0_(c.R_0), c_(c)  { }
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
    MagneticField c_;
};

/**
 * @brief \f[  |B|^{-1} = R/R_0\sqrt{I^2+(\nabla\psi)^2}    \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct InvB
{
    InvB(  const MagneticField& c):  R_0_(c.R_0), c_(c)  { }
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
    MagneticField c_;
};

/**
 * @brief \f[   \ln{|B|}  \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct LnB
{
    LnB(  const MagneticField& c):  R_0_(c.R_0), c_(c)  { }
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
    MagneticField c_;
};

/**
 * @brief \f[  \frac{\partial |\hat{B}| }{ \partial \hat{R}}  \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct BR
{
    BR(const MagneticField& c):  R_0_(c.R_0), invB_(c, R0), c_(c) { }
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{R}} = 
      -\frac{1}{\hat B \hat R}   
      +  \frac{\hat I \left(\frac{\partial\hat I}{\partial\hat R} \right) 
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
      + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)\left( \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\right)}
      {\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f]
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
    InvB<MagneticField> invB_;
    MagneticField c_;
};

/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}}  \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct BZ
{

    BZ(const MagneticField& c ):  R_0_(c.R_0), c_(c), invB_(c, R0) { }
    /**
     * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}} = 
     \frac{ \hat I \left(\frac{\partial \hat I}{\partial\hat Z}    \right)+
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
    MagneticField c_;
    InvB<MagneticField> invB_; 
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct CurvatureNablaBR
{
    CurvatureNablaBR(const MagneticField& c): invB_(c), bZ_(c) { }
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
    InvB<MagneticField>   invB_;
    BZ<MagneticField> bZ_;    
};

/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B}  \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct CurvatureNablaBZ
{
    CurvatureNablaBZ( const MagneticField& c): invB_(c), bR_(c) { }
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
    InvB<MagneticField> invB_;
    BR<MagneticField> bR_;   
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}}=0 \f]
 */ 
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
 * @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct CurvatureKappaZ
{
    CurvatureKappaZ( const MagneticField c):
        invB_(c) { }
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
    InvB<MagneticField>   invB_;
};

/**
 * @brief \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  \f]
 * @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct DivCurvatureKappa
{
    DivCurvatureKappa( const MagneticField& c):
        invB_(c),
        bZ_(c){ }
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
    InvB<MagneticField>   invB_;
    BZ<MagneticField> bZ_;    
};

/**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} \f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct GradLnB
{
    GradLnB( const MagneticField& c): R_0_(c.R_0), c_(c), invB_(c), bR_(c), bZ_(c) { } 
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
    MagneticField c_;
    InvB<MagneticField>   invB_;
    BR<MagneticField> bR_;
    BZ<MagneticField> bZ_;   
};

/**
 * @brief \f[ B_\varphi = R_0I/R^2\f]
 @tparam MagneticField models aTokamakMagneticField
*/
template<class MagneticField>
struct FieldP
{
    FieldP( const MagneticField& c): R_0(c.R_0), c_(c){}
    double operator()( double R, double Z, double phi) const
    {
        return R_0*c_.ipol(R,Z)/R/R;
    }
    
    private:
    double R_0;
    MagneticField c_;
}; 

/**
 * @brief \f[ B_R = R_0\psi_Z /R\f]
 @tparam MagneticField models aTokamakMagneticField
 */
template<class MagneticField>
struct FieldR
{
    FieldR( const MagneticField& c): R_0(c.R_0), c_(c){}
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
    MagneticField c_;
   
};

/**
 * @brief \f[ B_Z = -R_0\psi_R /R\f]
 @tparam MagneticField models aTokamakMagneticField
 */
template<class MagneticField>
struct FieldZ
{
    FieldZ( const MagneticField& c): R_0(c.R_0), c_(c){}
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
    MagneticField c_;
   
};

/**
 * @brief \f[  B^{\theta} = B^R\partial_R\theta + B^Z\partial_Z\theta\f]
 @tparam MagneticField models aTokamakMagneticField
 */ 
template<class MagneticField>
struct FieldT
{
    FieldT( const MagneticField& c):  R_0_(c.R_0), fieldR_(c), fieldZ_(c){}
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
    FieldR<MagneticField> fieldR_;
    FieldZ<MagneticField> fieldZ_;

};

/**
 * @brief \f[ b_R = B_R/|B|\f]
 @tparam MagneticField models aTokamakMagneticField
 */
template<class MagneticField>
struct BHatR
{
    BHatR( const MagneticField& c): c_(c), R_0(c.R_0), invB_(c){ }
    double operator()( double R, double Z, double phi) const
    {
        return  invB_(R,Z)*R_0/R*c_.psipZ(R,Z);
    }
    private:
    MagneticField c_;
    double R_0;
    InvB<MagneticField>   invB_;

};

/**
 * @brief \f[ b_Z = B_Z/|B|\f]
 @tparam MagneticField models aTokamakMagneticField
 */
template<class MagneticField>
struct BHatZ
{
    BHatZ( const MagneticField& c): c_(c), R_0(c.R_0), invB_(c){ }

    double operator()( double R, double Z, double phi) const
    {
        return  -invB_(R,Z)*R_0/R*c_.psipR(R,Z);
    }
    private:
    MagneticField c_;
    double R_0;
    InvB<MagneticField>   invB_;

};

/**
 * @brief \f[ b_\varphi = B_\varphi/|B|\f]
 @tparam MagneticField models aTokamakMagneticField
 */
template<class MagneticField>
struct BHatP
{
    BHatP( const MagneticField& c): c_(c), R_0(c.R_0), invB_(c){ }
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z)*R_0*c_.ipol(R,Z)/R/R;
    }
    
    private:
    MagneticField c_;
    double R_0;
    InvB<MagneticField>   invB_;
  
}; 

///@} 

} //namespace geo
} //namespace dg

