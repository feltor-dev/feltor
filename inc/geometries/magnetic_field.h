#pragma once

#include "fluxfunctions.h"

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
* @brief container class of R, psi and Ipol 

 This is the representation of toroidally axisymmetric magnetic fields that can be modeled in the form
 \f[
 \vec B = \frac{R_0}{R} \left( I(\psi_p) \hat e_\varphi + \nabla \psi_p \times \hat e_\varphi\right)
 \f]
 where \f$ R_0\f$ is a normalization constant, \f$ I\f$ the poloidal current 
 and \f$ \psi_p\f$ the poloidal flux function.
 @note We implicitly also assume the toroidal field line approximation, i.e. all curvature
 and other perpendicular terms created with this field will assume that the perpendicular 
 direction lies within the
 R-Z planes of a cylindrical grid (The plane \f$ \perp \hat e_\varphi\f$ )
*/
struct TokamakMagneticField
{
    ///as long as the field stays empty the access functions are undefined
    TokamakMagneticField(){}
    TokamakMagneticField( double R0, const BinaryFunctorsLvl2& psip, const BinaryFunctorsLvl1& ipol): R0_(R0), psip_(psip), ipol_(ipol){}
    void set( double R0, const BinaryFunctorsLvl2& psip, const BinaryFunctorsLvl1& ipol)
    {
        R0_=R0;
        psip_=psip; 
        ipol_=ipol;
    }
    /// \f$ R_0 \f$ 
    double R0()const {return R0_;}
    /// \f$ \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const aBinaryFunctor& psip()const{return psip_.f();}
    /// \f$ \partial_R \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const aBinaryFunctor& psipR()const{return psip_.dfx();}
    /// \f$ \partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const aBinaryFunctor& psipZ()const{return psip_.dfy();}
    /// \f$ \partial_R\partial_R \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const aBinaryFunctor& psipRR()const{return psip_.dfxx();}
    /// \f$ \partial_R\partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const aBinaryFunctor& psipRZ()const{return psip_.dfxy();}
    /// \f$ \partial_Z\partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    const aBinaryFunctor& psipZZ()const{return psip_.dfyy();}
    /// \f$ I(\psi_p) \f$ the current
    const aBinaryFunctor& ipol()const{return ipol_.f();}
    /// \f$ \partial_R I(\psi_p) \f$ 
    const aBinaryFunctor& ipolR()const{return ipol_.dfx();}
    /// \f$ \partial_Z I(\psi_p) \f$ 
    const aBinaryFunctor& ipolZ()const{return ipol_.dfy();}

    const BinaryFunctorsLvl2& get_psip() const{return psip_;}
    const BinaryFunctorsLvl1& get_ipol() const{return ipol_;}

    private:
    double R0_;
    BinaryFunctorsLvl2 psip_;
    BinaryFunctorsLvl1 ipol_;
};


/**
 * @brief \f[   |B| = R_0\sqrt{I^2+(\nabla\psi)^2}/R   \f]
 */ 
struct Bmodule : public aCloneableBinaryFunctor<Bmodule>
{
    Bmodule( const TokamakMagneticField& mag): mag_(mag)  { }
  private:
    double do_compute(double R, double Z) const
    {    
        double psipR = mag_.psipR()(R,Z), psipZ = mag_.psipZ()(R,Z), ipol = mag_.ipol()(R,Z);
        return mag_.R0()/R*sqrt(ipol*ipol+psipR*psipR +psipZ*psipZ);
    }
    TokamakMagneticField mag_;
};

/**
 * @brief \f[  |B|^{-1} = R/R_0\sqrt{I^2+(\nabla\psi)^2}    \f]

    \f[   \frac{1}{\hat{B}} = 
        \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
        + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
 */ 
struct InvB : public aCloneableBinaryFunctor<InvB>
{
    InvB(  const TokamakMagneticField& mag): mag_(mag){ }
  private:
    double do_compute(double R, double Z) const
    {    
        double psipR = mag_.psipR()(R,Z), psipZ = mag_.psipZ()(R,Z), ipol = mag_.ipol()(R,Z);
        return R/(mag_.R0()*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
    TokamakMagneticField mag_;
};

/**
 * @brief \f[   \ln{|B|}  \f]
 *
   \f[   \ln{(   \hat{B})} = \ln{\left[
          \frac{\hat{R}_0}{\hat{R}} \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
          + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2} \right] } \f]
 */ 
struct LnB : public aCloneableBinaryFunctor<LnB>
{
    LnB(const TokamakMagneticField& mag): mag_(mag) { }
  private:
    double do_compute(double R, double Z) const
    {    
        double psipR = mag_.psipR()(R,Z), psipZ = mag_.psipZ()(R,Z), ipol = mag_.ipol()(R,Z);
        return log(mag_.R0()/R*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
    TokamakMagneticField mag_;
};

/**
 * @brief \f[  \frac{\partial |\hat{B}| }{ \partial \hat{R}}  \f]
 * 
 \f[  \frac{\partial \hat{B} }{ \partial \hat{R}} = 
      -\frac{1}{\hat B \hat R}   
      +  \frac{\hat I \left(\frac{\partial\hat I}{\partial\hat R} \right) 
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
      + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)\left( \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\right)}
      {\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f]
 */ 
struct BR: public aCloneableBinaryFunctor<BR>
{
    BR(const TokamakMagneticField& mag): invB_(mag), mag_(mag) { }
  private:
    double do_compute(double R, double Z) const
    { 
        double Rn;
        Rn = R/mag_.R0();
        //sign before A changed to +
        //return -( Rn*Rn/invB_(R,Z)/invB_(R,Z)+ qampl_*qampl_*Rn *A_*psipR_(R,Z) - R  *(psipZ_(R,Z)*psipRZ_(R,Z)+psipR_(R,Z)*psipRR_(R,Z)))/(R*Rn*Rn/invB_(R,Z));
        return -1./R/invB_(R,Z) + invB_(R,Z)/Rn/Rn*(mag_.ipol()(R,Z)*mag_.ipolR()(R,Z) + mag_.psipR()(R,Z)*mag_.psipRR()(R,Z) + mag_.psipZ()(R,Z)*mag_.psipRZ()(R,Z));
    }
    InvB invB_;
    TokamakMagneticField mag_;
};

/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}}  \f]
 *
  \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}} = 
     \frac{ \hat I \left(\frac{\partial \hat I}{\partial\hat Z}    \right)+
     \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
          + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2} \right)}{\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f]
 */ 
struct BZ: public aCloneableBinaryFunctor<BZ>
{
    BZ(const TokamakMagneticField& mag ): mag_(mag), invB_(mag) { }
  private:
    double do_compute(double R, double Z) const
    { 
        double Rn;
        Rn = R/mag_.R0();
        //sign before A changed to -
        //return (-qampl_*qampl_*A_/R_0_*psipZ_(R,Z) + psipR_(R,Z)*psipRZ_(R,Z)+psipZ_(R,Z)*psipZZ_(R,Z))/(Rn*Rn/invB_(R,Z));
        return (invB_(R,Z)/Rn/Rn)*(mag_.ipol()(R,Z)*mag_.ipolZ()(R,Z) + mag_.psipR()(R,Z)*mag_.psipRZ()(R,Z) + mag_.psipZ()(R,Z)*mag_.psipZZ()(R,Z));
    }
    TokamakMagneticField mag_;
    InvB invB_; 
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} \f]
 *
    \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} =-\frac{1}{ \hat{B}^2}  \frac{\partial \hat{B}}{\partial \hat{Z}}  \f]
 */ 
struct CurvatureNablaBR: public aCloneableBinaryFunctor<CurvatureNablaBR>
{
    CurvatureNablaBR(const TokamakMagneticField& mag): invB_(mag), bZ_(mag) { }
    private:    
    double do_compute( double R, double Z) const
    {
        return -invB_(R,Z)*invB_(R,Z)*bZ_(R,Z); 
    }
    InvB invB_;
    BZ bZ_;    
};

/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B}  \f]
 *
   \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B} =\frac{1}{ \hat{B}^2}   \frac{\partial \hat{B}}{\partial \hat{R}} \f]
 */ 
struct CurvatureNablaBZ: public aCloneableBinaryFunctor<CurvatureNablaBZ>
{
    CurvatureNablaBZ( const TokamakMagneticField& mag): invB_(mag), bR_(mag) { }
    private:    
    double do_compute( double R, double Z) const
    {
        return invB_(R,Z)*invB_(R,Z)*bR_(R,Z);
    }
    InvB invB_;
    BR bR_;   
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}}=0 \f]
 *
 \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}} =0  \f]
 */ 
struct CurvatureKappaR: public aCloneableBinaryFunctor<CurvatureKappaR>
{
    CurvatureKappaR( ){ }
    CurvatureKappaR( const TokamakMagneticField& mag){ }
    private:
    double do_compute( double R, double Z) const
    {
        return  0.;
    }
};

/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\vec{\kappa}}  \f]
 *
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\vec{\kappa}} = - \frac{1}{\hat{R} \hat{B}} \f]
 */ 
struct CurvatureKappaZ: public aCloneableBinaryFunctor<CurvatureKappaZ>
{
    CurvatureKappaZ( const TokamakMagneticField& mag): invB_(mag) { }
    private:    
    double do_compute( double R, double Z) const
    {
        return -invB_(R,Z)/R;
    }
    InvB invB_;
};

/**
 * @brief \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  \f]

     \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  = \frac{1}{\hat{R}  \hat{B}^2 } \partial_{\hat{Z}} \hat{B}\f]
 */ 
struct DivCurvatureKappa: public aCloneableBinaryFunctor<DivCurvatureKappa>
{
    DivCurvatureKappa( const TokamakMagneticField& mag): invB_(mag), bZ_(mag){ }
    private:    
    double do_compute( double R, double Z) const
    {
        return bZ_(R,Z)*invB_(R,Z)*invB_(R,Z)/R;
    }
    InvB invB_;
    BZ bZ_;    
};

/**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} \f]
 *
 *    \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} = \frac{1}{\hat{R}\hat{B}^2 } \left[ \hat{B}, \hat{\psi}_p\right]_{\hat{R}\hat{Z}} \f]
 */ 
struct GradLnB: public aCloneableBinaryFunctor<GradLnB>
{
    GradLnB( const TokamakMagneticField& mag): mag_(mag), invB_(mag), bR_(mag), bZ_(mag) { } 
    private:
    double do_compute( double R, double Z) const
    {
        double invB = invB_(R,Z);
        return mag_.R0()*invB*invB*(bR_(R,Z)*mag_.psipZ()(R,Z)-bZ_(R,Z)*mag_.psipR()(R,Z))/R ;
    }
    TokamakMagneticField mag_;
    InvB invB_;
    BR bR_;
    BZ bZ_;   
};

/**
 * @brief \f[ B_\varphi = R_0I/R^2\f]
*/
struct FieldP: public aCloneableBinaryFunctor<FieldP>
{
    FieldP( const TokamakMagneticField& mag): mag_(mag){}
    private:
    double do_compute( double R, double Z) const
    {
        return mag_.R0()*mag_.ipol()(R,Z)/R/R;
    }
    
    TokamakMagneticField mag_;
}; 

/**
 * @brief \f[ B_R = R_0\psi_Z /R\f]
 */
struct FieldR: public aCloneableBinaryFunctor<FieldR>
{
    FieldR( const TokamakMagneticField& mag): mag_(mag){}
    private:
    double do_compute( double R, double Z) const
    {
        return  mag_.R0()/R*mag_.psipZ()(R,Z);
    }
    TokamakMagneticField mag_;
   
};

/**
 * @brief \f[ B_Z = -R_0\psi_R /R\f]
 */
struct FieldZ: public aCloneableBinaryFunctor<FieldZ>
{
    FieldZ( const TokamakMagneticField& mag): mag_(mag){}
    private:
    double do_compute( double R, double Z) const
    {
        return -mag_.R0()/R*mag_.psipR()(R,Z);
    }
    TokamakMagneticField mag_;
};

/**
 * @brief \f[  B^{\theta} = B^R\partial_R\theta + B^Z\partial_Z\theta\f]
 */ 
struct FieldT: public aCloneableBinaryFunctor<FieldT>

{
    FieldT( const TokamakMagneticField& mag):  R_0_(mag.R0()), fieldR_(mag), fieldZ_(mag){}
    /**
     * @brief \f[  B^{\theta} = 
     * B^R\partial_R\theta + B^Z\partial_Z\theta\f]
     * where \f$ \theta \f$ is the geometrical poloidal angle.
     */ 
    private:
    double do_compute(double R, double Z) const
    { 
        double r2 = (R-R_0_)*(R-R_0_) + Z*Z;
        return fieldR_(R,Z)*(-Z/r2) + fieldZ_(R,Z)*(R-R_0_)/r2;
    }
    double R_0_;
    FieldR fieldR_;
    FieldZ fieldZ_;
};

/**
 * @brief \f[ b_R = B_R/|B|\f]
 */
struct BHatR: public aCloneableBinaryFunctor<BHatR>
{
    BHatR( const TokamakMagneticField& mag): mag_(mag), invB_(mag){ }
    private:
    double do_compute( double R, double Z) const
    {
        return  invB_(R,Z)*mag_.R0()/R*mag_.psipZ()(R,Z);
    }
    TokamakMagneticField mag_;
    InvB invB_;

};

/**
 * @brief \f[ b_Z = B_Z/|B|\f]
 */
struct BHatZ: public aCloneableBinaryFunctor<BHatZ>
{
    BHatZ( const TokamakMagneticField& mag): mag_(mag), invB_(mag){ }
    private:
    double do_compute( double R, double Z) const
    {
        return  -invB_(R,Z)*mag_.R0()/R*mag_.psipR()(R,Z);
    }
    TokamakMagneticField mag_;
    InvB invB_;
};

/**
 * @brief \f[ b_\varphi = B_\varphi/|B|\f]
 */
struct BHatP: public aCloneableBinaryFunctor<BHatP>
{
    BHatP( const TokamakMagneticField& mag): mag_(mag), invB_(mag){ }
    private:
    double do_compute( double R, double Z) const
    {
        return invB_(R,Z)*mag_.R0()*mag_.ipol()(R,Z)/R/R;
    }
    TokamakMagneticField mag_;
    InvB invB_;
}; 

///@} 

} //namespace geo
} //namespace dg

