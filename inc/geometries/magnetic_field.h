#pragma once


/*!@file
 *
 * MagneticField objects 
 */
namespace dg
{
namespace geo
{

struct aBinaryFunctor
{
    virtual double operator()(double R, double Z) const=0;
    double operator()(double R, double Z, double phi)
    {
        return this->operator()(R,Z);
    }
    virtual aBinaryFunctor* clone()const=0;
    virtual ~aBinaryFunctor(){}
};

//https://katyscode.wordpress.com/2013/08/22/c-polymorphic-cloning-and-the-crtp-curiously-recurring-template-pattern/
template<class Derived<
struct aCloneableBinaryFunctor : public a BinaryFunctor
{
    virtual aBinaryFunctor* clone() const
    {
        return new Derived(statimag_cast<Derived const &>(*this));
    }
};
/**
 * @brief Contains all solovev fields (models aTokamakMagneticField)
 */
struct aTokamakMagneticField
{
    aTokamakMagneticField( const aTokamakMagneticField& mag)
    {
        R0_ = mag.R0_;
        for( unsigned i=0; i<p_.size(); i++)
            p_[i] = mag.p_[i]->clone();
    }
    aTokamakMagneticField& operator=( const aTokamakMagneticField& mag)
    {
        aTokamakMagneticField temp(mag);
        std::swap( temp.p_, p_);
        return *this;
    }
    double R0()const {return R0_;}
    const aBinaryFunctor& psip()const{return *p_[0];}
    const aBinaryFunctor& psipR()const{return *p_[1];}
    const aBinaryFunctor& psipZ()const{return *p_[2];}
    const aBinaryFunctor& psipRR()const{return *p_[3];}
    const aBinaryFunctor& psipRZ()const{return *p_[4];}
    const aBinaryFunctor& psipZZ()const{return *p_[5];}
    const aBinaryFunctor& laplacePsip()const{return *p_[6];}
    const aBinaryFunctor& ipol()const{return *p_[7];}
    const aBinaryFunctor& ipolR()const{return *p_[8];}
    const aBinaryFunctor& ipolZ()const{return *p_[9];}

    protected:
    ~aTokamakMagneticField(){
        for( unsigned i=0; i<p_.size(); i++)
            delete p_[i];
    }
    aTokamakMagneticField( double R0,
        aBinaryFunctor* psip,
        aBinaryFunctor* psipR,
        aBinaryFunctor* psipZ,
        aBinaryFunctor* psipRR,
        aBinaryFunctor* psipRZ,
        aBinaryFunctor* psipZZ,
        aBinaryFunctor* laplacePsip,
        aBinaryFunctor* ipol,
        aBinaryFunctor* ipolR,
        aBinaryFunctor* ipolZ
        ):p_(10){ 
            p_[0] = psip,
            p_[1] = psipR,
            p_[2] = psipZ,
            p_[3] = psipRR,
            p_[4] = psipRZ,
            p_[5] = psipZZ,
            p_[6] = laplacePsip,
            p_[7] = psip,
            p_[8] = ipolR,
            p_[9] = ipolZ,
        }
    private:
    double R0_;
    std::vector<aBinaryFunctor*> p_;
};

///@addtogroup magnetic
///@{

/**
 * @brief \f[   |B| = R_0\sqrt{I^2+(\nabla\psi)^2}/R   \f]
 */ 
struct Bmodule : public aCloneableBinaryFunctor<Bmodule>
{
    Bmodule( const aTokamakMagneticField& mag): mag_(mag)  { }
    /**
    * @brief \f[   \hat{B} \f]
    */ 
    double operator()(double R, double Z) const
    {    
        double psipR = mag_.psipR()(R,Z), psipZ = mag_.psipZ()(R,Z), ipol = mag_.ipol()(R,Z);
        return mag_.R0()/R*sqrt(ipol*ipol+psipR*psipR +psipZ*psipZ);
    }
  private:
    aTokamakMagneticField mag_;
};

/**
 * @brief \f[  |B|^{-1} = R/R_0\sqrt{I^2+(\nabla\psi)^2}    \f]
 */ 
struct InvB : public aCloneableBinaryFunctor<InvB>
{
    InvB(  const aTokamakMagneticField& mag): mag_(mag){ }
    /**
    * @brief \f[   \frac{1}{\hat{B}} = 
        \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
        + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
    */ 
    double operator()(double R, double Z) const
    {    
        double psipR = mag_.psipR()(R,Z), psipZ = mag_.psipZ()(R,Z), ipol = mag_.ipol()(R,Z);
        return R/(mag_.R0()*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
  private:
    aTokamakMagneticField mag_;
};

/**
 * @brief \f[   \ln{|B|}  \f]
 */ 
struct LnB : public aCloneableBinaryFunctor<LnB>
{
    LnB(const aTokamakMagneticField& mag): mag_(mag) { }
    /**
     * @brief \f[   \ln{(   \hat{B})} = \ln{\left[
          \frac{\hat{R}_0}{\hat{R}} \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
          + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2} \right] } \f]
     */ 
    double operator()(double R, double Z) const
    {    
        double psipR = mag_.psipR()(R,Z), psipZ = mag_.psipZ()(R,Z), ipol = mag_.ipol()(R,Z);
        return log(mag_.R0()/R*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
  private:
    aTokamakMagneticField mag_;
};

/**
 * @brief \f[  \frac{\partial |\hat{B}| }{ \partial \hat{R}}  \f]
 */ 
struct BR: public aCloneableBinaryFunctor<BR>

{
    BR(const aTokamakMagneticField& mag): invB_(mag), mag_(mag) { }
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
        Rn = R/mag.R0();
        //sign before A changed to +
        //return -( Rn*Rn/invB_(R,Z)/invB_(R,Z)+ qampl_*qampl_*Rn *A_*psipR_(R,Z) - R  *(psipZ_(R,Z)*psipRZ_(R,Z)+psipR_(R,Z)*psipRR_(R,Z)))/(R*Rn*Rn/invB_(R,Z));
        return -1./R/invB_(R,Z) + invB_(R,Z)/Rn/Rn*(mag_.ipol()(R,Z)*mag_.ipolR()(R,Z) + mag_.psipR()(R,Z)*mag_.psipRR()(R,Z) + mag_.psipZ()(R,Z)*mag_.psipRZ()(R,Z));
    }
  private:
    InvB invB_;
    aTokamakMagneticField mag_;
};

/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}}  \f]
 */ 
struct BZ: public aCloneableBinaryFunctor<BZ>
{
    BZ(const aTokamakMagneticField& mag ): mag_(mag), invB_(mag) { }
    /**
     * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}} = 
     \frac{ \hat I \left(\frac{\partial \hat I}{\partial\hat Z}    \right)+
     \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
          + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2} \right)}{\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f]
     */ 
    double operator()(double R, double Z) const
    { 
        double Rn;
        Rn = R/mag_.R0();
        //sign before A changed to -
        //return (-qampl_*qampl_*A_/R_0_*psipZ_(R,Z) + psipR_(R,Z)*psipRZ_(R,Z)+psipZ_(R,Z)*psipZZ_(R,Z))/(Rn*Rn/invB_(R,Z));
        return (invB_(R,Z)/Rn/Rn)*(mag_.ipol()(R,Z)*mag_.ipolZ(R,Z) + mag_.psipR()(R,Z)*mag_.psipRZ()(R,Z) + mag_.psipZ()(R,Z)*mag_.psipZZ()(R,Z));
    }
  private:
    aTokamakMagneticField mag_;
    InvB invB_; 
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} \f]
 */ 
struct CurvatureNablaBR: public aCloneableBinaryFunctor<CurvatureNablaBR>
{
    CurvatureNablaBR(const aTokamakMagneticField& mag): invB_(mag), bZ_(mag) { }
    /**
     * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} =-\frac{1}{ \hat{B}^2}  \frac{\partial \hat{B}}{\partial \hat{Z}}  \f]
     */ 
    double operator()( double R, double Z) const
    {
        return -invB_(R,Z)*invB_(R,Z)*bZ_(R,Z); 
    }
    private:    
    InvB invB_;
    BZ bZ_;    
};

/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B}  \f]
 */ 
struct CurvatureNablaBZ: public aCloneableBinaryFunctor<CurvatureNablaBZ>
{
    CurvatureNablaBZ( const aTokamakMagneticField& mag): invB_(mag), bR_(mag) { }
    /**
     * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B} =\frac{1}{ \hat{B}^2}   \frac{\partial \hat{B}}{\partial \hat{R}} \f]
     */    
    double operator()( double R, double Z) const
    {
        return invB_(R,Z)*invB_(R,Z)*bR_(R,Z);
    }
    private:    
    InvB invB_;
    BR bR_;   
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}}=0 \f]
 */ 
struct CurvatureKappaR: public aCloneableBinaryFunctor<CurvatureKappaR>
{
    CurvatureKappaR( ){ }
    CurvatureKappaR( const aTokamakMagneticField& mag){ }
    /**
     * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}} =0  \f]
     */ 
    double operator()( double R, double Z) const
    {
        return  0.;
    }
};

/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\vec{\kappa}}  \f]
 */ 
struct CurvatureKappaZ: public aCloneableBinaryFunctor<CurvatureKappaZ>
{
    CurvatureKappaZ( const aTokamakMagneticField& mag): invB_(mag) { }
    /**
     * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\vec{\kappa}} = - \frac{1}{\hat{R} \hat{B}} \f]
     */    
    double operator()( double R, double Z) const
    {
        return -invB_(R,Z)/R;
    }
    private:    
    InvB invB_;
};

/**
 * @brief \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  \f]
 */ 
struct DivCurvatureKappa: public aCloneableBinaryFunctor<DivCurvatureKappa>
{
    DivCurvatureKappa( const aTokamakMagneticField& mag): invB_(mag), bZ_(mag){ }
    /**
     * @brief \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  = \frac{1}{\hat{R}  \hat{B}^2 } \partial_{\hat{Z}} \hat{B}\f]
     */    
    double operator()( double R, double Z) const
    {
        return bZ_(R,Z)*invB_(R,Z)*invB_(R,Z)/R;
    }
    private:    
    InvB invB_;
    BZ bZ_;    
};

/**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} \f]
 */ 
struct GradLnB: public aCloneableBinaryFunctor<GradLnB>
{
    GradLnB( const aTokamakMagneticField& mag): mag_(mag), invB_(mag), bR_(mag), bZ_(mag) { } 
    /**
     * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} = \frac{1}{\hat{R}\hat{B}^2 } \left[ \hat{B}, \hat{\psi}_p\right]_{\hat{R}\hat{Z}} \f]
     */ 
    double operator()( double R, double Z) const
    {
        double invB = invB_(R,Z);
        return mag_.R0()*invB*invB*(bR_(R,Z)*mag_.psipZ()(R,Z)-bZ_(R,Z)*mag_.psipR()(R,Z))/R ;
    }
    private:
    aTokamakMagneticField mag_;
    InvB invB_;
    BR bR_;
    BZ bZ_;   
};

/**
 * @brief \f[ B_\varphi = R_0I/R^2\f]
*/
struct FieldP: public aCloneableBinaryFunctor<LnB>
{
    FieldP( const aTokamakMagneticField& mag): mag_(mag){}
    double operator()( double R, double Z, double phi) const
    {
        return mag.R0()*mag_.ipol()(R,Z)/R/R;
    }
    
    private:
    aTokamakMagneticField mag_;
}; 

/**
 * @brief \f[ B_R = R_0\psi_Z /R\f]
 */
struct FieldR: public aCloneableBinaryFunctor<FieldR>
{
    FieldR( const aTokamakMagneticField& mag): mag_(mag){}
    double operator()( double R, double Z) const
    {
        return  mag.R0()/R*mag_.psipZ()(R,Z);
    }
    private:
    aTokamakMagneticField mag_;
   
};

/**
 * @brief \f[ B_Z = -R_0\psi_R /R\f]
 */
struct FieldZ: public aCloneableBinaryFunctor<FieldZ>
{
    FieldZ( const aTokamakMagneticField& mag): mag_(mag){}
    double operator()( double R, double Z) const
    {
        return -mag_.R0()/R*mag_.psipR()(R,Z);
    }
    private:
    aTokamakMagneticField mag_;
};

/**
 * @brief \f[  B^{\theta} = B^R\partial_R\theta + B^Z\partial_Z\theta\f]
 */ 
struct FieldT: public aCloneableBinaryFunctor<FieldT>

{
    FieldT( const aTokamakMagneticField& mag):  R_0_(mag.R0()), fieldR_(mag), fieldZ_(mag){}
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
    private:
    double R_0_;
    FieldR fieldR_;
    FieldZ fieldZ_;
};

/**
 * @brief \f[ b_R = B_R/|B|\f]
 */
struct BHatR: public aCloneableBinaryFunctor<BHatR>
{
    BHatR( const aTokamakMagneticField& mag): mag_(mag), invB_(mag){ }
    double operator()( double R, double Z) const
    {
        return  invB_(R,Z)*mag_.R0()/R*mag_.psipZ()(R,Z);
    }
    private:
    aTokamakMagneticField mag_;
    InvB invB_;

};

/**
 * @brief \f[ b_Z = B_Z/|B|\f]
 */
struct BHatZ: public aCloneableBinaryFunctor<BHatZ>
{
    BHatZ( const aTokamakMagneticField& mag): mag_(mag), invB_(mag){ }
    double operator()( double R, double Z) const
    {
        return  -invB_(R,Z)*mag_.R0()/R*mag_.psipR()(R,Z);
    }
    private:
    aTokamakMagneticField mag_;
    InvB invB_;
};

/**
 * @brief \f[ b_\varphi = B_\varphi/|B|\f]
 */
struct BHatP: public aCloneableBinaryFunctor<BHatP>
{
    BHatP( const aTokamakMagneticField& mag): mag_(mag), invB_(mag){ }
    double operator()( double R, double Z) const
    {
        return invB_(R,Z)*mag_.R0()*mag_.ipol()(R,Z)/R/R;
    }
    
    private:
    aTokamakMagneticField mag_;
    InvB invB_;
}; 

///@} 

} //namespace geo
} //namespace dg

