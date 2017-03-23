#pragma once


/*!@file
 *
 * Geometry objects 
 */
namespace dg
{
namespace geo
{
///@addtogroup magnetic
///@{

/**
 * @brief \f[   |B| = R_0\sqrt{I^2+(\nabla\psi)^2}/R   \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct Bmodule
{
    Bmodule( const Geometry& c, double R0 ):  R_0_(R0), c_(c)  { }
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
    Geometry c_;
};

/**
 * @brief \f[  |B|^{-1} = R/R_0\sqrt{I^2+(\nabla\psi)^2}    \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct InvB
{
    InvB(  const Geometry& c, double R0 ):  R_0_(R0), c_(c)  { }
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
    Geometry c_;
};

/**
 * @brief \f[   \ln{|B|}  \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct LnB
{
    LnB(  const Geometry& c, double R0 ):  R_0_(R0), c_(c)  { }
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
    Geometry c_;
};

/**
 * @brief \f[  \frac{\partial |\hat{B}| }{ \partial \hat{R}}  \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct BR
{
    BR(const Geometry& c, double R0):  R_0_(R0), invB_(c, R0), c_(c) { }
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
    InvB<Geometry> invB_;
    Geometry c_;
};

/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}}  \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct BZ
{

    BZ(const Geometry& c, double R0):  R_0_(R0), c_(c), invB_(c, R0) { }
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
    Geometry c_;
    InvB<Geometry> invB_; 
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\nabla B} \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct CurvatureNablaBR
{
    CurvatureNablaBR(const Geometry& c, double R0 ): invB_(c, R0), bZ_(c, R0) { }
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
    InvB<Geometry>   invB_;
    BZ<Geometry> bZ_;    
};

/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}_{\nabla B}  \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct CurvatureNablaBZ
{
    CurvatureNablaBZ( const Geometry& c, double R0): invB_(c, R0), bR_(c, R0) { }
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
    InvB<Geometry> invB_;
    BR<Geometry> bR_;   
};

/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}_{\vec{\kappa}} \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
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
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct CurvatureKappaZ
{
    CurvatureKappaZ( const Geometry c, double R0):
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
    InvB<Geometry>   invB_;
};

/**
 * @brief \f[  \vec{\hat{\nabla}}\cdot \mathcal{\hat{K}}_{\vec{\kappa}}  \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct DivCurvatureKappa
{
    DivCurvatureKappa( const Geometry& c, double R0):
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
    InvB<Geometry>   invB_;
    BZ<Geometry> bZ_;    
};

/**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} \f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct GradLnB
{
    GradLnB( const Geometry& c, double R0): R_0_(R0), c_(c), invB_(c, R0), bR_(c, R0), bZ_(c, R0) { } 
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
    Geometry c_;
    InvB<Geometry>   invB_;
    BR<Geometry> bR_;
    BZ<Geometry> bZ_;   
};

/**
 * @brief \f[ B_\varphi = R_0I/R^2\f]
 @tparam Geometry models aTokamakGeometry
*/
template<class Geometry>
struct FieldP
{
    FieldP( const Geometry& c, double R0): R_0(R0), c_(c){}
    double operator()( double R, double Z, double phi) const
    {
        return R_0*c_.ipol(R,Z)/R/R;
    }
    
    private:
    double R_0;
    Geometry c_;
}; 

/**
 * @brief \f[ B_R = R_0\psi_Z /R\f]
 @tparam Geometry models aTokamakGeometry
 */
template<class Geometry>
struct FieldR
{
    FieldR( const Geometry& c, double R0): R_0(R0), c_(c){}
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
    Geometry c_;
   
};

/**
 * @brief \f[ B_Z = -R_0\psi_R /R\f]
 @tparam Geometry models aTokamakGeometry
 */
template<class Geometry>
struct FieldZ
{
    FieldZ( const Geometry& c, double R0): R_0(R0), c_(c){}
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
    Geometry c_;
   
};

/**
 * @brief \f[  B^{\theta} = B^R\partial_R\theta + B^Z\partial_Z\theta\f]
 @tparam Geometry models aTokamakGeometry
 */ 
template<class Geometry>
struct FieldT
{
    FieldT( const Geometry& c, double R0):  R_0_(R0), fieldR_(c, R0), fieldZ_(c, R0){}
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
    FieldR<Geometry> fieldR_;
    FieldZ<Geometry> fieldZ_;

};

/**
 * @brief \f[ b_R = B_R/|B|\f]
 @tparam Geometry models aTokamakGeometry
 */
template<class Geometry>
struct BHatR
{
    BHatR( const Geometry& c, double R0): c_(c), R_0(R0), invB_(c, R0){ }
    double operator()( double R, double Z, double phi) const
    {
        return  invB_(R,Z)*R_0/R*c_.psipZ(R,Z);
    }
    private:
    Geometry c_;
    double R_0;
    InvB<Geometry>   invB_;

};

/**
 * @brief \f[ b_Z = B_Z/|B|\f]
 @tparam Geometry models aTokamakGeometry
 */
template<class Geometry>
struct BHatZ
{
    BHatZ( const Geometry& c, double R0): c_(c), R_0(R0), invB_(c, R0){ }

    double operator()( double R, double Z, double phi) const
    {
        return  -invB_(R,Z)*R_0/R*c_.psipR(R,Z);
    }
    private:
    Geometry c_;
    double R_0;
    InvB<Geometry>   invB_;

};

/**
 * @brief \f[ b_\varphi = B_\varphi/|B|\f]
 @tparam Geometry models aTokamakGeometry
 */
template<class Geometry>
struct BHatP
{
    BHatP( const Geometry& c, double R0): c_(c), R_0(R0), invB_(c, R0){ }
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z)*R_0*c_.ipol(R,Z)/R/R;
    }
    
    private:
    Geometry c_;
    double R_0;
    InvB<Geometry>   invB_;
  
}; 

///@} 

/**
 * @brief Integrates the equations for a field line and 1/B
 * @tparam Geometry models aTokamakGeometry
 * @ingroup misc
 */ 
template<class Geometry>
struct Field
{
    Field( const Geometry& c, double R0):c_(c), R_0_(R0), invB_(c, R0) { }
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
    Geometry c_;
    double R_0_;
    InvB<Geometry>   invB_;
   
};

///**
// * @brief Integrates the equations for a field line and 1/B
// */ 
//template<class Geometry>
//struct OrthogonalField
//{
//    OrthogonalField( const Geometry& c, double R0, dg::solovev::GeomParameters gp, const dg::Grid2d& gXY, const thrust::host_vector<double>& f2):
//        c_(c), R_0_(R0), invB_(c, R0), gXY_(gXY), 
//        g_(dg::create::forward_transform(f2, gXY)) 
//    { }
//
//    /**
//     * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
//     \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
//     \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I}  \hat{R}_0}  \f]
//     */ 
//    void operator()( const dg::HVec& y, dg::HVec& yp)
//    {
//        //x,y,s,R,Z
//        double psipR = c_.psipR(y[3],y[4]), psipZ = c_.psipZ(y[3],y[4]), ipol = c_.ipol( y[3],y[4]);
//        double xs = y[0],ys=y[1];
//        gXY_.shift_topologic( y[0], M_PI, xs,ys);
//        double g = dg::interpolate( xs,  ys, g_, gXY_);
//        yp[0] = 0;
//        yp[1] = y[3]*g*(psipR*psipR+psipZ*psipZ)/ipol;
//        //yp[1] = g/ipol;
//        yp[2] =  y[3]*y[3]/invB_(y[3],y[4])/ipol/R_0_; //ds/dphi =  R^2 B/I/R_0_hat
//        yp[3] =  y[3]*psipZ/ipol;              //dR/dphi =  R/I Psip_Z
//        yp[4] = -y[3]*psipR/ipol;             //dZ/dphi = -R/I Psip_R
//
//    }
//    /**
//     * @brief \f[   \frac{1}{\hat{B}} = 
//      \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
//      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
//     */ 
//    double operator()( double R, double Z) const { return invB_(R,Z); }
//    /**
//     * @brief == operator()(R,Z)
//     */ 
//    double operator()( double R, double Z, double phi) const { return invB_(R,Z,phi); }
//    double error( const dg::HVec& x0, const dg::HVec& x1)
//    {
//        //compute error in x,y,s
//        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
//    }
//    bool monitor( const dg::HVec& end){ 
//        if ( isnan(end[1]) || isnan(end[2]) || isnan(end[3])||isnan( end[4]) ) 
//        {
//            return false;
//        }
//        if( (end[3] < 1e-5) || end[3]*end[3] > 1e10 ||end[1]*end[1] > 1e10 ||end[2]*end[2] > 1e10 ||(end[4]*end[4] > 1e10) )
//        {
//            return false;
//        }
//        return true;
//    }
//    
//    private:
//    Geometry c_;
//    double R_0_;
//    dg::magnetic::InvB<Geometry>   invB_;
//    const dg::Grid2d gXY_;
//    thrust::host_vector<double> g_;
//};



} //namespace geo
} //namespace dg

