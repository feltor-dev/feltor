#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "guenther_parameters.h"
#include "magnetic_field.h"

//TODO somebody document the functions as in solovev/geometry.h

/*!@file
 *
 * MagneticField objects 
 */
namespace dg
{
namespace geo
{
/**
 * @brief Contains the Guenther type flux functions
 */
namespace guenther
{
///@addtogroup geom
///@{

   
/**
 * @brief \f[\cos(\pi(R-R_0)/2)\cos(\pi Z/2)\f]
 */
struct Psip : public aCloneableBinaryFunctor<Psip>
{
    Psip(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {    
        return cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi\sin(\pi(R-R_0)/2)\cos(\pi Z/2)/2\f]
 */
struct PsipR : public aCloneableBinaryFunctor<PsipR>
{
    PsipR(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {    
        return -M_PI*0.5*sin(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi^2\cos(\pi(R-R_0)/2)\cos(\pi Z/2)/4\f]
 */
struct PsipRR : public aCloneableBinaryFunctor<PsipRR>
{
    PsipRR(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {    
        return -M_PI*M_PI*0.25*cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi\cos(\pi(R-R_0)/2)\sin(\pi Z/2)/2\f]
 */
struct PsipZ : public aCloneableBinaryFunctor<PsipZ>

{
    PsipZ(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {    
        return -M_PI*0.5*cos(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[-\pi^2\cos(\pi(R-R_0)/2)\cos(\pi Z/2)/4\f]
 */
struct PsipZZ : public aCloneableBinaryFunctor<PsipZZ>
{
    PsipZZ(double R_0 ):   R_0(R_0){}
  private:
    double do_compute(double R, double Z) const
    {    
        return -M_PI*M_PI*0.25*cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }
    double R_0;
};
/**
 * @brief \f[ \pi^2\sin(\pi(R-R_0)/2)\sin(\pi Z/2)/4\f]
 */
struct PsipRZ : public aCloneableBinaryFunctor<PsipRZ>
{
    PsipRZ(double R_0 ):   R_0(R_0) {}
  private:
    double do_compute(double R, double Z) const
    {    
        return M_PI*M_PI*0.25*sin(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5);
    }
    double R_0;
};

/**
 * @brief \f[ I_0\f]
 */
struct Ipol : public aCloneableBinaryFunctor<Ipol>
{
    Ipol( double I_0):   I_0(I_0) {}
    private:
    double do_compute(double R, double Z) const { return I_0; }
    double I_0;
};
/**
 * @brief \f[0\f]
 */
struct IpolR : public aCloneableBinaryFunctor<IpolR>
{
    IpolR(  ) {}
    private:
    double do_compute(double R, double Z) const { return 0; }
};
/**
 * @brief \f[0\f]
 */
struct IpolZ : public aCloneableBinaryFunctor<IpolZ>
{
    IpolZ(  ) {}
    private:
    double do_compute(double R, double Z) const { return 0; }
};

BinaryFunctorsLvl2 createPsip( double R_0)
{
    BinaryFunctorsLvl2 psip( new Psip(R_0), new PsipR(R_0), new PsipZ(R_0),new PsipRR(R_0), new PsipRZ(R_0), new PsipZZ(R_0));
    return psip;
}
BinaryFunctorsLvl1 createIpol( double I_0)
{
    BinaryFunctorsLvl1 ipol( new Ipol(I_0), new IpolR(), new IpolZ());
    return ipol;
}
TokamakMagneticField createMagField( double R_0, double I_0)
{
    return TokamakMagneticField( R_0, createPsip(R_0), createIpol(I_0));
}
///@}

/////////////////////////////////////////These should not be necessary!////////
///@cond
struct InvB
{
    InvB( double R_0, double I_0 ):  R_0(R_0), I_0(I_0){}

    double operator()(double R, double Z) const
    {    
        return sqrt(8.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    }
    double operator()(double R, double Z, double phi) const
    {  

        return sqrt(8.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;

    }
  private: 
    double R_0,I_0;

};
struct B
{
    B( double R_0, double I_0):  R_0(R_0), I_0(I_0){}

    double operator()(double R, double Z) const
    {    
        return sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))*R_0/ (sqrt(8.)*R);
    }
    double operator()(double R, double Z, double phi) const
    {    
        return sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))*R_0/ (sqrt(8.)*R);

    }
  private:
    double R_0,I_0;

};
struct LnB
{
    LnB( double R_0, double I_0 ):  R_0(R_0), I_0(I_0) {}

    double operator()(double R, double Z) const
    {    
        double invB = sqrt(8.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return log(1./invB);    }
    double operator()(double R, double Z, double phi) const
    {    
        double invB = sqrt(8.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;

        return log(1./invB);    }
  private:
    double R_0,I_0;

};
struct GradLnB
{
    GradLnB(double R_0, double I_0):  R_0(R_0), I_0(I_0) {} 
 
    double operator()( double R, double Z) const
    {
        double fac1 = sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
        double z1 = cos(M_PI*0.5*(R-R_0))*(32.*I_0*I_0+5.*M_PI*M_PI)+
        M_PI*M_PI* cos(M_PI*3.*(R-R_0)/2.)+
        M_PI*R*sin(M_PI*3.*(R-R_0)/2.) ;
        double z2 = cos(M_PI*0.5*(R-R_0)) + 
        cos(M_PI*3*(R-R_0)/2) + 
        M_PI*R*sin(M_PI*0.5*(R-R_0));
        double nenner = fac1*fac1*fac1*2.*sqrt(2.)*R;
        double divb = -M_PI*(z1*sin(M_PI*Z*0.5)-z2*M_PI*M_PI*sin(M_PI*Z*3./2.))/(nenner);
       return -divb ;
    }
    /*
    double operator()( double R, double Z, double phi) const
    {
        double fac1 = sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
        double z1 = cos(M_PI*0.5*(R-R_0))*(32.*I_0*I_0+5.*M_PI*M_PI)+
        M_PI*M_PI* cos(M_PI*3.*(R-R_0)/2.)+
        M_PI*R*sin(M_PI*3.*(R-R_0)/2.) ;
        double z2 = cos(M_PI*0.5*(R-R_0)) + 
        cos(M_PI*3*(R-R_0)/2) + 
        M_PI*R*sin(M_PI*0.5*(R-R_0));
        double nenner = fac1*fac1*fac1*2.*sqrt(2.)*R;
        double divb = -M_PI*(z1*sin(M_PI*Z*0.5)-z2*M_PI*M_PI*sin(M_PI*Z*3./2.))/(nenner);
       return -divb ;
    }
    */
    double operator()( double R, double Z, double phi)const{return operator()(R,Z);}
    private:
    double R_0,I_0;

};
struct Field
{
     Field( double R0, double I0):  R_0(R0), I_0(I0){}
     Field( GeomParameters gp ):  R_0(gp.R_0), I_0(gp.I_0){}
    void operator()( const std::vector<thrust::host_vector<double> >& y, std::vector<thrust::host_vector<double> >& yp) const
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {        

            yp[2][i] = y[0][i]*sqrt(1.+ M_PI*M_PI*(1.- cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/8./I_0/I_0);            
            yp[0][i] = -M_PI*y[0][i]*cos(M_PI*(y[0][i]-R_0)/2.)*sin(M_PI*y[1][i]/2.)/2./I_0;
            yp[1][i] =  M_PI*y[0][i]*sin(M_PI*(y[0][i]-R_0)/2.)*cos(M_PI*y[1][i]/2.)/2./I_0 ;

        }
    }
    void operator()( const thrust::host_vector<double> & y, thrust::host_vector<double> & yp) const
    {
            yp[2] = y[0]*sqrt(1.+ M_PI*M_PI*(1.-cos(M_PI*(y[0]-R_0))*cos(M_PI*y[1]))/8./I_0/I_0);
            yp[0] = -M_PI*y[0]*cos(M_PI*(y[0]-R_0)/2.)*sin(M_PI*y[1]/2.)/2./I_0;
            yp[1] =  M_PI*y[0]*sin(M_PI*(y[0]-R_0)/2.)*cos(M_PI*y[1]/2.)/2./I_0 ;
    }
    double operator()( double R, double Z) const
    {

        return sqrt(8.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    }
    double operator()( double R, double Z, double phi) const
    {

        return sqrt(8.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        
    }
    double error( const dg::HVec& x0, const dg::HVec& x1)
    {
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end){ 
        if ( std::isnan(end[0]) || std::isnan(end[1]) || std::isnan(end[2]) ) 
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
    double R_0, I_0;
   
};
/**
 * @brief \f[ b^R\f]
 */
struct FieldR
{
    FieldR( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi) const
    {
        return -sqrt(2.)*M_PI*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2)/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[ b^Z)\f]
 */
struct FieldZ
{
    FieldZ( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi) const
    {
        return sqrt(2.)*M_PI*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2)/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));

    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[ b^\phi\f]
 */
struct FieldP
{
    FieldP( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi) const
    {
        return 2.*sqrt(2.)*I_0/R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    }
    private:
    double R_0, I_0;
}; 
/**
 * @brief \f[ b_t\f]
 */
struct bt
{
    bt( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi) const
    {
        double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return invB*I_0*R_0/R;
    }
    private:
    double R_0, I_0;
}; 
/////////////////////Test functions//////////////////
/**
 * @brief \f[ f(R,Z,\phi)\f]
 */
struct FuncNeu
{
    FuncNeu( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi) const
    {
        double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
        return -psi*cos(phi);
        //return cos(phi);
        //return -psi;
    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[ f2(R,Z,\phi)\f]
 */
struct FuncNeu2
{
    FuncNeu2( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi) const
    {
        double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
        return -psi*cos(phi)+0.5*(R-R_0)*0.5*(R-R_0) +Z*0.5*0.5*(R-R_0) ;
    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[\nabla_\parallel f(R,Z,\phi)\f]
 */
struct DeriNeu
{
    DeriNeu( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi) const
    {
        double dldp = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/I_0;
        double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
        return psi*sin(phi)/dldp;
        //return -sin(phi)/dldp;
        //return 0;
    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[\Delta_\parallel f\f]
 */
struct DeriNeuT2
{
    DeriNeuT2( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi) const
    {
        double fac1 = 4.*I_0*cos(0.5*M_PI*(R-R_0))*cos(0.5*M_PI*Z);
        double fac2 = 8.*I_0*I_0+M_PI*M_PI - M_PI*M_PI*cos(M_PI*(R-R_0))*cos(M_PI*Z);
        double fac3 = (cos(M_PI*Z)-cos(M_PI*(R-R_0)))*
                        sin(phi)*sin(0.5*M_PI*(R-R_0))*sin(0.5*M_PI*Z);
        double nenner = R*R*(8.*I_0*I_0+M_PI*M_PI - M_PI*M_PI*cos(M_PI*(R-R_0))*cos(M_PI*Z))*(8.*I_0*I_0+M_PI*M_PI - M_PI*M_PI*cos(M_PI*(R-R_0))*cos(M_PI*Z));
        return fac1*(2.*I_0*cos(phi)*fac2 + M_PI*M_PI*M_PI*M_PI*R*fac3)/nenner;

    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[\Delta_\parallel f\f]
 */
struct FuncMinusDeriNeuT2
{
    FuncMinusDeriNeuT2( double R_0, double I_0): func_(R_0, I_0), der_(R_0, I_0){}
    double operator()(double R, double Z, double phi) const { return func_(R,Z,phi) - der_(R,Z,phi); }
    private:
    FuncNeu func_;
    DeriNeuT2 der_;
};

/**
 * @brief \f[\nabla_\parallel \nabla_\parallel  f\f]
 */
struct DeriNeu2
{
    DeriNeu2( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi) const
    {
        double cosfac = cos(M_PI*0.5*(R-R_0));
        double psi  = cosfac*cos(M_PI*Z*0.5);
//         double fac2 = R*(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
        double fac2 = R*(8.*I_0*I_0+ M_PI*M_PI*(1.-cos(M_PI*(R-R_0))*cos(M_PI*Z)));
        double fac3 = 4.*I_0*cos(phi)*fac2/R;
        double fac4 = 16.*I_0*I_0*cosfac+ M_PI*M_PI*sin(M_PI*0.5*(R-R_0))*
                    (-M_PI*R*(cos(M_PI*(R-R_0))+cos(M_PI*Z)) + sin(M_PI*(R-R_0))*(1.+cos(M_PI*Z)))
                    + 4.*M_PI*M_PI*cosfac*cosfac*cosfac*sin(M_PI*Z*0.5)*sin(M_PI*Z*0.5);
        double fac5 = M_PI*sin(phi)*sin(0.5*Z*M_PI)*fac4;
        return 2.*I_0*psi*(fac3+fac5)/fac2/fac2;
    }
    private:
    double R_0, I_0;
};

/**
 * @brief \f[\nabla b  f\f]
 */
struct DeriNeuT
{
    DeriNeuT( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi) const
    {
        double dldp = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/I_0;
        double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
        //for divb
        double fac1 = sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
        double z1 = cos(M_PI*0.5*(R-R_0))*(32.*I_0*I_0+5.*M_PI*M_PI)+
                    M_PI*M_PI* cos(M_PI*3.*(R-R_0)/2.)+
                    M_PI*R*sin(M_PI*3.*(R-R_0)/2.) ;
        double z2 = cos(M_PI*0.5*(R-R_0)) + 
                    cos(M_PI*3*(R-R_0)/2) + 
                    M_PI*R*sin(M_PI*0.5*(R-R_0));
        double nenner = fac1*fac1*fac1*2.*sqrt(2.)*R;
        double divb = -M_PI*(z1*sin(M_PI*Z*0.5)-z2*M_PI*M_PI*sin(M_PI*Z*3./2.))/(nenner);
        
        double func = -psi*cos(phi);
        double deri = psi*sin(phi)/dldp;
        return divb*func + deri;
    }
    private:
    double R_0, I_0;
};


/**
 * @brief \f[\nabla b\f]
 */
struct Divb
{
    Divb( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi) const
    {
        double fac1 = sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
        double z1 = cos(M_PI*0.5*(R-R_0))*(32.*I_0*I_0+5.*M_PI*M_PI)+
                    M_PI*M_PI* cos(M_PI*3.*(R-R_0)/2.)+
                    M_PI*R*sin(M_PI*3.*(R-R_0)/2.) ;
        double z2 = cos(M_PI*0.5*(R-R_0)) + 
                    cos(M_PI*3*(R-R_0)/2) + 
                    M_PI*R*sin(M_PI*0.5*(R-R_0));
        double nenner = fac1*fac1*fac1*2.*sqrt(2.)*R;
        double divb = -M_PI*(z1*sin(M_PI*Z*0.5)-z2*M_PI*M_PI*sin(M_PI*Z*3./2.))/(nenner);
        return divb;
    }
    private:
    double R_0, I_0;
};
///@endcond
} //namespace guenther
TokamakMagneticField createGuentherField( double R_0, double I_0)
{
    return TokamakMagneticField( R_0, guenther::createPsip(R_0), guenther::createIpol(I_0));
}
} //namespace geo
}//namespace dg
