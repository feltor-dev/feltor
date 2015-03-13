#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "geom_parameters_g.h"


/*!@file
 *
 * Geometry objects 
 */
namespace solovev
{
///@addtogroup geom
///@{

   
struct Psip
{

    Psip(GeomParameters gp ):   R_0(gp.R_0), I_0(gp.I_0) {}

    double operator()(double R, double Z)
    {    
        return cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }

    double operator()(double R, double Z, double phi)
    {    
        return cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    }

    void display()
    {

    }
  private:
    double R_0,I_0;
};

struct Ipol
{
    Ipol( GeomParameters gp ):  R_0(gp.R_0), I_0(gp.I_0) {}

    double operator()(double R, double Z)
    {    
        //sign before A changed to -
        return I_0;
    }
    double operator()(double R, double Z, double phi)
    {    
        //sign before A changed to -
      return I_0;
    }
    void display()
    {
    }
  private:
    double R_0,I_0;

};

struct InvB
{
    InvB( GeomParameters gp ):  R_0(gp.R_0), I_0(gp.I_0){}

    double operator()(double R, double Z)
    {    
        return 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    }
    double operator()(double R, double Z, double phi)
    {    
        return 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    }
    void display() { }
  private:
    double R_0,I_0;

};

struct LnB
{
    LnB( GeomParameters gp ):  R_0(gp.R_0), I_0(gp.I_0) {}

    double operator()(double R, double Z)
    {    
        double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return log(1./invB);    }
    double operator()(double R, double Z, double phi)
    {    
        double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return log(1./invB);    }
    void display() { }
  private:
    double R_0,I_0;

};
struct GradLnB
{
    GradLnB(GeomParameters gp ):  R_0(gp.R_0), I_0(gp.I_0) {} 
 
    double operator()( double R, double Z)
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
    double operator()( double R, double Z, double phi)
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
    private:
    double R_0,I_0;

};
struct Field
{
     Field( GeomParameters gp ):  R_0(gp.R_0), I_0(gp.I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {        
            
            yp[2][i] = y[0][i]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0][i]-R_0))*cos(M_PI*y[1][i]))/2./sqrt(2)/I_0;            
            yp[0][i] = -M_PI*y[0][i]*cos(M_PI*(y[0][i]-R_0)/2.)*sin(M_PI*y[1][i]/2)/2./I_0;
            yp[1][i] =  M_PI*y[0][i]*sin(M_PI*(y[0][i]-R_0)/2.)*cos(M_PI*y[1][i]/2)/2./I_0 ;
        }
    }
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
            yp[2] = y[0]*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(y[0]-R_0))*cos(M_PI*y[1]))/2./sqrt(2.)/I_0;            
            yp[0] = -M_PI*y[0]*cos(M_PI*(y[0]-R_0)/2.)*sin(M_PI*y[1]/2)/2./I_0;
            yp[1] =  M_PI*y[0]*sin(M_PI*(y[0]-R_0)/2.)*cos(M_PI*y[1]/2)/2./I_0 ;
    }
    double operator()( double R, double Z)
    {
        return 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    }
    double operator()( double R, double Z, double phi)
    {
        return 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
    }
    private:
    double R_0, I_0;
   
};
/**
 * @brief \f[ b^R\f]
 */
struct bR
{
    bR( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return -invB*M_PI*R_0*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2)/2./R;
    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[ b^Z)\f]
 */
struct bZ
{
    bZ( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return invB*M_PI*R_0*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2)/2./R;

    }
    private:
    double R_0, I_0;
};
/**
 * @brief \f[ b^\phi\f]
 */
struct bPhi
{
    bPhi( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        double invB = 2.*sqrt(2.)*R/sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/R_0;
        return invB*I_0*R_0/R/R;
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
    double operator()( double R, double Z, double phi)
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
    double operator()(double R, double Z, double phi)
    {
        double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
        return -psi*cos(phi);
    //     return -psi;
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
    double operator()(double R, double Z, double phi)
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
    double operator()(double R, double Z, double phi)
    {
        double dldp = R*sqrt(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z))/2./sqrt(2.)/I_0;
        double psi = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
        return psi*sin(phi)/dldp;
    //     return psi/dldp;
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
    double operator()(double R, double Z, double phi)
    {
        double fac1 = 4.*I_0*cos(0.5*M_PI*(R-R_0))*cos(0.5*M_PI*Z);
        double fac2 = 8.*I_0*I_0+M_PI*M_PI - M_PI*M_PI*cos(M_PI*(R-R_0))*cos(M_PI*Z);
        double fac3 = (cos(M_PI*Z)-cos(M_PI*(R-R_0)))*sin(phi)*sin(0.5*M_PI*(R-R_0))*sin(0.5*M_PI*Z);
        double nenner = R*R*(8.*I_0*I_0+M_PI*M_PI - M_PI*M_PI*cos(M_PI*(R-R_0))*cos(M_PI*Z))*(8.*I_0*I_0+M_PI*M_PI - M_PI*M_PI*cos(M_PI*(R-R_0))*cos(M_PI*Z));
        return fac1*(2.*I_0*cos(phi)*fac2 + M_PI*M_PI*R*fac3)/nenner;
    }
    private:
    double R_0, I_0;
};

/**
 * @brief \f[\nabla_\parallel \nabla_\parallel  f\f]
 */
struct DeriNeu2
{
    DeriNeu2( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()(double R, double Z, double phi)
    {
    double psi  = cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
    double fac2 = R*(8.*I_0*I_0+ M_PI*M_PI-M_PI*M_PI* cos(M_PI*(R-R_0))*cos(M_PI*Z));
    double fac3 = 4*I_0*cos(phi)*fac2/R;
    double fac4 = 16.*I_0*I_0*cos(M_PI*0.5*(R-R_0))+
                  M_PI*M_PI*sin(M_PI*0.5*(R-R_0))*
                  (-M_PI*R*cos(M_PI*(R-R_0))+cos(M_PI*Z))+
                  sin(M_PI*(R-R_0))*(1+cos(M_PI*Z))
                  +4*M_PI*M_PI*cos(M_PI*0.5*(R-R_0))*cos(M_PI*0.5*(R-R_0))*cos(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5)*sin(M_PI*Z*0.5);
    double fac5 = M_PI*sin(phi)*sin(0.5*Z*M_PI)*fac4;
    double dz2 = 2.*I_0*psi*(fac3+fac5)/fac2/fac2;
    return dz2 ;
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
    double operator()(double R, double Z, double phi)
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
    double operator()(double R, double Z, double phi)
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
///@} 
} //namespace guenther
