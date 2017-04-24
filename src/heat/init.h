#pragma once
// #include "solovev/geometry.h"
#include "geometry_g.h"
#include "heat/parameters.h"

/*!@file
 *
 * Initialize and Damping objects
 * @deprecated
 */
namespace solovev
{
///@addtogroup geom
///@{
/**
 * @brief Returns zero outside psipmax and inside psipmin, otherwise 1
 */ 
struct Iris
{
    Iris( GeomParameters gp ): 
        gp_(gp),
        psip_(gp) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax) return 0.;
        if( psip_(R,Z) < gp_.psipmin) return 0.;
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Returns zero outside psipmaxcut otherwise 1
 */ 
struct Pupil
{
    Pupil( GeomParameters gp): 
        gp_(gp),
        psip_(gp) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmaxcut) return 0.;
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Returns psi inside psipmax and psipmax outside psipmax
 */ 
struct PsiPupil
{
    PsiPupil( GeomParameters gp, double psipmax): 
        gp_(gp),
        psipmax_(psipmax),
        psip_(gp) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > psipmax_) return psipmax_;
        return  psip_(R,Z);
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    GeomParameters gp_;
    double psipmax_;
    Psip psip_;
};
/**
 * @brief Sets values to one outside psipmaxcut, zero else
 *
 * \f$ 1 \f$, if \f$ \psi_p(R,Z) > \psi_{p,maxcut}\f$
 *
 * \f$ 0 \f$, if \f$ \psi_p(R,Z) < \psi_{p,maxcut}\f$
 */ 
struct PsiLimiter
{
    PsiLimiter( GeomParameters gp): 
        gp_(gp),
        psip_(gp) {
        }

    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmaxlim) return 1.;
        return 0.;
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};



/**
 * @brief Damps the outer boundary in a zone 
 * from psipmaxcut to psipmaxcut+ 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
 *
 * \f$ 0 \f$, if \f$ \psi_p(R,Z) > \psi_{p,max,cut} + 4\alpha \f$
 *
 * \f$ 1 \f$, if \f$ \psi_p(R,Z) < \psi_{p,max,cut}\f$
 *
 * \f$ \exp\left( - \frac{(\psi_p - \psi_{p,max,cut})^2}{2\alpha^2}\right)\f$, else
 */ 
struct GaussianDamping
{
    GaussianDamping( GeomParameters gp):
        gp_(gp),
        psip_(gp) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmaxcut + 4.*gp_.alpha) return 0.;
        if( psip_(R,Z) < (gp_.psipmaxcut)) return 1.;
        return exp( -( psip_(R,Z)-gp_.psipmaxcut)*( psip_(R,Z)-gp_.psipmaxcut)/2./gp_.alpha/gp_.alpha);
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Damps the inner boundary in a zone 
 * from psipmax to psipmax+ 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
 *
 * \f$ 0 \f$, if \f$ \psi_p(R,Z) > \psi_{p,max} + 4\alpha \f$
 *
 * \f$ 1 \f$, if \f$ \psi_p(R,Z) < \psi_{p,max}\f$
 *
 * \f$ \exp\left( - \frac{(\psi_p - \psi_{p,max})^2}{2\alpha^2}\right)\f$, else
 */ 
struct GaussianProfDamping
{
    GaussianProfDamping( GeomParameters gp):
        gp_(gp),
        psip_(gp) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax ) return 0.;
        if( psip_(R,Z) < (gp_.psipmax-4.*gp_.alpha)) return 1.;
        return exp( -( psip_(R,Z)-(gp_.psipmax-4.*gp_.alpha))*( psip_(R,Z)-(gp_.psipmax-4.*gp_.alpha))/2./gp_.alpha/gp_.alpha);
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Damps the inner boundary in a zone 
 * from psipmax to psipmax+ 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
 * Additionaly cuts if Z < Z_xpoint
 *
 * \f$ 0 \f$, if \f$ \psi_p(R,Z) > \psi_{p,max} + 4\alpha \f$
 *
 * \f$ 1 \f$, if \f$ \psi_p(R,Z) < \psi_{p,max}\f$
 *
 * \f$ \exp\left( - \frac{(\psi_p - \psi_{p,max})^2}{2\alpha^2}\right)\f$, else
 */ 
struct GaussianProfXDamping
{
    GaussianProfXDamping( GeomParameters gp):
        gp_(gp),
        psip_(gp) {
        }
    double operator( )(double R, double Z)
    {
     if( psip_(R,Z) > gp_.psipmax || Z<-1.1*gp_.elongation*gp_.a) return 0.;
        if( psip_(R,Z) < (gp_.psipmax-4.*gp_.alpha)) return 1.;
        return exp( -( psip_(R,Z)-(gp_.psipmax-4.*gp_.alpha))*( psip_(R,Z)-(gp_.psipmax-4.*gp_.alpha))/2./gp_.alpha/gp_.alpha);
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};



// struct Gradient
// {
//     Gradient(  eule::Parameters p, GeomParameters gp):
//         p_(p),
//         gp_(gp),
//         psip_(gp) {
//     }
//     double operator( )(double R, double Z)
//     {
//         if( psip_(R,Z) < (gp_.psipmin)) return p_.nprofileamp+p_.bgprofamp;
//         if( psip_(R,Z) < 0.) return p_.nprofileamp+p_.bgprofamp-(gp_.psipmin-psip_(R,Z))*(p_.nprofileamp/gp_.psipmin);
//         return p_.bgprofamp;
//     }
//     double operator( )(double R, double Z, double phi)
//     {
//         return (*this)(R,Z);
// 
//     }
//     private:
//     eule::Parameters p_;
//     GeomParameters gp_;
//     Psip psip_;
// };

/**
 * @brief Returns density profile with variable peak amplitude and background amplitude 
 *
 * \f$ N(R,Z) =  A_{bg} + A_{peak}\frac{\psi_p} {\psi_p(R_0, 0)} \f$, for \f$\psi_p < \f$\psi_p_max\f$ 
 *
 * \f$ N(R,Z) =  A_{bg} \f$, else
 */ 
struct Nprofile
{
     Nprofile( eule::Parameters p, GeomParameters gp):
        p_(p),
        gp_(gp),
        psip_(gp) {
        }
   double operator( )(double R, double Z)
    {
        if (psip_(R,Z)<gp_.psipmax) return p_.bgprofamp +(psip_(R,Z)/psip_(gp_.R_0,0.0)*p_.nprofileamp);
        return p_.bgprofamp;
    }
    double operator( )(double R, double Z, double phi)
    {
        return (*this)(R,Z);
    }
    private:
    eule::Parameters p_;
    GeomParameters gp_;
    Psip psip_;
};

/**
 * @brief returns zonal flow field 
 */ 
struct ZonalFlow
{
    ZonalFlow(  eule::Parameters p,GeomParameters gp):
        p_(p),
        gp_(gp),
        psip_(gp) {
    }
    double operator() (double R, double Z)
    {
//       if (psip_(R,Z)<gp_.psipmax) return p_.amp*sin(M_PI*psip_(R,Z)*psip_(R,Z));
        return p_.amp*psip_(R,Z)*psip_(R,Z);
//         return p_.amp*(1.+sin(M_PI*psip_(R,Z)-M_PI/2.));
//         return  p_.amp*exp(-(psip_(R,Z)-0.5)*(psip_(R,Z)-0.5)/0.01);
    }
    double operator() (double R, double Z,double phi)
    {
        return (*this)(R,Z);
    }
    private:
    eule::Parameters p_;
    GeomParameters gp_;
    Psip psip_;
};

/**
 * @brief testfunction to test the parallel derivative \f[ f = \psi_p(R,Z) \sin{(\varphi)}\f]
 */ 
/*
struct TestFunction
{
    TestFunction( eule::Parameters p,GeomParameters gp) :  
        p_(p),
        gp_(gp),
        bhatR_(gp),
        bhatZ_(gp),
        bhatP_(gp) {}
    double operator()( double R, double Z, double phi)
    {
//         return psip_(R,Z,phi)*sin(phi);
//         double Rmin = gp_.R_0-(p_.boxscaleRm)*gp_.a;
//         double Rmax = gp_.R_0+(p_.boxscaleRp)*gp_.a;
//         double kR = 1.*M_PI/(Rmax - Rmin);
//         double Zmin = -(p_.boxscaleZm)*gp_.a*gp_.elongation;
//         double Zmax = (p_.boxscaleZp)*gp_.a*gp_.elongation;
//         double kZ = 1.*M_PI/(Zmax - Zmin);
        double kP = 1.;
//         return sin(phi*kP)*sin((R-Rmin)*kR)*sin((Z-Zmin)*kZ); //DIR
//         return cos(phi)*cos((R-Rmin)*kR)*cos((Z-Zmin)*kZ);
//         return sin(phi*kP); //DIR
//         return cos(phi*kP); //NEU
                return -cos(phi*kP)/bhatP_(R,Z,phi)/R; //NEU 2

    }
    private:
    eule::Parameters p_;
    GeomParameters gp_;
    BHatR bhatR_;
    BHatZ bhatZ_;
    BHatP bhatP_;
};
*/
/**
 * @brief analyitcal solution of the parallel derivative of the testfunction
 *  \f[ \nabla_\parallel f = \psi_p(R,Z) b^\varphi \cos{(\varphi)}\f]
 */ 
/*
struct DeriTestFunction
{
    DeriTestFunction( eule::Parameters p, GeomParameters gp) :
        p_(p),
        gp_(gp),
        bhatR_(gp),
        bhatZ_(gp),
        bhatP_(gp) {}
    double operator()( double R, double Z, double phi)
    {
//         double Rmin = gp_.R_0-(p_.boxscaleRm)*gp_.a;
//         double Rmax = gp_.R_0+(p_.boxscaleRp)*gp_.a;
//         double kR = 1.*M_PI/(Rmax - Rmin);
//         double Zmin = -(p_.boxscaleZm)*gp_.a*gp_.elongation;
//         double Zmax = (p_.boxscaleZp)*gp_.a*gp_.elongation;
//         double kZ = 1.*M_PI/(Zmax - Zmin);
        double kP = 1.;
//          return (bhatR_(R,Z,phi)*sin(phi)*sin((Z-Zmin)*kZ)*cos((R-Rmin)*kR)*kR+
//                 bhatZ_(R,Z,phi)*sin(phi)*sin((R-Rmin)*kR)*cos((Z-Zmin)*kZ)*kZ+
//                 bhatP_(R,Z,phi)*cos(phi)*sin((R-Rmin)*kR)*sin((Z-Zmin)*kZ)*kP); //DIR
//         return -bhatR_(R,Z,phi)*cos(phi)*cos((Z-Zmin)*kZ)*sin((R-Rmin)*kR)*kR-
//                bhatZ_(R,Z,phi)*cos(phi)*cos((R-Rmin)*kR)*sin((Z-Zmin)*kZ)*kZ-
//                bhatP_(R,Z,phi)*sin(phi)*cos((R-Rmin)*kR)*cos((Z-Zmin)*kZ)*kP;
//         return  bhatP_(R,Z,phi)*cos(phi*kP)*kP; //DIR
//         return  -bhatP_(R,Z,phi)*sin(phi*kP)*kP; //NEU
                return sin(phi*kP)*kP/R; //NEU 2

    }
    private:
    eule::Parameters p_;
    GeomParameters gp_;
    BHatR bhatR_;
    BHatZ bhatZ_;
    BHatP bhatP_;
};*/


///@}
}//namespace solovev

